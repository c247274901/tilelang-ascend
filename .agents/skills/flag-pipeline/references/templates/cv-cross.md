# CV 跨核流水模板（Cube ↔ Vector cross_flag）

CV 融合算子（FlashAttention、matmul+activation、GEMM+softmax+GEMM）的标准模板，对齐 `examples/flash_attention/fa_opt/flash_attn_bhsd_expert_h16_d128.py` 的 expert 范本（pass_configs 全 False）。

> 本 skill 最复杂的模板。读前先看 [hardware.md §物理拓扑](../hardware.md)（C0/V0/V1/FFTS）和 [flag-mechanism.md §三类合法模式](../flag-mechanism.md)。

## 1. CV 协作动机

以 FlashAttention 为例，主体是两个 GEMM + 一个 softmax：

```
GEMM1 (Cube):  S = Q @ K^T      ← 矩阵乘，适合 Cube
Softmax (Vec): P = softmax(S)    ← 逐行归一化，适合 Vector
GEMM2 (Cube):  O = P @ V         ← 矩阵乘，适合 Cube
```

全在 Cube 上做会占用 Cube 的 M 单元做向量运算，严重浪费。解法：Cube 做 GEMM，Vector 做 softmax，通过 workspace 交接。

## 2. Workspace 交接机制

3 个 workspace 在 GM 上，Cube 写、Vector 读（或反过来）：

```
Cube (GEMM1) → ws1 (S) → Vector (Softmax) → ws2 (P) → Cube (GEMM2) → ws3 (O) → Vector (累加)
```

每个 workspace 有 C2V 和 V2C 两个方向的 cross_flag，形成握手：

```
Cube 写 ws1 → set C2V → Vector wait C2V → Vector 读 ws1
Vector 读完 → set V2C → Cube wait V2C → Cube 写下一轮 ws1
```

两个方向的原因：workspace 是共享资源，Cube 写时 Vector 不能读，Vector 读时 Cube 不能写。两个方向的 flag 形成生产者-消费者双向握手，保证 workspace 不被同时读写。

## 3. 6 个核间信号 ID（FA 的分配）

```python
SEM_WS1_C2V = 0   # ws1: Cube 写完，Vector 可读
SEM_WS1_V2C = 1   # ws1: Vector 读完，Cube 可写
SEM_WS2_V2C = 2   # ws2: Vector 写完，Cube 可读
SEM_WS2_C2V = 3   # ws2: Cube 读完，Vector 可写
SEM_WS3_C2V = 4   # ws3: Cube 写完，Vector 可读
SEM_WS3_V2C = 5   # ws3: Vector 读完，Cube 可写
```

命名规律：`SEM_WS<编号>_<方向>`。C2V = Cube 发给 Vector，V2C = Vector 发给 Cube。id 0-5，核间 eid 独立于核内 eid（核内另有 0-7）。

## 4. cross_interval 节流（模式 C）

### 4.1 动机

每次迭代都 cross_flag，Cube 要停下来等 Vector，核间同步开销大。

### 4.2 解法

每 cross_interval 次才同步一次：

```python
if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
    T.set_cross_flag("FIX", SEM_WS1_C2V)
```

`cross_interval=2` 表示每 2 次迭代才 set 一次 C2V。Vector 侧对应：

```python
if i % cross_interval == 0:
    T.wait_cross_flag(SEM_WS1_C2V)
```

### 4.3 容量前提（硬约束）

cross_interval=N 要求 workspace 能容纳 N 个版本：

```python
workspace_1: T.Tensor([NUM_CORES, num_stages, block_M, block_N], dtype)
#                                ^^^^^^^^^^
#                                这个维度 ≥ cross_interval
```

否则第 i 次写的数据还没被消费，第 i+1 次又要写同一个 slot，数据竞争。

### 4.4 模式 C 实例

```python
# Cube 侧 init 借信用（1 个 set）
T.set_cross_flag("MTE2", SEM_WS1_V2C)

for i in T.serial(batch_iters):
    # ... Cube 写 ws1 ...
    if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
        T.set_cross_flag("FIX", SEM_WS1_C2V)   # 循环内节流 set
```

set 数 = 1（init）+ ⌈N/cross_interval⌉（循环内），wait 数 = N（Vector 侧每次 wait）。set < wait，但合法——即 [flag-mechanism.md §模式 C](../flag-mechanism.md)。关键确认点：init 借了信用、节流条件含 `or i==N-1`、workspace num_stages ≥ cross_interval。

## 5. Cube 侧模板（GEMM1 → ws1）

```python
with T.Scope("C"):
    # === init: 借核间信用 + 核内信用 ===
    T.set_cross_flag("MTE2", SEM_WS2_C2V)   # 借：ws2 Vector 已读，Cube 第一轮可写
    # 核内四级流水的 init_flag() 见 cube-4stage.md

    for i in T.serial(batch_iters):
        side = i % 2   # L0 双缓冲轮转

        # === MTE2/MTE1/M: 四级流水做 GEMM1 ===
        # (照搬 cube-4stage.md 的四级 flag，此处省略)
        # ... T.mma(l0a[side], l0b[side], l0c[side], init=...) ...

        # === FIX: L0C → ws1 ===
        T.set_flag("m", "fix", SIG_L0C + side)
        T.wait_flag("m", "fix", SIG_L0C + side)
        T.copy(l0c[side], workspace_1[cid, i, :, :])    # L0C → ws1
        T.set_flag("fix", "m", SIG_L0C + side)

        # === 核间:每 cross_interval 次通知 Vector 读 ws1 ===
        if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
            T.set_cross_flag("FIX", SEM_WS1_C2V)

    # === clear: 等 Vector 消费完最后一批 ws1/ws3 ===
    # Cube 侧 clear 要等 V2C 的最后信号，确认 Vector 读完了 workspace
    T.wait_cross_flag(SEM_WS1_V2C)
    T.wait_cross_flag(SEM_WS3_V2C)
```

关键点：

- `set_cross_flag("FIX", SEM_WS1_C2V)`——pipe 是 `"FIX"`，因为 Cube 写 GM 走 L0C→GM 的 FIX 通路（见 [hardware.md §pipe 参数](../hardware.md)）。这个 set 必须在 FIX copy（L0C→ws1）**之后**，否则信号先发、数据后到 → 数据错乱。
- 节流条件 `(i+1) % cross_interval == 0 or i == N-1`：末次必须 set，保证最后一批数据被 Vector 消费。
- clear 段等 `SEM_WS1_V2C` 和 `SEM_WS3_V2C`：确认 Vector 读完了最后一批 ws1/ws3，workspace 不再被占用才退出 scope。不等的话 Cube 提前退出可能和 Vector 还在进行的 GM 读冲突。

## 6. Vector 侧模板（ws1 → Softmax → ws2）

```python
with T.Scope("V"):
    # === init: 借核间信用 ===
    T.set_cross_flag("MTE2", SEM_WS1_V2C)   # V→C: ws1 Cube 可写
    T.set_cross_flag("MTE2", SEM_WS3_V2C)   # V→C: ws3 Cube 可写

    for i in T.serial(batch_iters):
        cur = i % 2
        prv = 1 - cur

        # === MTE2: ws1(GM) → UB ===
        T.wait_flag("v", "mte2", SIG_IO_UB)
        if i % cross_interval == 0:                      # 节流：每 cross_interval 次 wait
            T.wait_cross_flag(SEM_WS1_C2V)               # 等 Cube 写完 ws1
        T.copy(workspace_1[cid, i, ...], io_buf)
        T.set_flag("mte2", "v", SIG_IO_UB)

        # === V: softmax 计算 ===
        T.wait_flag("mte2", "v", SIG_IO_UB)
        # ... reduce_max / mul / exp / ... 见 FA 源码 ...
        T.set_flag("v", "mte2", SIG_IO_UB)

        # === MTE3: UB → ws2(GM) ===
        T.wait_flag("mte3", "v", SIG_S_HALF)
        T.copy(work_ub, acc_s_half)
        T.set_flag("v", "mte3", SIG_S_HALF)
        T.wait_flag("v", "mte3", SIG_S_HALF)
        T.copy(acc_s_half, workspace_2[cid, i, ...])     # UB → ws2
        T.set_flag("mte3", "v", SIG_S_HALF)

        # === 核间:每 cross_interval 次通知 Cube 读 ws2 ===
        if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
            T.set_cross_flag("MTE3", SEM_WS2_V2C)        # pipe=MTE3，Vector 写 GM 走 MTE3

    # === clear ===
    # Vector 侧不需要显式 wait cross_flag，循环结束后所有信号已消费
```

关键点：

- `set_cross_flag("MTE3", SEM_WS2_V2C)`——pipe 是 `"MTE3"`，因为 Vector 写 GM 走 UB→GM 的 MTE3 通路。用错成 `"FIX"` 会报 `Illegal instruction`。这个 set 必须在 MTE3 copy（UB→ws2）**之后**。
- Vector 侧的 `wait_cross_flag(SEM_WS1_C2V)` 用 `if i % cross_interval == 0` 节流，和 Cube 侧的 set 节流对齐。wait 在 MTE2 copy（ws1→UB）**之前**，等信号到了才开始读 GM。
- init 借信用的方向要理解清楚：`SEM_WS1_V2C` 是 V→C 方向，意思是"V告诉C：ws1 你可以写"。kernel 启动时 V 先 set 这个，C 第一轮 wait V2C 就不会阻塞。init set 的 pipe 用 `"MTE2"` 因为 V 此时还没做 MTE3 写操作，只是预先发信号。

## 7. 核间 init/clear 信用借贷

### 7.1 init：谁先等谁先借

```python
# Cube 侧 init（C scope 入口、循环前）
T.set_cross_flag("MTE2", SEM_WS2_C2V)   # C→V: ws2 Vector 可写（因为Cube还没读ws2）
# （SEM_WS2_C2V 是C等V写完ws2的信号，C先set表示"ws2我暂时不需要，你先写"）

# Vector 侧 init（V scope 入口、循环前）
T.set_cross_flag("MTE2", SEM_WS1_V2C)   # V→C: ws1 Cube 可写
T.set_cross_flag("MTE2", SEM_WS3_V2C)   # V→C: ws3 Cube 可写
```

init 借信用的原则：**第一轮会先 wait 的那一方，需要对方先 set 一个信用**。
- C 循环第一行就 `wait_cross_flag(SEM_WS1_V2C)`（等V说ws1可写），所以 V 必须 init set SEM_WS1_V2C
- C 也 `wait_cross_flag(SEM_WS3_V2C)`，所以 V 必须 init set SEM_WS3_V2C
- V 循环第一行 `wait_cross_flag(SEM_WS2_C2V)`（等C说ws2可读，其实是等C写完ws2——这里方向需要仔细对照实际代码，FA中ws2是V写C读，所以C init set SEM_WS2_C2V告诉V"ws2你可以写"）

### 7.2 clear：等最后一批完成

cross_flag 的 clear 和核内 flag 不同，不需要 `wait_cross_flag` 回所有 init 借的信用，而是：
- **Cube 侧 clear**：等最后一批 V2C 信号（`SEM_WS1_V2C`、`SEM_WS3_V2C`），确认 V 读完了 C 写的最后一批 workspace，C 才能安全退出。
- **Vector 侧 clear**：通常不需要额外 wait——循环内最后一轮 MTE3 copy 完成 + set_cross_flag 发出后，V 的工作完成了。最后一批 C2V 信号由 C 在 clear 段等。

判断是否需要额外 clear wait：看 kernel 结束时是否有**对方还在等你发信号**。如果循环结构保证最后一轮 set 被对侧 wait 消费了（节流条件含 `or i==N-1`），就不需要额外 clear。

### 7.3 信用流转验证

写完全部 flag 后，对每个 SEM id 按以下步骤验证：
1. init 有没有 set？第一轮 wait 会不会阻塞？
2. 循环内 set 和 wait 的条件对齐吗？（C set 条件 `(i+1)%N==0 or i==N-1` 对应 V wait 条件 `i%N==0`）
3. 末次 i=N-1 有没有 set？最后一批会不会漏信号？
4. clear 段有没有等对侧最后一批信号？

## 8. pass_configs 四件套全 False

手写 CV 流水时四件套全 False（`auto_sync` / `auto_cv_sync` / `auto_cv_combine` / `memory_planning`），手写的 flag 已表达全部同步，pass 再插会重复导致死锁或性能回退。配置代码见 [SKILL.md Step 4](../../SKILL.md)，语义详见 [pitfalls.md §3](../pitfalls.md)。

## 9. CV 流水时序图

cross_interval=2，Cube 和 Vector 异步 overlap：

```
Cube:   write_ws1_0  write_ws1_1  write_ws1_2  write_ws1_3
        ---          wait_V2C     ---          wait_V2C
        set_C2V                   set_C2V
Vector: ---          read_ws1_0   read_ws1_1   read_ws1_2
                       wait_C2V                 wait_C2V
                       set_V2C                  set_V2C
```

Cube 写第 0/1 块时，Vector 在读上一批；两者并行。cross_interval=2 时 Cube 攒 2 个 batch 发一次 C2V，Vector 攒 2 个 batch 处理完发一次 V2C，workspace 需要 2 个 slot。

## 10. 适配 Checklist

- [ ] workspace 维度 `num_stages ≥ cross_interval`
- [ ] 6 个 SEM id 不和核内 id 冲突（核间独立 0-7）
- [ ] Cube 侧 `set_cross_flag` 的 pipe 是 `"FIX"`（在 FIX copy 之后）
- [ ] Vector 侧 `set_cross_flag` 的 pipe 是 `"MTE3"`（在 MTE3 copy 之后）
- [ ] Cube set 节流条件 = Vector wait 节流条件对齐（`(i+1)%N==0 or i==N-1` vs `i%N==0`）
- [ ] 末次必须 set（`or i == N-1`），保证最后一批数据被消费
- [ ] init 借的核间 flag 覆盖了所有第一轮 wait 的方向
- [ ] Cube clear 段等了 V2C 最后信号
- [ ] pass_configs 四件套全 False
- [ ] Cube 内四级流水照搬 [cube-4stage.md](cube-4stage.md)
- [ ] Vector 内三级流水照搬 [vector-3stage.md](vector-3stage.md) 的 prefetch/main/epilogue 结构
- [ ] 对每个 SEM id 验证信用流转（init→循环→clear 无死锁）

## 11. 参考

- 范本：`examples/flash_attention/fa_opt/flash_attn_bhsd_expert_h16_d128.py`
- 最简 CV 协作：`examples/simple_fusion/matmul_add.py`
- 三类合法模式：[flag-mechanism.md §模式 C](../flag-mechanism.md)
- 物理拓扑（C↔V 必须用 cross_flag 的硬件原因）：[hardware.md §FFTS](../hardware.md)
- 常见死锁：[pitfalls.md §1](../pitfalls.md)（漏 init/pipe 错/末次漏 set/cross_interval 超限）
