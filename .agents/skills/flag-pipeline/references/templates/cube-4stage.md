# Cube 四级流水模板（MTE2 / MTE1 / M / FIX）

纯 Cube 算子（GEMM、MatMul、Linear）的标准模板，对齐 `examples/gemm/example_gemm_intrinsic.py` 的真实 id 写法。所有四级流水 kernel 照此模板。

> 本模板的 id 分配和时序对齐原版 GEMM intrinsic。示例默认 `K_L1=256, block_K=64`，但归还 L1 slot 的条件必须写成 `kk == loop_kk - 1`，不要把 `kk == 3` 当成通用规则。

## 1. 参数与缓冲

```python
S1 = 2  # L1 双缓冲深度（推荐 2-4）
S2 = 2  # L0 双缓冲深度（几乎总 2）

A_L1 = T.alloc_L1((S1, block_M, K_L1), dtype)       # L1 双缓冲
B_L1 = T.alloc_L1((S1, K_L1, block_N), dtype)
A_L0 = T.alloc_L0A((S2, block_M, block_K), dtype)   # L0 双缓冲
B_L0 = T.alloc_L0B((S2, block_K, block_N), dtype)
C_L0 = T.alloc_L0C((block_M, block_N), accum_dtype) # L0C 单缓冲
```

两层 K：

- 外层 `K_L1`（如 256）：GM → L1 一次搬一大片
- 内层 `block_K`（如 64）：L1 → L0 分块搬

## 2. 信号 ID 表

| ID | 管线对 | 含义 |
|---|---|---|
| 0 | MTE1 → MTE2 | L1 slot 0 可被 MTE2 覆盖写 |
| 1 | MTE1 → MTE2 | L1 slot 1 可被 MTE2 覆盖写 |
| 0 | M → MTE1 | L0 slot 0 可被 MTE1 重用 |
| 1 | M → MTE1 | L0 slot 1 可被 MTE1 重用 |
| 0 | FIX → M | L0C 可被 M 重用（单缓冲，一个 id） |

MTE1→MTE2 用 id 0,1；M→MTE1 也用 id 0,1。这两组是不同 `(src,dst)` 信箱，id 复用不冲突。FIX→M 用 id 0，又一个独立信箱。

## 3. init_flag / clear_flag（信用借贷）

```python
@T.macro
def init_flag():
    # 借 L1 两个 slot（让 MTE2 第一拍 wait MTE1→MTE2 通过）
    T.set_flag("mte1", "mte2", 0)
    T.set_flag("mte1", "mte2", 1)
    # 借 L0 两个 slot（让 MTE1 第一拍 wait M→MTE1 通过）
    T.set_flag("m", "mte1", 0)
    T.set_flag("m", "mte1", 1)
    # 借 L0C（让 M 第一拍 wait FIX→M 通过）
    T.set_flag("fix", "m", 0)

@T.macro
def clear_flag():
    # 还：借几个还几个，id 对齐
    T.wait_flag("mte1", "mte2", 0)
    T.wait_flag("mte1", "mte2", 1)
    T.wait_flag("m", "mte1", 0)
    T.wait_flag("m", "mte1", 1)
    T.wait_flag("fix", "m", 0)
```

借还对称：init 借 5 个（2+2+1），clear 必须还 5 个，id 完全对齐。

## 4. 主循环（对齐原版）

```python
with T.Scope("C"):
    init_flag()

    for i in T.serial(T.ceildiv(m_num * n_num, core_num)):
        # ... 任务映射 (bx, by) ...

        loop_k = T.ceildiv(K, K_L1)

        # === 第一片 L1 预装（循环外） ===
        T.wait_flag("mte1", "mte2", 0)                    # 等 MTE1 释放 L1 slot 0
        T.copy(A[bx * block_M, 0], A_L1[0, :, :])
        T.copy(B[0, by * block_N], B_L1[0, :, :])
        T.set_flag("mte2", "mte1", 0)                     # 通知 MTE1：L1 slot 0 有数据

        # === 第一拍 M 等 FIX 释放 L0C（借的信用） ===
        T.wait_flag("fix", "m", 0)

        for k in T.serial(loop_k):
            # ---------- MTE2: 预取下一片到 L1 ----------
            if k < loop_k - 1:
                T.wait_flag("mte1", "mte2", (k + 1) % S1)
                T.copy(A[bx * block_M, (k + 1) * K_L1], A_L1[(k + 1) % S1, :, :])
                T.copy(B[(k + 1) * K_L1, by * block_N], B_L1[(k + 1) % S1, :, :])
                T.set_flag("mte2", "mte1", (k + 1) % S1)

            # ---------- MTE1 + M: L1 → L0 → MMA ----------
            loop_kk = T.ceildiv(K_L1, block_K)
            for kk in T.serial(loop_kk):
                # MTE1: L1 → L0
                if kk == 0:
                    T.wait_flag("mte2", "mte1", k % S1)      # 等 MTE2 装好 L1（仅 kk=0）
                T.wait_flag("m", "mte1", kk % S2)            # 等 M 释放 L0 slot
                T.copy(A_L1[k % S1, 0, kk * block_K], A_L0[kk % S2, :, :])
                T.copy(B_L1[k % S1, kk * block_K, 0], B_L0[kk % S2, :, :])
                if kk == loop_kk - 1:                         # 最后一片 L1→L0 后归还 L1 slot
                    T.set_flag("mte1", "mte2", k % S1)
                T.set_flag("mte1", "m", kk % S2)              # 通知 M：L0 就绪
                T.wait_flag("mte1", "m", kk % S2)             # 等 MTE1 搬完，M 才能读 L0

                # M: MMA
                T.mma(A_L0[kk % S2, :, :], B_L0[kk % S2, :, :], C_L0,
                      init=T.And(k == 0, kk == 0))
                T.set_flag("m", "mte1", kk % S2)              # 归还 L0 slot

        # ---------- FIX: L0C → GM ----------
        # 注意：GEMM 的 C_L0 跨完整 K loop 累加，FIX 只能在所有 K 分块完成后写回一次。
        T.set_flag("m", "fix", 0)                             # 通知 FIX：L0C 可读
        T.wait_flag("m", "fix", 0)                            # 等 M 生产 L0C 完成
        T.copy(C_L0, C[bx * block_M, by * block_N])
        T.set_flag("fix", "m", 0)                             # FIX 写回完成，归还 L0C

    clear_flag()
```

## 5. 逐阶段解读

### 阶段 1：MTE2（GM → L1）

```python
T.wait_flag("mte1", "mte2", (k+1) % S1)    # 等 MTE1 用完 L1 这个 slot
T.copy(A[..., (k+1)*K_L1], A_L1[(k+1)%S1]) # GM → L1
T.set_flag("mte2", "mte1", (k+1) % S1)     # 通知 MTE1：L1 slot 有数据了
```

### 阶段 2：MTE1（L1 → L0A/L0B）

```python
T.wait_flag("mte2", "mte1", k % S1)        # 等 MTE2 把 L1 装好
T.wait_flag("m", "mte1", kk % S2)          # 等 M 用完 L0 这个 slot
T.copy(A_L1[k%S1, ...], A_L0[kk % S2])     # L1 → L0A
T.set_flag("mte1", "m", kk % S2)           # 通知 M：L0 有数据了
```

### 阶段 3：M（L0A · L0B → L0C）

```python
T.wait_flag("mte1", "m", kk % S2)          # 等 MTE1 把 L0 装好
T.mma(A_L0[kk % S2], B_L0[kk % S2], C_L0)  # 矩阵乘累加
T.set_flag("m", "mte1", kk % S2)           # 通知 MTE1：L0 可重用
```

### 阶段 4：FIX（L0C → GM）

```python
T.set_flag("m", "fix", 0)                  # 建立 M→FIX 依赖：L0C 可读
T.wait_flag("m", "fix", 0)                 # 等 M 生产 L0C 完成
T.copy(C_L0, C[...])                       # L0C → GM
T.set_flag("fix", "m", 0)                  # 通知 M：L0C 可重用
```

FIX 阶段的紧邻 `set("m","fix",0)→wait("m","fix",0)`：这是在发起 FIX copy 前等待 M 管线把 L0C 生产完成，确保 FIX 读到完整 C。FIX 写回 GM 后，必须显式 `set_flag("fix", "m", 0)` 释放 L0C 给后续 M 使用；不要把 `wait("m","fix",0)` 理解为等待 FIX 自动回信。

## 6. 四级流水时序图

预热后 4 条管线真正并行：

```
时间 →   t0     t1     t2     t3     t4     t5     t6     ...
MTE2:   K0    K1    K2    K3    K4    K5    K6    ...
MTE1:         L0_0  L0_1  L0_2  L0_3  L0_4  L0_5  ...
M:                 mma0  mma1  mma2  mma3  mma4  ...
FIX:                      ws0   ws1   ws2   ws3   ws4   ...
```

对比 barrier_all 串行：

```
MTE2:   [K0]              [K1]              [K2]
M:           [mma0]             [mma1]             [mma2]
FIX:              [ws0]              [ws1]              [ws2]
```

吞吐量约 4 倍。

## 7. 原版特殊写法（照搬，别改）

模板里有几处原版的"非显然"写法，照搬即可：

1. **第一片 L1 预装在循环外**（原版 L69-72）：`wait_flag("mte1","mte2",0)` + copy + `set_flag("mte2","mte1",0)`，然后才进循环。让循环内 `kk==0` 时的 `wait_flag("mte2","mte1",k%S1)` 第一拍有数据。
2. **`wait_flag("fix","m",0)` 在循环外**（原版 L73）：借 init 的信用，让第一拍 M 的 mma 不阻塞。
3. **`kk==0` 时才 wait MTE2→MTE1**（原版 L85）：一片 L1 被多个 kk 复用，只需第一次等 MTE2 装好。
4. **最后一个 `kk` 时归还 L1 slot**：写成 `kk == loop_kk - 1`。只有 `K_L1=256, block_K=64` 时才等价于 `kk==3`。
5. **FIX 写回在完整 K loop 之后**：`C_L0` 跨所有 K 分块累加，不能每个 `k` 分块都写回。
6. **FIX 阶段 `set("m","fix",0)→wait("m","fix",0)` 紧邻**：见 §5 注释，FIX 完成后仍需显式 `set("fix","m",0)` 释放 L0C。

## 8. 适配 Checklist

- [ ] `block_M × block_N × sizeof(accum) ≤ 128KB`（L0C 容量）
- [ ] `S2 × block_M × block_K × sizeof(dtype) ≤ 64KB`（L0A）
- [ ] `S1 × (block_M + block_N) × K_L1 × sizeof(dtype) + extra_L1_buffers ≤ 512KB`（A/B 共享 L1）
- [ ] init_flag 借的 id 覆盖循环里所有 wait 的 slot
- [ ] clear_flag 的 wait 数 = init_flag 的 set 数
- [ ] 归还 L1 slot 使用 `kk == loop_kk - 1`，不要硬编码 `kk==3`
- [ ] FIX 写回位于完整 `for k` 循环之后，只写回最终 C_L0

## 9. 参考

- 源码：`examples/gemm/example_gemm_intrinsic.py`
- 持续化版（多任务循环）：`examples/gemm/example_gemm_intrinsic_persistent.py`
- 机制原理：[flag-mechanism.md](../flag-mechanism.md)
- 硬件管线：[hardware.md](../hardware.md)
