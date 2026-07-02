# 调试方法论：flag 配对验证 → 删减定位 → msprof 看 overlap

flag 流水出问题（死锁/数据错/性能差）时的系统化调试流程。不是盲目试错，而是二分定位。

```
出问题
  ├─ 死锁/卡住 ──→ §2 flag 配对验证 ──→ §3 删减定位（找哪步开始卡）
  ├─ 数据错乱 ──→ §4 检查容量/覆盖/cross_interval
  └─ 性能差 ──→ §5 msprof 看 stall ratio
```

## 1. 先确认环境

死锁或性能异常前先确认 pass_configs 四件套全 False（`auto_sync` / `auto_cv_sync` / `auto_cv_combine` / `memory_planning`）。没关的话 pass 自动插 flag 和手写的冲突，可能死锁也可能数据错。配置代码见 [SKILL.md Step 4](../SKILL.md)，语义见 [pitfalls.md §3](pitfalls.md)。

## 2. flag 配对验证

写完代码或遇到死锁时，按以下步骤逐 flag 验证，不需要跑硬件。

### 2.1 逐项 checklist

逐个 `(src, dst, eid)` 三元组检查：

1. **init 借了吗？** —— scope 入口、循环之前，有没有 set 第一个 wait 需要的信用？
2. **clear 还了吗？** —— 循环结束后、scope 退出前，有没有 wait 回来 init 借的？id 对齐吗？
3. **set/wait 属于三类合法模式吗？** —— 模式 A（严格配对）/ B（纯广播）/ C（混合节流），定义和示例见 [flag-mechanism.md §3](flag-mechanism.md)。
4. **src/dst 方向一致吗？** —— set 的 src 是写完数据的管线，wait 的 dst 是消费数据的管线。比如 MTE2→MTE1 方向：set("mte2","mte1",eid) 配 wait("mte2","mte1",eid)。
5. **cross_flag 的 pipe 对吗？** —— Cube 侧用 `"FIX"`，Vector 侧用 `"MTE3"`。
6. **cross_flag 走 cross_flag API 了吗？** —— C↔V 不用核内 set_flag/wait_flag。
7. **cross_interval 末次 set 了吗？** —— 条件必须含 `or i == N-1`。

### 2.2 逐 flag 统计 count

对每个 flag id 统计 set/wait 数量和位置，确认符合对应模式：

```
flag 0 (MTE2→MTE1, L1 buf0):
  init:   set × 1 (借信用)
  循环内: set × N (kk%S1==0 时), wait × N (每次 kk)
  clear:  wait × 1 (还信用)
  总计:   set N+1, wait N+1 → 模式 A ✓
```

模式 C 的 cross_flag 会 set < wait（init 1 + 节流 ⌈N/k⌉ vs wait N），这是合法的，统计 count 时要按模式 C 确认 init 和节流条件是否正确。

### 2.3 @T.macro 展开验证

`init_flag()` / `clear_flag()` 如果用 `@T.macro` 定义，将宏体 inline 展开到调用点后再验证：展开后的 set/wait 顺序是否在正确的位置（scope 入口/循环前/循环后/scope 退出前）。

宏只是文本替换，不会改变执行位置。宏定义在外层不代表 set 在外层执行——取决于调用点在哪。

## 3. 删减定位：二分找哪步卡

死锁了，配对验证没发现问题，按此方法找根因。

核心思路：**把代码逐步删减到最简，每减一步验证是否还死锁，找到导致死锁的最小代码段。**

### 3.1 步骤

1. **先去掉 Vector scope（CV 融合时）**：只保留 C scope（Cube 四级流水），看能不能跑通
   - 能跑 → 问题在 V 或 cross_flag，恢复 V 继续二分
   - 不能跑 → 问题在 C scope 内部

2. **去掉 cross_flag**：如果 CV 双 scope，先注释掉所有 set_cross_flag/wait_cross_flag，每个 scope 内部跑自己的循环（V 不读 workspace，C 不写 workspace 给 V）
   - 能跑 → cross_flag 配对有问题，逐个加回 cross_flag
   - 不能跑 → 问题在核内 flag

3. **核内 flag 逐个去掉**：从 FIX→M→MTE1→MTE2 方向（反数据流方向）逐个注释 flag 对
   - 注释到某对 flag 时不死锁了 → 就是那对 flag 的问题

4. **单 flag 对定位**：定位到具体某对 flag 后，检查：
   - init 有没有借信用
   - set 和 wait 的 src/dst/eid 是否一致
   - wait 是否在 set 可能到达之前执行（set 在 copy 之后、wait 在 copy 之前）

### 3.2 删减原则

- 删减后代码必须语法正确、能编译
- 删减的 flag 对改成"数据依赖自然满足"的串行（即暂时退化为无流水）
- 每次只改一处，改完验证
- **保留 buffer alloc 和 T.copy**，只删 flag 操作，否则容量问题也会导致 crash 干扰定位

### 3.3 加打印辅助

在关键位置加 T.print 或 host 侧 printf（如果支持），确认执行到哪一步：
- 循环开始前打一行
- 每次 wait 之后打一行（确认过了几个 wait）
- 循环结束打一行

如果打印停在"等 cross_flag SEM_WS1_V2C"且 V 侧打印也停在 wait，就是双向死锁。

## 4. 数据错乱排查

不死锁但结果错，常见原因：

1. **cross_interval > num_stages**：workspace 不够，数据被覆盖。见 [pitfalls.md §1.6](pitfalls.md)。
2. **wait 的时机不对**：wait 之后才写数据，或 set 之前就读数据——即 flag 方向反了。
3. **pipe 参数错**：set_cross_flag 的 pipe 和实际写 GM 的 pipe 不一致，信号提前发出但数据还没写完。Cube 侧必须 FIX 写完 GM 再 set_cross_flag("FIX",...)，Vector 侧必须 MTE3 写完 GM 再 set_cross_flag("MTE3",...)。
4. **buffer 轮转错误**：ping-pong index `% stages` 算错，读写同一片 buffer。
5. **clear_flag 位置错**：在 V 还没读完 workspace 时 C 就 clear 了 wait（把信用还了但覆盖了数据），或反之。

### 快速验证法

- 关掉双缓冲（S1=1, S2=1, num_stages=1），退化为串行，如果结果正确 → 问题在 flag 配对或轮转
- 串行都不对 → 问题在计算逻辑本身，和 flag 无关

## 5. msprof 性能分析

能跑且结果对，但性能差，用 msprof 看各管线的时间线。

### 5.1 采集

```bash
msprof op --output=./prof_data --application="python your_script.py"
```

### 5.2 看什么

打开 msprof 查看器，重点看：

1. **时间线是否 overlap**：MTE2/MTE1/M/FIX 应该各自是连续的条带，互相重叠。如果看到大段空白（某管线空等），说明那对 flag 节流过头了。
2. **Cube (M) 占比**：Cube 计算时间应该占总时间 60%+，如果 Cube 经常空等 MTE1 或 FIX，说明 MTE2/MTE1 流水没跟上。
3. **跨核同步 gap**：CV 融合时看 C 和 V 之间的 wait_cross_flag 是否有大 gap。如果 C 等 V 很久，说明 Vector 侧处理太慢或 cross_interval 太小。

### 5.3 注意

- **不要用 Python time.time() 计时**，NPU 异步执行，Python 计时包含 host 开销不准。用 msprof 或 `torch.npu.Event` 计时。
- 先跑小 shape 确认功能正确，再跑大 shape 看性能。

## 6. 纯 flag 硬件验证

当配对验证和删减定位都难以定位时，写一个最小纯 flag kernel——只保留 flag 操作和空循环（去掉 T.copy/T.gemm 等计算），在板子上跑：

- **不死锁** → flag 配对本身没问题，bug 在计算逻辑或 buffer 操作
- **死锁** → flag 配对本身有错，继续二分 flag 对

这是最快的死锁验证手段，因为纯 flag kernel 编译快、运行快，能在秒级确认 flag 拓扑是否正确。

最小示例结构：
```python
@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def kernel():
    @T.prim_func
    def main(A: T.Tensor(...)):
        with T.Kernel(1, is_npu=True) as (bid, vid):
            with T.Scope("C"):
                T.set_flag("mte2", "mte1", 0)  # init
                for kk in T.serial(NUM_K):
                    T.wait_flag("mte2", "mte1", kk % S1)
                    T.set_flag("mte1", "m", kk % S2)
                T.wait_flag(...)  # clear
            with T.Scope("V"):
                T.set_cross_flag("MTE3", SEM_V2C)  # init 借信用
                for i in T.serial(batch_iters):
                    T.wait_cross_flag(SEM_C2V)
                    T.set_cross_flag("MTE3", SEM_V2C)
```

去掉所有计算和 buffer 操作后，如果还死锁，问题就锁定在 flag 拓扑本身。

## 7. 常见症状 → 根因速查

| 症状 | 最可能根因 |
|------|----------|
| kernel 启动即 hang（第一拍） | 漏 init_flag 或 init 没在循环前 |
| 跑一会儿 hang（中途死锁） | set/wait 不配对或 cross_flag 节流条件错 |
| kernel 跑完下个 kernel 死锁 | 漏 clear_flag 或 clear 不对称 |
| 不死锁但 Illegal instruction | Vector 用了 "FIX" pipe |
| 不死锁但结果全错/NaN | buffer 覆盖（cross_interval > num_stages）或 pipe 参数错 |
| 能跑但性能和单缓冲差不多 | 混了 barrier_all 或 S1/S2=1 |
| Cube stall 等 Vector | cross_interval=1 节流不够，或 Vector 处理太慢 |
| msprof 看到某管线大段空白 | 该管线的 wait flag 没及时 set，前序管线是瓶颈 |

## 8. 调试纪律

1. **不要凭感觉改 flag**。每次只改一处，改完验证，记录改动。
2. **从模板出发，不要自创**。三种合法模式覆盖了几乎所有场景。
3. **配对验证不过不要上板**。init/clear/pair/count 逐项验证通过再跑硬件。
4. **先验证功能再优化性能**。跑通、结果对，再调 S1/S2/cross_interval 等参数。
5. **保留最小复现**。找到死锁原因前不要删掉触发死锁的代码，留着做回归验证。
