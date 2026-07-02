# 避坑指南：死锁根因 / 容量约束

手写 flag 容易踩的坑集中列出，每条给现象、根因、解法。

## 1. 死锁类

### 1.1 漏 init_flag → 第一拍死锁

- **现象**：kernel 跑出去没反应，`torch.npu.synchronize()` 卡住，msprof 看到 stall。
- **根因**：循环第一拍 `wait_flag("mte1","mte2",0)`，MTE1 还没跑过没 set，wait 永远阻塞。
- **解法**：kernel 入口 `init_flag()` 预借信用。见 [flag-mechanism.md §信用借贷](flag-mechanism.md)。

### 1.2 漏 clear_flag → 下个 kernel 死锁

- **现象**：当前 kernel 跑完没问题，接着跑下一个 kernel 时卡住。
- **根因**：init 借的 flag 没 wait 回来，硬件队列没清空，影响下一个 kernel 的 flag 状态。
- **解法**：kernel 退出前 `clear_flag()` 归还，借几个还几个、id 对齐。

### 1.3 init/clear 不对称

- **现象**：死锁。
- **根因**：init 借了 5 个 flag，clear 只还了 4 个（漏了一个 id）。
- **解法**：严格对照 init 的 set 列表写 clear 的 wait 列表，id 一一对应。

### 1.4 set/wait 不配对（跨循环）

- **现象**：死锁或数据错乱。
- **根因**：某 `(src,dst,eid)` 循环里 set N 次 wait M 次（M≠N），且不属于 [flag-mechanism.md §三类合法模式](flag-mechanism.md) 任何一种。
- **解法**：对照三类合法模式验证（A 严格配对 / B 一次性通知 / C 混合节流）。

### 1.5 Vector 用 "FIX" 做 cross_flag

- **现象**：`Illegal instruction (unaligned UUB addresses)`。
- **根因**：`set_cross_flag("FIX", ...)` 在 Vector scope 内。Vector 核没有 FIX 管线。
- **解法**：Vector 侧 `set_cross_flag` 的 pipe 用 `"MTE3"`（UB→GM）。见 [hardware.md §pipe 参数](hardware.md)。

### 1.6 cross_interval > workspace num_stages

- **现象**：数据错乱（前一批还没消费就被覆盖）或死锁。
- **根因**：cross_interval=N 要求 workspace 容纳 N 个版本，`workspace.shape[1] < N` 时第 i 次写的数据被第 i+N 次覆盖。
- **解法**：`workspace: T.Tensor([NUM_CORES, num_stages, ...])`，`num_stages ≥ cross_interval`。

### 1.7 同核 C↔V 用核内 flag

- **现象**：死锁，C 和 V 互相等。
- **根因**：即使 C0 和 V0 在同一物理核，C↔V 同步也走 FFTS 跨核总线，不能用 `set_flag/wait_flag`。
- **解法**：CV 协作一律用 `set_cross_flag/wait_cross_flag`。见 [hardware.md §FFTS](hardware.md)。

### 1.8 auto_sync 没关

- **现象**：手写 flag 的 kernel 死锁或性能回退。
- **根因**：`auto_sync=True` 时 pass 重复插入核内 `set/wait_flag`，和手写 flag 冲突。
- **解法**：手写核内 flag 时关 `auto_sync`；手写 `cross_flag` 时关 `auto_cv_sync`/`auto_cv_combine`。完整 pass_configs 配置见 `tilelang-mode-guide` skill。

### 1.9 @T.macro 内 flag 位置错误

- **现象**：死锁，init/clear 看似写了但没生效。
- **根因**：`@T.macro` 是 inline 展开，宏体内的 `set_flag/wait_flag` 位置必须和调用点一致。如果宏定义在循环外但调用在循环内，flag 的 iter_ctx 属于循环内；反之亦然。
- **解法**：宏只是代码替换，**不要以为宏定义在哪就在哪执行**。将宏体 inline 展开到调用点后验证 flag 顺序。init_flag 宏必须在 scope 入口、循环之前调用；clear_flag 宏必须在循环结束后、scope 退出前调用。

### 1.10 cross_flag 末次漏 set

- **现象**：最后几轮数据没被消费，或 C 等 V 的信号死锁。
- **根因**：节流 set 条件只写了 `if (i+1) % cross_interval == 0`，漏了 `or i == N-1`，最后一个版本没发信号。
- **解法**：节流 set 条件必须写 `if (i+1) % cross_interval == 0 or i == N-1`。见 [cv-cross.md §2](templates/cv-cross.md)。

## 2. 容量约束类

### 2.1 L0C 溢出

- **现象**：segfault。
- **根因**：`block_M × block_N × sizeof(accum) > 128KB`。
- **解法**：减小 block_M 或 block_N，或拆分。满足 `block_M × block_N ≤ 32768`（float32）/ `≤ 65536`（float16）。

### 2.2 L0A/L0B 溢出（S2 过大）

- **现象**：编译错误或运行错误。
- **根因**：`S2 × block_M × block_K × sizeof(dtype) > 64KB`。
- **解法**：S2 几乎总是 2。block 太大时 S2=1 也能跑（单缓冲，流水退化）。

### 2.3 L1 溢出（S1 过大）

- **现象**：编译错误。
- **根因**：A_L1 和 B_L1 的合计占用超过 L1。常见误判是只检查 `S1 × block_M × K_L1` 或 `S1 × K_L1 × block_N` 的单项占用。
- **解法**：按 `S1 × (block_M + block_N) × K_L1 × sizeof(dtype) + extra_L1_buffers ≤ 512KB` 检查，减小 S1、K_L1 或 block_N。

### 2.4 UB 溢出（Vector 侧缓冲过多）

- **现象**：编译错误或运行错误。
- **根因**：`Σ buffers × stages × rows × cols × sizeof(dtype) > 192KB`。
- **解法**：减少 stages（从 2 降到 1），或减小 rows_per_vec，或合并临时 buffer。

## 3. 性能回退类

### 3.1 barrier_all 没换成 flag

- **现象**：能跑，但性能没提升（和单缓冲串行差不多）。
- **根因**：循环里还有 `T.barrier_all()`，所有管线全局屏障，退化成串行。
- **解法**：把 `T.barrier_all()` 换成细粒度 `set_flag/wait_flag`。

### 3.2 单缓冲没开双缓冲

- **现象**：性能只有四级的 1/4。
- **根因**：S1=1 或 S2=1，流水深度不够，MTE2 写时 MTE1 不能读同一片。
- **解法**：S1=2-4，S2=2。见 [hardware.md §缓冲容量](hardware.md)。

### 3.3 cross_interval=1 没节流

- **现象**：CV 融合性能不达标，msprof 看 Cube 经常 stall 等 Vector。
- **根因**：每次迭代都 cross_flag，核间同步开销大。
- **解法**：cross_interval=2（默认），workspace 开 num_stages=2 容纳 2 个版本。

### 3.4 pipe 参数选错导致等错管线

- **现象**：不死锁但数据错乱或性能差。
- **根因**：`set_flag(src,dst,eid)` 的 src 不匹配实际生产数据的管线。比如 Cube 算完 L0C 应该用 `"m"` 发信号，但写成 `"mte1"`，wait 方在等 MTE1 的信号但 MTE1 早就完成了。
- **解法**：set 的 src 必须是**写完数据的那个管线**。Cube 算完 → src="m"；MTE2 搬完 → src="mte2"；MTE1 搬完 → src="mte1"；FIX 写完 GM → src="fix"。

## 4. 速查：常见错误对照表

| 错误 | 现象 | 根因 | 解法 |
|---|---|---|---|
| 漏 init_flag | 第一拍死锁 | wait 时没 set | init 借信用 |
| 漏 clear_flag | 下个 kernel 死锁 | 借的没还 | clear 还信用 |
| init/clear 不对称 | 死锁 | 借还数量/id 不对 | 严格对照 |
| set/wait 不配对 | 死锁或错乱 | 不属三类合法模式 | 对照三类模式验证 |
| Vector 用 "FIX" | Illegal instruction | pipe 选错 | Vector 用 "MTE3" |
| cross_interval > num_stages | 数据错乱 | workspace 不够 | num_stages ≥ cross_interval |
| 同核 C↔V 用核内 flag | 死锁 | 应走 FFTS | 用 cross_flag |
| auto_sync 没关 | 死锁/回退 | pass 重复插核内 flag | 关 auto_sync |
| 宏内 flag 位置错 | 死锁 | 展开后顺序不对 | inline 展开后验证 |
| cross_flag 末次漏 set | 末尾死锁 | 节流条件缺 or i==N-1 | 补末次条件 |
| L0C 溢出 | segfault | block 太大 | 减小 block |
| barrier_all 没换 | 性能没提升 | 全局屏障 | 换成 flag |
| pipe 参数选错 | 不死锁但错乱/慢 | src 不匹配生产管线 | set src = 写完数据的管线 |
