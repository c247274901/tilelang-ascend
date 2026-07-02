# Vector 三级流水模板（MTE2 / V / MTE3）

纯 Vector 算子（elementwise、reduce、softmax 前后处理）的标准模板，对齐 `.agents/skills/tilelang-perf-optimization/references/best-practices/vector_add_pipeline.md` 的三阶段写法，补充"非显然"点解释。

> Vector 算子优先用 `T.Pipelined` 自动模式。自动模式不达标、或要和 Cube 跨核协作时，才手写 flag。

## 1. 三阶段结构（Prefetch / Main Body / Epilogue）

Vector 三管线（MTE2/V/MTE3）的手写 flag 循环，直接写成 `for tile: wait→copy→compute→copy→set` 会让首尾边界混乱：

- 第一拍 MTE2 没有"上一拍 MTE3 释放 UB"的 flag → 死锁（靠 init 借信用解）
- 最后一拍预取了下一块但没有"下一块" → 要 if 判断

三阶段写法把首尾特殊处理拆出来：

- **Prefetch**：循环前搬第一块，给主循环提供第一个可消费 tile
- **Main Body**：循环 N-1 次，每次预取下一块 + 消费当前块 + 写回当前块
- **Epilogue**：循环后消费最后一块，避免尾块遗漏

主循环里没有 `if tile < N-1` 的判断，边界清晰。

## 2. 参数与缓冲

```python
VEC_NUM = 2                       # Vector sub-core 数
stages = 2                        # UB 双缓冲
rows_per_vec = sub_M // VEC_NUM   # 每个 Vector sub-core 处理的行数
tiles_per_vec = block_M // sub_M  # 主循环次数（至少 2，否则流水起停开销 > 收益）

a_ub = T.alloc_ub((stages, rows_per_vec, block_N), dtype)
b_ub = T.alloc_ub((stages, rows_per_vec, block_N), dtype)
c_ub = T.alloc_ub((stages, rows_per_vec, block_N), dtype)
```

UB 占用估算：`3 buffers × stages × rows_per_vec × block_N × sizeof(dtype)`，要给临时 buffer 留余量，总不超 192KB。

## 3. 信号 ID 表

| ID | 管线对 | 含义 |
|---|---|---|
| 0, 1 | MTE3 → MTE2 | UB slot 可被 MTE2 重新搬入（写回完成后） |
| 0, 1 | MTE2 → V | UB slot 输入已搬入，可被 V 消费 |
| 0, 1 | V → MTE3 | UB slot 计算结果已产生，可写回 GM |

三组都用 id 0,1（stages=2 双缓冲），但 `(src,dst)` 不同，是独立信箱。

## 4. init_flags / drain_flags（信用借贷）

```python
@T.macro
def init_flags():
    # 借：让 Prefetch 的 wait MTE3→MTE2 通过（UB slot 0,1 初始"可写"）
    T.set_flag("mte3", "mte2", 0)
    T.set_flag("mte3", "mte2", 1)

@T.macro
def drain_flags():
    # 还：借几个还几个
    T.wait_flag("mte3", "mte2", 0)
    T.wait_flag("mte3", "mte2", 1)
```

只借 MTE3→MTE2 两个（Prefetch 只 wait 这一组）。MTE2→V 和 V→MTE3 不用借——它们的 wait 都在主循环里，前面有对应的 set（Prefetch 的 set 或上一拍的 set）。

## 5. 三阶段主体

```python
with T.Scope("V"):
    init_flags()

    # === Prefetch: 搬第一块 ===
    first_row = bx * block_M + vid * rows_per_vec
    T.wait_flag("mte3", "mte2", 0)
    T.copy(A[first_row, by * block_N], a_ub[0, :, :])
    T.copy(B[first_row, by * block_N], b_ub[0, :, :])
    T.set_flag("mte2", "v", 0)

    # === Main Body: N-1 轮，预取下一块 + 消费当前块 ===
    for tile in T.serial(0, tiles_per_vec - 1):
        cur = tile % stages
        nxt = (tile + 1) % stages
        cur_row = bx * block_M + vid * rows_per_vec + tile * sub_M
        next_row = bx * block_M + vid * rows_per_vec + (tile + 1) * sub_M

        # 预取下一块到 nxt
        T.wait_flag("mte3", "mte2", nxt)
        T.copy(A[next_row, by * block_N], a_ub[nxt, :, :])
        T.copy(B[next_row, by * block_N], b_ub[nxt, :, :])
        T.set_flag("mte2", "v", nxt)

        # 消费当前块 cur
        T.wait_flag("mte2", "v", cur)
        T.tile.add(c_ub[cur, :, :], a_ub[cur, :, :], b_ub[cur, :, :])
        T.set_flag("v", "mte3", cur)

        # 写回当前块
        T.wait_flag("v", "mte3", cur)
        T.copy(c_ub[cur, :, :], C[cur_row, by * block_N])
        T.set_flag("mte3", "mte2", cur)

    # === Epilogue: 消费最后一块 ===
    last_tile = tiles_per_vec - 1
    last_stage = last_tile % stages
    last_row = bx * block_M + vid * rows_per_vec + last_tile * sub_M

    T.wait_flag("mte2", "v", last_stage)
    T.tile.add(c_ub[last_stage, :, :], a_ub[last_stage, :, :], b_ub[last_stage, :, :])
    T.set_flag("v", "mte3", last_stage)

    T.wait_flag("v", "mte3", last_stage)
    T.copy(c_ub[last_stage, :, :], C[last_row, by * block_N])
    T.set_flag("mte3", "mte2", last_stage)

    drain_flags()
```

## 6. 非显然点

### 6.1 init 只借 MTE3→MTE2

Prefetch 第一行 `wait_flag("mte3","mte2",0)`——MTE3 还没跑过，没 set，所以要借。主循环里 `wait_flag("mte2","v",cur)` 的第一拍，前面有 Prefetch 的 `set_flag("mte2","v",0)` 兜底；`wait_flag("v","mte3",cur)` 的第一拍，前面有主循环里 `set_flag("v","mte3",cur)`——这些 wait 都有前置 set，不用借。

### 6.2 drain 也只 wait MTE3→MTE2

借了几个就还几个。init 只借了 MTE3→MTE2 两个，drain 就只还这两个。其他信箱在循环里自然配对清零了。

### 6.3 Epilogue 的最后 `set_flag("mte3","mte2",last_stage)`

把 last_stage 的 UB slot 释放回"可写"状态，让 drain 的 `wait_flag("mte3","mte2",last_stage)` 能取到令牌。漏了这个 set，drain 会死锁。

### 6.4 不能用 `T.barrier_all()`

`barrier_all` 是所有管线全局屏障，会让 MTE2、V、MTE3 三条管线相互等，退化成串行。手写 flag 的目的就是用细粒度 `set/wait` 替代 `barrier_all`，让三管线 overlap。

## 7. 三级流水时序图

```
时间 →   t0      t1      t2      t3      t4      t5
MTE2:   pref0   pref1   pref2   pref3   ...
V:              add0    add1    add2    add3    ...
MTE3:                   store0  store1  store2  store3
```

对比 barrier_all 串行：

```
MTE2:   [pref0]              [pref1]              [pref2]
V:             [add0]               [add1]               [add2]
MTE3:                  [store0]              [store1]
```

吞吐量约 3 倍。

## 8. 适配 Checklist

- [ ] `block_M % sub_M == 0`
- [ ] `sub_M % VEC_NUM == 0`
- [ ] `tiles_per_vec = block_M // sub_M ≥ 2`（否则流水起停开销 > 收益）
- [ ] UB 占用 `3 × stages × rows_per_vec × block_N × sizeof(dtype) < 192KB`
- [ ] init_flags 借的 id = drain_flags 还的 id
- [ ] 把 `T.tile.add` 换成目标算子的 Vector 计算（mul/exp/reduce/...）
- [ ] 生成的 Ascend C 里能看到 `MTE3_MTE2`、`MTE2_V`、`V_MTE3` 三类事件

## 9. 参考

- 完整版来源：`.agents/skills/tilelang-perf-optimization/references/best-practices/vector_add_pipeline.md`
- 基线：`examples/elementwise/elementwise_add.py`
- 机制原理：[flag-mechanism.md](../flag-mechanism.md)
- 硬件管线：[hardware.md](../hardware.md)
