---
name: flag-pipeline
description: 昇腾 NPU (910B) 上 TileLang-Ascend 算子的手写 flag 流水线编排。覆盖核内四级流水 (MTE2/MTE1/M/FIX)、Vector 三级流水 (MTE2/V/MTE3)、Cube↔Vector 跨核流水 (cross_flag)。对性能有要求的场景使用 expert 模式手写 flag，让多管线独立 overlap；也用于手写 flag 出现死锁时的根因定位。触发：手写 flag 流水线、expert 模式、cross_flag 编排、flag 死锁。
---

# Flag Pipeline Skill

昇腾 910B 上 TileLang-Ascend 算子的**手写 flag 流水线**编排。目标是让 MTE2(GM→L1) / MTE1(L1→L0A/L0B) / M(Cube) / FIX(GM 写回) 或 MTE2/V/MTE3 独立流水并行，最大化计算吞吐。

## 适用场景

- **性能要求高**：自动模式 (T.Pipelined / Developer) 性能不达标，需要 expert 模式手写 flag 让多管线 overlap
- **复杂流水模式**：如 Cube+Vector 跨核协作 (FA)、非标准流水深度
- **调试死锁**：手写 flag 出现挂起/死锁/数据错乱，需要定位根因
- **性能回退诊断**：kernel 能跑但没达到预期吞吐

## 不适用场景

- 性能要求不高 → 用 `T.Pipelined` 自动流水即可
- 简单逐元素算子 → 不需要写 flag
- 算子正确性调试（和 flag 无关的 shape/dtype 问题）

## 工作流

### Step 1: 判断流水类型

根据算子类型选模板：

| 算子类型 | 流水级别 | 模板 | 管线 |
|---------|---------|------|------|
| GEMDAT / Matmul 类（Cube 主导） | 核内四级 | [cube-4stage.md](references/templates/cube-4stage.md) | MTE2 → MTE1 → M → FIX |
| Element-wise 类（Vector 主导） | 核内三级 | [vector-3stage.md](references/templates/vector-3stage.md) | MTE2 → V → MTE3 |
| FA / CV 融合类（Cube+Vector 协作） | 核内+跨核 | [cv-cross.md](references/templates/cv-cross.md) | 核内四级 + cross_flag |

先读 [hardware.md](references/hardware.md) 理解 MTE2/MTE1/M/FIX/MTE3 对应的数据搬运方向和 buffer，再读对应模板。

### Step 2: 设计 buffer 轮转

关键约束（详见 [hardware.md §缓冲容量](references/hardware.md)）：
- L0A/L0B: S2=2 (双缓冲)，`S2 × block_M × block_K × sizeof(dtype) ≤ 64KB`
- L1: S1=2~4，`S1 × block_M × K_L1 × sizeof(dtype) ≤ 512KB`
- L0C: 不轮转但大小受限，`block_M × block_N × sizeof(accum) ≤ 128KB`
- UB: S_vec=2，`S_vec × rows × cols × sizeof(dtype) ≤ 192KB`

buffer shape 第一维永远是 stage 轮转维，`T.alloc_buf` + `% stages` 实现 ping-pong。

### Step 3: 按模板写 flag

严格按模板填，**不要自创 flag 模式**。三种合法 flag 模式见 [flag-mechanism.md §三类合法模式](references/flag-mechanism.md)：

- **模式 A（严格配对）**：循环内 set N 次 wait N 次，单 buffer 轮转
- **模式 B（纯广播）**：init set 1 次，循环内 wait N 次（少用）
- **模式 C（混合节流）**：init 借信用 + 循环内节流 set，消费者逐轮 wait（FA 模式）

核心原则：**init 借信用，clear 归还；谁等待谁 wait，谁完成谁 set**。

### Step 4: pass_configs 四件套全关

```python
pass_configs = {
    "tl.ascend.auto_sync": False,
    "tl.ascend.auto_cv_sync": False,
    "tl.ascend.auto_cv_combine": False,
    "tl.ascend.memory_planning": False,
}
```

详见 [pitfalls.md §4](references/pitfalls.md)。不关的话 pass 会自动插 flag，和手写的冲突。

### Step 5: 静态自检 + 上板验证

- **静态自检**：对照 [flag-mechanism.md §三类合法模式](references/flag-mechanism.md) 检查 set/wait 配对，确认 init/clear 对称。
- **上板**：先跑小 shape 验证不死锁，再跑 msprof 看 overlap。
- **死锁调试**：见 [debugging.md](references/debugging.md) 的删减定位法。
- **性能验证**：msprof 看各管线 stall ratio，确认 overlap 充分。

## 关键规则（不可违反）

1. **init 必借、clear 必还**：kernel 入口 set 所有需要的初始信用，退出前 wait 归还。漏 init → 第一拍死锁；漏 clear → 下个 kernel 死锁。
2. **pipe 参数必须匹配**：Cube set_cross_flag 用 `"FIX"`（写完 GM 再通知），Vector set_cross_flag 用 `"MTE3"`（UB→GM 完成后通知）。
3. **cross_interval ≥ num_stages**：跨核节流时 workspace 缓冲数 ≥ 节流间隔，否则数据覆盖。
4. **C↔V 一律 cross_flag**：即使同物理核也走 FFTS 跨核总线，不能用核内 set_flag/wait_flag。
5. **同 id 同方向**：一个 `(src,dst,eid)` 只能一个方向 set→wait，不能双向混用。
6. **不要混 barrier_all**：barrier_all 全局屏障会破坏流水 overlap，换成细粒度 set/wait_flag。

## 参考文件索引

| 文件 | 内容 |
|------|------|
| [hardware.md](references/hardware.md) | 管线定义 / 缓冲容量 / FFTS 跨核 / pipe 参数 / scope 分配 |
| [flag-mechanism.md](references/flag-mechanism.md) | set/wait 语义 / 三类合法模式 / 信用借贷 / 核间 cross_flag |
| [pitfalls.md](references/pitfalls.md) | 死锁根因、容量约束、pass_configs 语义 |
| [debugging.md](references/debugging.md) | 静态自检 → 删减定位 → msprof |
| [templates/cube-4stage.md](references/templates/cube-4stage.md) | Cube 四级流水模板（MTE2/MTE1/M/FIX） |
| [templates/vector-3stage.md](references/templates/vector-3stage.md) | Vector 三级流水模板（MTE2/V/MTE3） |
| [templates/cv-cross.md](references/templates/cv-cross.md) | Cube↔Vector 跨核流水模板（cross_flag） |
