# 硬件基础：管线、内存层级、物理拓扑

本章只讲写算子需要知道的硬件事实，不涉及 pass 实现细节。

## 1. 内存层级与数据通路

昇腾 910B 一个 AI Core 内部的内存层级：

```
GM (全局内存, HBM)
  ↓ MTE2 (Memory Transfer Engine 2)
L1 (L1 Buffer, 512KB, Cube 缓存)
  ↓ MTE1 (Memory Transfer Engine 1)
L0A / L0B (矩阵乘输入寄存器, 64KB 各)
  ↓ M (Matrix Multiply Unit)
L0C (矩阵乘输出寄存器, 128KB)
  ↓ FIX (Fixed Function Unit)
GM
```

数据必须逐级搬运，不能跨级。GM → L0 直接搬运非法，Cube 侧必须走 GM→L1→L0→L0C→GM 四级。

Vector 核（AIV）通路更简单，只有三级：

```
GM ──MTE2──→ UB (Unified Buffer, 192KB, Vector 缓冲)
                ↑↓ V (Vector Unit, 向量计算)
GM ←─MTE3─── UB
```

## 2. Cube 四条独立硬件管线

每条管线是独立的硬件执行单元，可以并行工作。这是 flag 存在的根本原因——四条管线各自异步，需要信号量协调。

| 管线 | 全名 | 职责 | 典型 API |
|---|---|---|---|
| **MTE2** | Memory Transfer Engine 2 | GM → L1 搬运 | `T.copy(GM_tensor, L1_buf)` |
| **MTE1** | Memory Transfer Engine 1 | L1 → L0A/L0B 搬运（含 transpose） | `T.copy(L1_buf, L0_buf, transpose=True)` |
| **M** | Matrix Multiply Unit | L0A · L0B → L0C 矩阵乘累加 | `T.mma(L0A, L0B, L0C)` |
| **FIX** | Fixed Function Unit | L0C → GM 写回 | `T.copy(L0C_buf, GM_tensor)` |

## 3. Vector 核的管线

| 管线 | 职责 | 典型 API |
|---|---|---|
| **MTE2** | GM → UB | `T.copy(GM, UB)` |
| **V** | UB 内向量计算 | `T.tile.add/mul/exp/...` |
| **MTE3** | UB → GM | `T.copy(UB, GM)` |

Cube 的 MTE2 和 Vector 的 MTE2 是不同物理单元，只是同名（都负责"从 GM 读"）。flag 的 `src/dst` 用管线名，靠 `T.Scope("C")` / `T.Scope("V")` 区分作用域。

## 4. 管线异步性

`T.copy` 和 `T.mma` 都是异步发射的指令，发射后立刻返回，不等执行完：

```python
T.copy(A, A_L1)                  # 发射 MTE2 指令，立刻返回
T.copy(B, B_L1)                  # 发射 MTE2 指令，立刻返回
T.gemm_v0(A_L1, B_L1, C_L0)      # 发射 M 指令，立刻返回
# 此时三条指令可能都还没执行完
```

下游管线会在上游还没写完时去读，读到脏数据。flag 用于协调：生产者完工后 set 信号，消费者 wait 到信号后才读。

## 5. 物理拓扑：C0 / V0 / V1 与 FFTS

手写 CV 融合算子的必读内容。

### 5.1 AI Core 组成

昇腾 910B 一个 AI Core（物理核）内部包含：

- 1 个 Cube 单元（记作 C0）
- 2 个 Vector 单元（记作 V0、V1）

C0 和 V0 在同一个物理核内；V1 在另一个核里。但"同核"并不意味着 C↔V 通信走核内通道。

### 5.2 FFTS 跨核信号总线

即使 C0 和 V0 在同一个物理核内，C↔V 之间的同步也走 FFTS（Fast Fabric for Task Synchronization）跨核信号总线，而不是核内 flag。

- **核内 flag**（`set_flag/wait_flag`）：用于同一条管线之间、或 Cube 内部四条管线之间、或 Vector 内部三条管线之间的同步。走核内信号。
- **跨核 flag**（`set_cross_flag/wait_cross_flag`）：用于 Cube 和 Vector 之间的同步，无论是否在同一物理核，都走 FFTS。

CV 协作时即使 Cube 和 Vector "在同一个核上"，也必须用 `set_cross_flag`。

### 5.3 set_cross_flag 的 mode 参数

`set_cross_flag(pipe, flag, mode=2)` 第三个参数控制信号广播范围：

| mode | 范围 | 用途 |
|---|---|---|
| 0 | 所有 AIC ↔ 所有 AIV | 全核广播，极少用 |
| 1 | 同组 AIV 之间 | Vector 内部同步 |
| 2（默认） | 同组 AIC 和 AIV 之间 | CV 协作的标准模式 |

手写 CV 流水几乎总用默认 mode=2。

### 5.4 pipe 参数：发信号的管线

`set_cross_flag(pipe, flag)` 的 `pipe` 参数是"由哪条管线发这个信号"，不是"数据走哪条通路"。二者有巧合：

| 核类型 | 写 GM 的实际通路 | set_cross_flag 的 pipe 参数 |
|---|---|---|
| Vector (V) | UB → GM | `"MTE3"` |
| Cube (C) | L0C → GM | `"FIX"` |

巧合原因：写完 GM 才能通知对方读，发信号的时机正好是数据写回 GM 完成时，所以 pipe 就是负责写 GM 的那条管线。

Vector 核用 `set_cross_flag("FIX", ...)` 会报 `Illegal instruction (unaligned UUB addresses)`，因为 Vector 核没有 FIX 管线。

## 6. 缓冲容量与双缓冲深度上限

S1（L1 深度）和 S2（L0 深度）的上限由硬件容量和分块大小共同决定。

### 6.1 硬件容量

| 缓冲 | 容量 | 对应参数 |
|---|---|---|
| L0A / L0B | 64KB 各 | S2 |
| L0C | 128KB | （若 L0C 也 ping-pong） |
| L1 | 512KB | S1 |
| UB | 192KB | Vector 侧缓冲深度 |

### 6.2 约束公式

```
S2 × block_M × block_K × sizeof(dtype) ≤ 64KB   (L0A)
S2 × block_K × block_N × sizeof(dtype) ≤ 64KB   (L0B)
S1 × block_M × K_L1   × sizeof(dtype) ≤ 512KB  (L1, A 分量)
S1 × K_L1   × block_N × sizeof(dtype) ≤ 512KB  (L1, B 分量)
block_M × block_N × sizeof(accum_dtype) ≤ 128KB (L0C, 单缓冲)
```

### 6.3 工程经验

| 参数 | 推荐值 | 原因 |
|---|---|---|
| **S1** | 2-4 | L1 容量大（512KB），可开深一点隐藏 MTE2 延迟 |
| **S2** | 2（几乎固定） | L0A/L0B 只有 64KB，开 2 已接近上限 |

举例 `block_M=128, block_K=64, fp16`（2 字节）：

```
单片 L0A = 128 × 64 × 2B = 16KB
S2=2 → 32KB ✅
S2=3 → 48KB（勉强，但 L0B 也要占空间）
S2=4 → 64KB ❌（L0A 满了，L0B 没空间）
```

S2 几乎总是 2，S1 在 2-4 之间调。S 越大流水越深、隐藏延迟越好，但受容量限制。

### 6.4 L0C 容量陷阱

A2/A3 设备 L0C = 128KB。`block_M × block_N × sizeof(accum) > 128KB` 会 segfault。

```
block_M=128, block_N=256, accum=float32 (4B)
128 × 256 × 4 = 128KB ← 正好顶满，OK
block_M=256, block_N=256, accum=float32
256 × 256 × 4 = 256KB ❌ 超 L0C
```

设计 block 时满足 `block_M × block_N × sizeof(accum) ≤ 128KB`。

## 7. 管线名速查

| 管线名 | Cube/Vector | 职责 |
|---|---|---|
| `mte2` | 都有（不同物理单元） | GM → L1 / GM → UB |
| `mte1` | Cube | L1 → L0A/L0B |
| `m` | Cube | L0A · L0B → L0C |
| `fix` | Cube | L0C → GM |
| `v` | Vector | UB 内向量计算 |
| `mte3` | Vector | UB → GM |

flag 的 `src/dst` 用这些名字，大小写不敏感（源码里都转小写）。
