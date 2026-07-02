# Flag 机制详解：set/wait 语义 / 三类合法模式 / 信用借贷 / 核间 cross_flag

理解 flag 的底层语义是正确写流水的前提。不要凭直觉写 set/wait，要理解"为什么"。

## 1. 基础概念

### 1.1 flag 是计数信号量

每个 flag 是一个硬件信箱，底层是**有向计数信号量**：
- `set_flag(src, dst, eid)`：在 dst 的信箱里投一个令牌（credit +1），非阻塞
- `wait_flag(src, dst, eid)`：等 dst 信箱里有令牌，取走一个（credit -1），阻塞等待
- 一个 flag 三元组 `(src, dst, eid)` 代表一条**有向边**：src → dst

```
set_flag("mte2","mte1",0)
  └─ src=mte2（谁发信号）
  └─ dst=mte1（发给谁）
  └─ eid=0（信箱编号，用于多条边复用同对 src→dst）
```

### 1.2 set 和 wait 不对称

- **set 是非阻塞的**：发信号方发完就走，不等消费
- **wait 是阻塞的**：消费方等信号到了才继续
- 因此生产者（写数据的一方）set，消费者（读数据的一方）wait
- **方向不可反**：不能让消费者 set 然后等生产者，那就变成"先通知再生产"了

### 1.3 pipe 含义

第一个参数 `src` 指定发信号的**生产管线**：

| src 字符串 | 对应管线 | 语义 |
|-----------|---------|------|
| `"mte2"` | MTE2 (DMA) | GM → L1/UB 搬运完成 |
| `"mte1"` | MTE1 (DMA) | L1 → L0A/L0B 搬运完成 |
| `"m"` | M (Cube/Matrix) | Cube 计算完成 |
| `"fix"` | FIX (DMA) | L0C → GM 写回完成 |
| `"v"` | V (Vector) | Vector 计算完成 |
| `"mte3"` | MTE3 (DMA) | UB → GM 写回完成 |

set_flag 的 src 必须是**写完数据的那个管线**。等错管线 = 数据还没写完就开始读 → 数据错乱。

### 1.4 核内 flag vs 核间 cross_flag

| | set_flag/wait_flag | set_cross_flag/wait_cross_flag |
|---|---|---|
| 作用域 | 单核内（C核内/V核内） | 跨核（C核↔V核） |
| 参数 | (src, dst, eid) | (pipe, flag_id) |
| 总线 | 核内寄存器 | FFTS 跨核总线 |
| 信箱 | 每核私有 | C0↔V0 共享 |
| 适用 | 核内流水（MTE2↔MTE1↔M↔FIX） | CV协作（C写完workspace通知V） |

**注意**：即使 C0 和 V0 在同一个 AI Core 上，C↔V 通信也**必须走 cross_flag**，不能用核内 flag。

## 2. 信用借贷模型

flag 本质是令牌（credit）流转。写流水的核心是理解"信用"。

### 2.1 初始状态

- 所有 flag 信箱初始信用为 0
- 如果循环第一拍就是 wait_flag，信用为 0 → wait 永远阻塞 → 第一拍死锁
- **解法：init_flag 预借信用**——kernel 入口先 set 一次，给第一个 wait 一个令牌

### 2.2 init 借、clear 还

```python
@T.macro
def init_flag():
    T.set_flag("mte2", "mte1", 0)  # 借1个给MTE1→M方向
    T.set_flag("mte2", "mte1", 1)  # S1=2双缓冲需要借2个
    # ... 所有第一拍wait需要的信用

@T.macro
def clear_flag():
    T.wait_flag("fix", "m", 0)  # 还回去
    T.wait_flag("fix", "m", 1)
    # ... 和init一一对应，等回来
```

- init 借几个，clear 还几个，id 严格对齐
- 漏 init → 第一拍死锁
- 漏 clear → 下个 kernel 死锁（硬件 flag 状态没清空）

### 2.3 双缓冲的信用流转

以 MTE2→MTE1 双缓冲（S1=2）为例：

```
时间轴:
  MTE2: set(buf0) ─ copy(buf1) ─ set(buf1) ─ copy(buf0) ─ set(buf0) ─ ...
  MTE1: wait(buf0)─copy(buf0)─wait(buf1)─copy(buf1)─wait(buf0)─...
         ↑ init借的buf0信用          ↑ MTE2 set(buf1)的信用
```

- init 借了 buf0（eid=0）的信用，MTE1 第一轮 wait(buf0) 有信用不阻塞
- MTE1 消费 buf0 时，MTE2 在 copy buf1，copy 完 set(buf1) → MTE1 下一轮 wait(buf1) 有信用
- 如此循环，buf0 和 buf1 交替轮转

## 3. 三类合法 set/wait 模式

一个 `(src, dst, eid)` 或 `(flag,)` 在整个 kernel 生命周期内，set 和 wait 的分布只有三种合法模式。

### 模式 A：严格配对（最常见）

循环内 set N 次，wait N 次，一一对应。

```
set:   S  S  S  S  ... S    (N次, 循环内)
wait:  W  W  W  W  ... W    (N次, 循环内)
```

典型场景：双缓冲 ping-pong，MTE2 copy 完 set，MTE1 wait 后 copy，循环内每次迭代 set+wait 各一次。

```python
# Cube 四级 MTE1→M 方向就是模式A
for kk in T.serial(NUM_K):
    T.wait_flag("mte1", "m", s2)       # wait L0A/L0B就绪
    T.gemm(...)
    T.set_flag("m", "fix", s2c)        # set L0C就绪
```

### 模式 B：纯广播

init set 1 次，循环内 wait N 次。

```
set:   S                        (1次, init/循环外)
wait:     W  W  W  W  ... W    (N次, 循环内)
```

典型场景：单向通知一次，消费者循环等待同一信号。纯形式下第一个 wait 消费 init 令牌后后续 wait 会阻塞，所以模式B 很少单独使用，通常演化为模式 C。

### 模式 C：混合节流（FA 模式，最复杂）

init 借信用（1个set）+ 循环内节流 set（每 K 次 set 一次）+ 循环内逐轮 wait（每次都wait）。

```
set:   S           S        S        ...  S     (1 init + ⌈N/K⌉ 循环内)
wait:     W  W  W  W  W  W  W  W  W  ... W     (N次, 每次循环wait)
```

K = cross_interval（节流间隔）。V 处理完 K 个 batch 才 set 一次通知 C，而 C 每轮都 wait（或者反过来）。

典型场景：**Cube↔Vector cross_flag**。Cube 算完 K 轮（K=cross_interval）写一批到 workspace，set_cross_flag 一次通知 V；V 每轮处理都 wait（或者 V 处理完 K 轮 set 一次）。

```python
# CV cross_flag C→V 方向 (FA模式)
# C 侧: 每cross_interval轮set一次
for i in T.serial(batch_iters):
    T.wait_cross_flag(SEM_WS1_V2C)
    # ... C 计算 ...
    if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
        T.set_cross_flag("FIX", SEM_WS1_C2V)  # 节流set

# V 侧: 每轮都wait，处理完cross_interval轮set一次
for i in T.serial(batch_iters):
    if i % cross_interval == 0:
        T.wait_cross_flag(SEM_WS1_C2V)         # 每K轮wait一次
    # ... V 处理 ...
    if (i + 1) % cross_interval == 0 or i == batch_iters - 1:
        T.set_cross_flag("MTE3", SEM_WS1_V2C)  # 节流set
```

**模式 C 的 count 特征**：set 数 < wait 数（或反过来取决于方向），这是合法的，因为一个 set 通知对应 K 个 wait 消费（K 轮数据就绪）。

关键注意点：
1. **末次必须 set**：条件 `if (i+1)%K==0 or i==N-1`，最后一轮即使不整除 K 也要 set，否则最后几个版本没发信号 → 死锁
2. **init 借信用**：循环前 set 一个，让消费者第一轮 wait 不阻塞
3. **workspace num_stages ≥ cross_interval**：要容纳 K 个版本的数据不被覆盖

## 4. 核间 cross_flag 详解

### 4.1 cross_flag 参数

```python
T.set_cross_flag(pipe, flag_id)
T.wait_cross_flag(flag_id)
```

- `pipe`：**生产数据的管线**，写完 GM 后发信号
  - C scope 内：C 写完 GM 走 FIX 管线 → `pipe="FIX"`
  - V scope 内：V 写完 GM 走 MTE3 管线 → `pipe="MTE3"`
  - V scope 内只读不写（init借信用）：也要写 pipe，V 没有 FIX → `pipe="MTE2"` 或 `"MTE3"`
- `flag_id`：flag 编号（整数），C→V 和 V→C 用不同 id

### 4.2 cross_flag 方向

```
C scope                           V scope
─────────                         ─────────
set_cross_flag("FIX", C2V_id)  → wait_cross_flag(C2V_id)
wait_cross_flag(V2C_id)        ← set_cross_flag("MTE3", V2C_id)
```

- C→V：C 写完 workspace（FIX 管线完成）后 set，V 等信号后读 workspace
- V→C：V 用完 workspace（MTE3 写完或读完）后 set，C 等信号后覆盖写 workspace
- **两个方向需要不同 flag_id**，不能复用

### 4.3 FA 的三缓冲 cross_flag

FA 用 3 个 workspace（Q/K/V），每个 workspace 双向通信 = 6 个 SEM id：

| flag_id | 方向 | 用途 |
|---------|------|------|
| SEM_WS1_C2V (0) | C→V | Q workspace 就绪 |
| SEM_WS1_V2C (1) | V→C | Q workspace 用完 |
| SEM_WS2_C2V (2) | C→V | K workspace 就绪 |
| SEM_WS2_V2C (3) | V→C | K workspace 用完 |
| SEM_WS3_C2V (4) | C→V | V workspace 就绪 |
| SEM_WS3_V2C (5) | V→C | V workspace 用完 |

每个方向都是模式 C（init借信用+节流set）。

### 4.4 cross_flag 的信用借贷

cross_flag 和核内 flag 一样需要 init 借信用。V 侧（消费者）需要先 set 一次给 C：

```python
with T.Scope("V"):
    T.set_cross_flag("MTE2", SEM_WS1_V2C)  # 借信用: 告诉C "workspace可用"
    T.set_cross_flag("MTE2", SEM_WS3_V2C)
    # ... 循环 ...
```

没有这个 init，C 第一轮 wait_cross_flag(SEM_WS1_V2C) 时 V 还没 set → C 阻塞 → V 也在等 C → 双向死锁。

### 4.5 cross_flag 的 clear

cross_flag 不需要像核内 flag 那样显式 clear_flag。核内 flag 的 clear 是等回 init 借的信用（避免影响下个 kernel），而 cross_flag 在 kernel 结束时自然消化——最后一轮 wait 消费完最后一个 set，信用归零。

如果是 CV 融合且 kernel 结束后还有 cross_flag 信用未消费，不影响下个 kernel（cross_flag 状态随 kernel 退出清理）。但核内 flag 必须 clear，因为核内 flag 是硬件全局状态，不清理会影响后续 kernel。

## 5. set_flag 的"自 set 自 wait"语义

Cube 四级里常见到这种写法：

```python
T.set_flag("m", "fix", s2c)    # M→FIX: L0C数据就绪
T.wait_flag("m", "fix", s2c)   # 等什么?
```

这不是"自己发信号自己等"。硬件上：
- `set("m","fix",eid)`：M 投令牌到 FIX 的信箱——"L0C 算完了，可以写 GM"
- FIX 写完 GM 后，硬件自动回发一个完成信号到 M 的信箱
- `wait("m","fix",eid)`：M 等 FIX 回的完成信号——"GM 写完了，可以覆盖 L0C"

set 是通知对方"数据就绪"，wait 是等对方"用完了"（回信号）。两个方向用同一组 `(src,dst,eid)` 标识——set 走 src→dst 方向，wait 等 dst→src 方向的回信号。

## 6. 常见 set/wait 语义错误

以下是与 set/wait **语义**直接相关的错误写法（死锁/容量/pass_configs 类错误见 [pitfalls.md](pitfalls.md) 速查表）：

| 错误 | 为什么错 | 正确做法 |
|------|---------|---------|
| set/wait 方向反了 | 消费者 set、生产者 wait，变成"先通知再生产" | 谁写完数据谁 set，谁要读谁 wait |
| src 管线写错 | wait 等了错误的管线，数据没写完就读 | set 的 src = 实际写完数据的管线 |
| eid 和轮转 index 不匹配 | 双缓冲 buf0 用了 eid=1，等错信号 | eid = buf_idx % stages |
| 自 set 自 wait 混淆 | 误以为 `set("m","fix",e)` 后 `wait("m","fix",e)` 是自发自等 | set 通知对方就绪，wait 等对方回信号，见 §5 |
