# 实时相位持久二进制传输方案

状态：待实施

适用硬件：SonicSurface 16×16，256 通道，32 档相位

目标：本机可微算法每次产生新相位后，以最低可控延迟发送并原子应用到 FPGA。

## 1. 目标与边界

本方案解决的是：

```text
实时目标/观测
    -> 本机 PDE 或响应基算法得到 256 路相位
    -> 相位量化与传输
    -> FPGA 原子切换
```

不要求提前知道完整运动路径，也不缓存完整周期。算法仍在本机运行，控制板不
重新求解 PDE。

本阶段不修改声学模型、优化目标、阵元映射或校准值。现有 HTTP 接口继续用于
状态查询、停止、固定图案和兼容操作；实时相位使用独立的持久 TCP 二进制连接。

## 2. 当前瓶颈

当前相位帧为：

```text
0xFE + 256 个相位档位 + 0xFD = 258 bytes
```

UART 使用 8N1，每字节占 10 bit。理论传输时间为：

```text
t_frame = 258 * 10 / baudrate
```

| 波特率 | 单帧理论时间 | 理论上限 |
|---:|---:|---:|
| 230400 | 11.20 ms | 89.3 fps |
| 460800 | 5.60 ms | 178.6 fps |
| 921600 | 2.80 ms | 357.2 fps |
| 1462857 | 1.76 ms | 567.0 fps |
| 3000000 | 0.86 ms | 1162.8 fps |

因此当前 `230400 baud` 下，`100 Hz` 已超过物理带宽。HTTP JSON、线程创建和
日志输出会进一步增加抖动，但不是 11.20 ms 下限的主要来源。

现有 `ThreadingHTTPServer` 还存在实时语义问题：当生产速度高于串口速度时，
请求可能等待串口锁。对实时触觉而言，过期相位没有价值，必须丢弃而不是排队。

## 3. 总体架构

```text
┌──────────────────┐
│ 实时目标 / 观测   │
└────────┬─────────┘
         v
┌──────────────────┐
│ 本机 PDE / 响应基 │
│ 输出 phases_rad   │
└────────┬─────────┘
         v
┌──────────────────┐
│ 客户端量化        │
│ logical uint8[256]│
└────────┬─────────┘
         │ persistent TCP :8766
         v
┌─────────────────────────────┐
│ 实时接收线程                 │
│ 校验协议、序号、CRC          │
│ latest-only slot，容量 = 1   │
└──────────────┬──────────────┘
               v
┌─────────────────────────────┐
│ 串口发送线程                 │
│ 校准 -> 通道映射 -> 0xFE帧   │
│ 过期帧不发送                 │
└──────────────┬──────────────┘
               v
┌─────────────────────────────┐
│ FPGA 双缓冲                  │
│ 收完整帧后 0xFD 原子提交     │
└─────────────────────────────┘
```

### 3.1 控制面

保留 `http://127.0.0.1:8765`：

- `GET /v1/status`
- `POST /v1/off`
- `POST /v1/pattern`
- 现有调制与频率扫描接口

HTTP 控制面不参与实时逐帧路径。

### 3.2 数据面

新增 `tcp://127.0.0.1:8766`：

- 单个持久连接；
- `TCP_NODELAY=true`；
- 二进制定长头；
- 相位负载为逻辑通道顺序的 `uint8[256]`；
- 服务端继续负责 host 校准和 FPGA 通道映射；
- 接收线程与串口线程分离；
- 待发送槽位永远只保存最新帧。

## 4. 二进制协议 v1

所有多字节整数使用网络字节序（big-endian）。

### 4.1 固定头

| 字段 | 类型 | 字节 | 说明 |
|---|---|---:|---|
| magic | `4s` | 4 | 固定为 `USPH` |
| version | `u8` | 1 | 当前为 1 |
| message_type | `u8` | 1 | 消息类型 |
| flags | `u16` | 2 | 保留，当前为 0 |
| sequence | `u32` | 4 | 单调递增帧序号 |
| payload_length | `u32` | 4 | 负载字节数 |
| produced_ns | `u64` | 8 | 客户端单调时钟时间戳 |
| crc32 | `u32` | 4 | 头前 24 字节与负载的 CRC32 |

固定头总计 28 字节。

### 4.2 消息类型

| 类型 | 编码 | 方向 | 负载 |
|---|---:|---|---|
| `HELLO` | `0x01` | 客户端 -> 服务端 | 能力与客户端名称 |
| `HELLO_ACK` | `0x02` | 服务端 -> 客户端 | 协议版本、通道数、配置摘要 |
| `PHASE_FRAME` | `0x10` | 客户端 -> 服务端 | 256 个逻辑相位档位 |
| `STATUS` | `0x11` | 服务端 -> 客户端 | 接收、应用、丢弃和错误计数 |
| `PING` | `0x20` | 双向 | 可为空 |
| `STOP` | `0x21` | 客户端 -> 服务端 | 可为空；立即关闭阵列 |
| `ERROR` | `0x7F` | 服务端 -> 客户端 | 错误码与 UTF-8 摘要 |

### 4.3 `PHASE_FRAME`

负载固定为 256 字节：

- `0..31`：32 档相位；
- `32`：关闭该通道；
- 逻辑通道顺序，不是 FPGA 物理地址顺序。

客户端只做：

```text
phases_rad -> floor(mod(phi, 2*pi) * 32 / (2*pi))
```

服务端负责：

1. 校验 256 个值均处于 `0..32`；
2. 在 `host` 校准模式下添加相位校正；
3. 按 `channel_order` 映射到 FPGA 地址；
4. 生成并发送 258 字节 FPGA 帧。

首版不做 5-bit 打包。256 字节负载简单、固定且容易诊断；5-bit 打包可在协议
v2 将负载降至 160 字节，但它不能降低当前 FPGA UART 的 258 字节发送量，
除非同步修改 FPGA 解码协议。

## 5. Latest-only 并发模型

服务端维护：

```text
accepted_sequence
applied_sequence
latest_pending_frame
received_count
applied_count
dropped_count
crc_error_count
serial_error_count
```

接收线程每收到合法帧：

1. 若序号不大于已接收序号，拒绝；
2. 若槽位已有未发送帧，将其计为 dropped；
3. 用新帧原子覆盖槽位；
4. 唤醒串口线程；
5. 立即继续接收，不等待串口写入。

串口线程：

1. 等待槽位非空；
2. 原子取走当前最新帧；
3. 应用校准与通道映射；
4. 写入完整 FPGA 帧并 flush；
5. 更新 `applied_sequence` 和延迟指标；
6. 立即检查是否出现更新帧。

严禁使用无界 FIFO。生产端达到 200 Hz、串口只能达到 70 Hz 时，系统应表现为
“丢帧但保持低延迟”，不能表现为“所有帧最终播放但越来越滞后”。

## 6. 所有权与安全状态

实时连接建立后，控制器进入 `STREAMING`：

- 仅允许一个实时生产者；
- `/v1/pattern` 与 `/v1/modulation` 返回 `409 Conflict`；
- `/v1/off` 始终具有最高优先级，可终止流并关闭阵列；
- TCP 断开后执行安全关闭；
- 超过 `watchdog_timeout_s` 未收到新帧时执行安全关闭；
- CRC、长度、版本或档位非法时拒绝该帧，不改变当前输出；
- 串口完整帧超时沿用现有“重开串口并完整重发一次”的恢复逻辑。

建议默认：

```yaml
realtime:
  host: "127.0.0.1"
  port: 8766
  watchdog_timeout_s: 0.5
  status_every_frames: 20
  socket_receive_buffer_bytes: 65536
```

## 7. 延迟指标

客户端记录：

- `phase_ready_ns`：可微算法完成相位；
- `socket_send_ns`：二进制帧提交。

服务端记录：

- `received_ns`：完整帧解析完成；
- `serial_start_ns`：开始写串口；
- `serial_done_ns`：flush 完成；
- `applied_sequence`：最近提交到 FPGA 的序号。

至少报告：

```text
compute_to_socket_ms
socket_to_serial_start_ms
serial_write_ms
phase_age_at_commit_ms
received / applied / dropped
effective_fps
```

## 8. 分阶段实施

### 阶段 A：持久二进制连接

不修改 FPGA 和波特率，先消除 JSON、HTTP 建连和错误排队：

- 新增二进制协议编解码；
- 新增持久 TCP 接收服务；
- 新增 latest-only 槽位与串口发送线程；
- 新增实时客户端；
- 保留现有 HTTP 控制面；
- 轨迹脚本和实时求解入口改用新客户端；
- 加入延迟、丢帧和错误统计。

预期：更新率仍受限于 89.3 fps，但延迟不会随负载累积，建议生产频率
`50..70 Hz`。

### 阶段 B：提高 UART 波特率

仓库官方实验代码记录了 `1462857 baud` 与 FPGA UART divisor 7 的组合。实施：

- 修改 FPGA `UARTReader` 分频参数；
- Quartus 重新综合并生成 JIC；
- USB 串口桥与 FPGA 同时切换波特率；
- 服务端配置改为新速率；
- 从 460800、921600 到 1462857 分级压力测试；
- 保持完整帧、CRC 和 latest-only 语义不变。

预期：`1462857 baud` 下理论 567 fps，工程目标为稳定 `300..500 Hz`。

### 阶段 C：独立硬件包络

空间相位更新和触觉包络必须解耦：

- 双缓冲只保存当前/下一空间相位；
- FPGA 独立计时器控制 200 Hz 输出门控；
- 相位更新不再上传关闭帧；
- 包络相位不因空间焦点变化而重置。

该阶段解决高速移动时的触觉连续性，不提高单帧 UART 原始带宽。

### 阶段 D：1 kHz 以上

若必须稳定达到 1 kHz：

- 258 字节帧至少需要 2.58 Mbaud；
- 3 Mbaud 仅有约 14% 理论余量；
- 优先改用 USB bulk、SPI 或并行接口；
- 或在 FPGA 中实现 5-bit 解包，将每帧降至 160 字节；
- 需要序号、CRC、ACK/状态回传和硬件级时钟验证。

## 9. 修改范围

### Python

- `control/sonicsurface/realtime_protocol.py`
  - 头部、消息、CRC32、增量解析器。
- `control/sonicsurface/realtime_server.py`
  - 持久连接、单生产者、latest-only 槽位。
- `control/sonicsurface/realtime_client.py`
  - 持久发送、序号、状态读取。
- `control/sonicsurface/protocol.py`
  - 增加从逻辑相位档位直接构造 FPGA 帧的接口。
- `control/sonicsurface/controller.py`
  - 增加量化帧提交、流所有权与统计。
- `control/sonicsurface/server.py`
  - 同时启动 HTTP 控制面与 TCP 数据面。
- `control/circle.py` 等实时入口
  - 使用持久客户端，不再逐帧 HTTP POST。

### FPGA（阶段 B/C）

- `Firmware/FPGA primary/src/QuadrupleBuffer.bdf`
  - UART divisor；
  - 独立包络门控连接。
- `Firmware/FPGA primary/src/Distribute.vhd`
  - 如需新增包络命令，扩展命令解析。
- 重新生成并物理验证 `.sof/.jic`。

## 10. 验收标准

### 功能

- [ ] 256 路逻辑相位经新旧协议生成完全相同的 FPGA 258 字节帧。
- [ ] 非法 magic、版本、长度、CRC 和相位档位被拒绝。
- [ ] 实时连接只允许一个生产者。
- [ ] 新帧覆盖待发送旧帧，槽位深度始终不超过 1。
- [ ] `/v1/off` 能抢占实时流并关闭阵列。
- [ ] 连接断开和 watchdog 超时后阵列关闭。
- [ ] HTTP 原有固定图案接口保持兼容。

### 性能

- [ ] `230400 baud`、生产端 60 Hz：连续运行 10 分钟无串口超时。
- [ ] `230400 baud`、生产端 200 Hz：出现 dropped，但延迟不持续增长。
- [ ] 当前速率下 `phase_age_at_commit` 的 p95 不超过 20 ms。
- [ ] 不发生超过两个相位周期的历史帧提交。
- [ ] 阵列更新期间不插入关闭帧。
- [ ] 阶段 B 在 921600 / 1462857 baud 下分别完成 10 分钟压力测试。
- [ ] 1.46 Mbaud 的稳定有效更新率达到至少 300 fps。

### 可观测性

- [ ] HTTP `/v1/status` 返回实时连接状态、最新接收/应用序号和丢帧数。
- [ ] 客户端每秒输出有效 FPS、p50/p95 延迟和 dropped。
- [ ] 串口恢复后序号继续单调递增，不重复提交未知 swap。

## 11. Todo

- [ ] 定义并实现 `realtime_protocol.py` 的 v1 编解码与 CRC。
- [ ] 在 `protocol.py` 增加 `compile_pattern_bins()`。
- [ ] 实现线程安全 latest-only 槽位。
- [ ] 实现独立串口发送线程和统计。
- [ ] 实现单生产者持久 TCP 服务。
- [ ] 实现持久 TCP 客户端与自动重连。
- [ ] 在主服务中同时启动 HTTP 与 TCP。
- [ ] 加入流所有权、409 冲突和 `/v1/off` 抢占。
- [ ] 接入一个轨迹脚本进行端到端验证。
- [ ] 接入实时 PDE 相位生产接口。
- [ ] 完成模拟串口正确性、丢帧和故障恢复验证。
- [ ] 完成 230400 baud 实机 10 分钟压力测试。
- [ ] 确认 Quartus 工程与硬件串口桥支持的高波特率。
- [ ] 分级验证 460800、921600、1462857 baud。
- [ ] 将独立 200 Hz 包络下沉到 FPGA。
- [ ] 更新控制文档和硬件烧录说明。

## 12. Commit 计划

1. `feat(control): 定义实时相位二进制协议`
2. `feat(control): 增加 latest-only 相位流服务`
3. `feat(control): 接入持久实时相位客户端`
4. `fix(control): 增加流所有权与故障安全关闭`
5. `perf(control): 增加实时链路延迟与丢帧统计`
6. `docs(control): 补充实时相位链路与压力测试说明`
7. `feat(fpga): 提升 UART 速率并加入独立包络门控`

## 13. 决策

首轮实施只完成阶段 A，保持 `230400 baud` 和现有 FPGA 固件不变。这样先验证
协议、并发、latest-only 语义和延迟测量，再决定是否烧录高波特率固件。

阶段 A 无法让当前硬件超过 89.3 fps；它解决的是排队、抖动和软件开销。
若目标是稳定超过 100 Hz，阶段 B 是必选项。
