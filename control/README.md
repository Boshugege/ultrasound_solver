# SonicSurface 本地控制服务

本目录提供独立的客户端/服务端控制链路。服务端独占串口并把收到的
256 通道相位发送给 SonicSurface；客户端只负责读取相位结果并通过 HTTP
提交，不直接访问硬件。

## 修正后的 PDE 求解器

当前 `front_focus_alg.py` 默认使用
`src/examples/pde_homogeneous_16x16.yaml`：
官方参考坐标 + `sinc_neumann` 有限孔径通量源 + 均匀介质有限体积 PDE。
阵列底面为反射挡板，其余五面使用开放边界和 PML；默认网格为 1/3 mm。
相位量化遵循官方截断规则，保留官方坐标表中的约 2 mm 原点偏移。
配置的物理后端仍为原有 `cudss_hybrid_direct`。均匀自由场使用
内置表面凝聚算法加速同一个 PDE 的目标响应，不存在第二套物理后端。

`front_focus_alg.py`、`circle.py` 和 `line.py` 均使用这一配置。
官方传播公式仅用于离线对照脚本，不再注册为项目求解后端。

```powershell
conda activate srt
python control/front_focus_alg.py --offline
```

该命令不访问真实硬件；`front_focus_alg.py` 未带 `--offline`
时仍执行原有发射流程。
PDE 目标响应缓存在 `outputs/pde_homogeneous_16x16/target_responses.npz`，
缓存指纹覆盖网格、几何、介质、声源、边界、PML 和实现代码。
场单位为归一化 `1/m`，不能解释成经过硬件标定的绝对 Pa。
物理修改和代码整理范围见 `docs/PHYSICAL_MODEL_ALIGNMENT_AND_BACKEND.md`。

```text
逆向优化结果 (.npy/.npz)
        |
        v
control/client.py
        |  HTTP JSON
        v
control/server.py
        |  映射 -> 校正 -> 32 档量化 -> UART
        v
SonicSurface FPGA
```

## 快速使用（Conda srt + COM5）

以下命令均在仓库根目录 `c:\workspace\ultrasound_solver` 执行。

首先激活环境并确认串口可见：

```powershell
conda activate srt
python -m serial.tools.list_ports
```

确认输出中存在 `COM5`。然后检查 `control/sonicsurface.example.yaml`：

```yaml
serial:
  port: "COM5"
  baudrate: 230400

channel_order: "sonicsurface"

calibration:
  mode: "firmware"
  phase_corrections_bins: "zeros"
  verified: true
```

只有确认板卡烧录的是仓库内已校准的
`thirdparty/SonicSurface/Firmware/FPGA primary/output_file.jic` 时，才能使用
上述 `firmware + verified: true`。如果烧录的是
`FPGAPrimaryNoCalib.jic`，必须先完成标定，并改用主机校正或重新编译固件。

打开终端 1，启动实际串口服务：

```powershell
conda activate srt
python control\server.py --config control\sonicsurface.example.yaml --live
```

打开终端 2，先检查状态并发送关闭命令：

```powershell
conda activate srt
python control\client.py status
python control\client.py off
```

发送全阵元零相位测试：

```powershell
python control\client.py uniform 0
```

以 `200 Hz` 波群频率、`50%` 占空比调制零相位图案，持续 3 秒：

```powershell
python control\client.py modulate `
  --uniform-phase 0 --frequency 200 --duty 0.5 --duration 3
```

发送逆向优化结果文件：

```powershell
python control\client.py send `
  outputs\<场景>\<算法>\result_best_phases_rad.npy
```

也可以直接指定结果目录，客户端会优先选择最佳相位文件：

```powershell
python control\client.py send outputs\<场景>\<算法>
```

以指定波群频率调制逆向优化结果：

```powershell
python control\client.py modulate `
  outputs\<场景>\<算法> --frequency 200 --duty 0.5
```

不提供 `--duration` 时会持续调制，直到执行 `off`、发送新的静态图案或启动
另一组调制。这里调整的是作用于 40 kHz 超声载波的包络/波群频率，不会改变
换能器的 40 kHz 载波频率。

周期扫描一段频率并实时显示当前值：

```powershell
python control\frequency_sweep.py
```

运行前直接修改 `control/frequency_sweep.py` 顶部的实验变量：

```python
SERVICE_URL = "http://127.0.0.1:8765"
MIN_FREQUENCY_HZ = 20.0
MAX_FREQUENCY_HZ = 300.0
SWEEP_PERIOD_S = 10.0
DUTY_CYCLE = 0.5
DURATION_S = None
REPORT_INTERVAL_S = 0.2
```

其中 `DURATION_S = None` 表示持续运行，直到按 `Ctrl+C`；也可以改为
`60.0`，表示扫描 60 秒后自动停止。程序默认使用：

- 场景配置：`src/examples/front_center_free_field_16x16.yaml`
- 相位文件：
  `outputs/front_center_free_field_16x16/result_best_phases_rad.npy`
- 目标点：阵列中心正前方 `[0.08, 0.08, 0.12] m`
- 边界：板面 `-z` 保持 `reflecting`，其余五面 `open`
- 障碍物：无

相位文件不存在时，程序会先调用当前 Conda 环境的 Python 运行伴随 Adam
优化并生成文件。扫描采用三角波规律：频率从最低值连续升至最高值，再降回
最低值；`SWEEP_PERIOD_S` 表示一次完整往返的时间。终端持续输出当前包络
频率和已完成周期数。按 `Ctrl+C` 会请求服务端停止扫描并关闭全部阵元。

实验结束后关闭全部阵元：

```powershell
python control\client.py off
```

## 圆周运动 Demo

`control/circle.py` 将焦点放在与阵列平行的圆周上：

- 阵列边长：`0.16 m`
- 圆心：阵列正中央 `[0.08, 0.08] m`
- 高度：`z = 0.06 m`
- 半径：阵列边长的二十分之一，即 `0.008 m`
- 离散点数：16
- 播放速度：1 帧/秒

参数集中在 `control/circle.py` 顶部：

```python
BOARD_SIDE_M = 0.16
TARGET_HEIGHT_M = 0.06
CIRCLE_RADIUS_M = BOARD_SIDE_M / 20.0
PATH_POINT_COUNT = 16
OPTIMIZATION_ITERATIONS = 120
PLAYBACK_HZ = 1.0
LOOP_PLAYBACK = True
ENVELOPE_FREQUENCY_HZ = 200.0
ENVELOPE_DUTY_CYCLE = 0.5
```

运行：

```powershell
conda activate srt
python control\circle.py
```

首次运行会：

1. 使用 `CirclePath` 将圆周离散成 16 个目标点。
2. 复用官方自由场解析响应接口。
3. 对每个目标点计算几何相位并评估；选择迭代算法时才使用相位热启动。
4. 将所有目标点和对应的 256 通道相位保存到：
   `outputs/circle_front_16x16/circle_phases.npz`。
5. 连接服务端并依次发送相位，使焦点沿圆周循环移动。

`ENVELOPE_FREQUENCY_HZ` 控制圆周运动时的触觉包络频率，可直接修改为
`40.0`、`100.0`、`200.0` 等；设为 `None` 时不使用包络调制。
`ENVELOPE_DUTY_CYCLE` 控制发射占空比。

相位缓存已经存在且参数未改变时，之后运行会直接进入播放，不再计算。修改
路径参数、优化配置或算法/几何实现后缓存会自动失效；也可将 `FORCE_RECOMPUTE` 改为 `True`
强制重新计算。按 `Ctrl+C` 会停止播放并关闭全部阵元。

## 直线前后运动 Demo

`control/line.py` 让焦点沿阵列中心法线往返移动：

```text
0.12 m → 0.03 m → 0.12 m → ...
```

直接运行：

```powershell
conda activate srt
python control\line.py
```

常用参数位于文件顶部：

```python
START_HEIGHT_M = 0.12
END_HEIGHT_M = 0.03
PATH_POINT_COUNT = 16
PLAYBACK_HZ = 5.0
ENVELOPE_FREQUENCY_HZ = 200.0
ENVELOPE_DUTY_CYCLE = 0.5
```

程序只计算 16 个唯一高度点，播放时按正序和反序拼成往返轨迹，避免在两个
端点重复停顿。首次运行生成：

```text
outputs/line_front_16x16/line_phases.npz
```

之后直接读取缓存并发送。设置 `ENVELOPE_FREQUENCY_HZ = None` 可关闭包络，
设置 `FORCE_RECOMPUTE = True` 可强制重新计算全部高度点。

## 文件说明

```text
control/
  client.py                     客户端入口
  circle.py                     圆周路径预计算与循环播放
  frequency_sweep.py            包络频率周期扫描
  front_focus_alg.py            PDE 正前方聚焦与离线验证
  line.py                       中心轴直线往返路径
  server.py                     服务端入口
  sonicsurface.example.yaml     配置模板
  sonicsurface/
    protocol.py                 通道映射、量化和帧编码
    phase_source.py             求解器相位结果加载
    controller.py               dry-run/串口控制器
    server.py                   HTTP 服务
    client.py                   客户端命令
  thirdparty/SonicSurface/      官方 SonicSurface 仓库
```

## 协议依据

实现直接依据：

- `thirdparty/SonicSurface/ControlSoftware/Python/SonicSurface.py`
- `thirdparty/SonicSurface/MyPatterns/my_patterns.py`
- `thirdparty/SonicSurface/Firmware/FPGA primary/src/Distribute.vhd`

板卡使用 `230400 baud, 8N1`。一帧共 258 字节：

```text
0xFE + 256 个相位字节 + 0xFD
```

相位字节 `0..31` 对应一个周期的 32 个离散档位，`32` 表示关闭通道。
`0xFD` 交换 FPGA 双缓冲，使整幅相位同时生效。

## 1. 准备配置

复制 `control/sonicsurface.example.yaml` 后修改。默认配置：

- 使用官方 `EMITTERS_ORDER` 通道映射。
- 使用 FPGA 固件内置校正。
- `calibration.verified` 为 `false`，禁止真实串口发送。

校正只能存在一处：

- `mode: firmware`：FPGA 的 `PHASE_CORRECTION` 已写入校正，主机不再补偿。
- `mode: host`：用于无校正固件，填写 256 项
  `phase_corrections_bins`，服务端发送前补偿。

第三方仓库的 `FPGAPrimaryNoCalib.jic` 明确是不含校正的预编译固件。不能在
不确认当前固件的情况下把配置标记为已验证。

## 2. Dry-run 联调

先启动服务端。此命令不会打开串口：

```powershell
python control\server.py `
  --config control\sonicsurface.example.yaml
```

另开终端查询状态：

```powershell
python control\client.py status
```

发送全阵元零相位测试帧：

```powershell
python control\client.py uniform 0
```

关闭全部通道：

```powershell
python control\client.py off
```

dry-run 返回 `"accepted": true` 和 `"transmitted": false`，说明数据已通过
校验并成功编码，但没有访问硬件。

## 3. 波群频率调制

当前已校准的 primary 固件不支持触觉调制固件的 `101xxxxx` 调频命令。
本服务采用官方示例 `TestBeamStearingWithAudibleMod.py` 和
`MyPatterns/my_patterns.py` 使用的双缓冲方案：

1. 将目标相位和全关闭图案预装进 FPGA 的两个缓冲区。
2. 服务端按指定频率发送单字节 `0xFD`，在两个缓冲区之间切换。
3. 调制结束、收到 `off` 或发生异常时，服务端发送全关闭帧。

支持范围：

- `--frequency`：`1..500 Hz`，默认 `200 Hz`。
- `--duty`：`0.05..0.95`，默认 `0.5`。
- `--duration`：可选，单位秒；省略表示持续运行。

Windows 上的软件定时不是硬实时。需要严格稳定的高频包络时，应改用
`FPGATactileModulation.jic` 及其 FPGA 内部调制命令，但该固件的相位校准值
必须针对当前阵列重新确认。

## 4. 发送逆向优化结果

直接指定推荐的最优相位：

```powershell
python control\client.py send `
  outputs\<场景>\<算法>\result_best_phases_rad.npy
```

也可以指定包含标准结果的目录。客户端按以下顺序自动选择：

1. `result_best_phases_rad.npy`
2. `result_final_phases_rad.npy`
3. `result.npz` 中的 `best_phases`
4. `result.npz` 中的 `phases`

```powershell
python control\client.py send outputs\<场景>\<算法>
```

指定 NPZ 键：

```powershell
python control\client.py send result.npz --npz-key best_phases
```

所有输入必须是单位为弧度、形状为 `(256,)` 的有限浮点数组。旧的 12×12
结果形状为 `(144,)`，客户端会明确拒绝，不能直接发送到 16×16 板卡。

## 5. 真实连接板卡

1. 用 USB 转串口连接电脑与控制板，确认公共地线、电平和供电正确。
2. 在 Windows 设备管理器确认串口，例如 `COM7`。
3. 确认 FPGA 已烧录 SonicSurface 兼容固件。
4. 确认固件是否内置相位校正，并按实际情况设置 `calibration.mode`。
5. 验证 256 通道顺序和校正结果。
6. 将配置中的 `serial.port` 改为实际端口。
7. 只有完成上述检查后，设置 `calibration.verified: true`。
8. 使用 `--live` 启动服务：

```powershell
python control\server.py `
  --config control\sonicsurface.local.yaml `
  --live
```

服务端日志会记录每次相位提交的来源、是否真实发送和帧 SHA-256。停止实验前
先执行：

```powershell
python control\client.py off
```

## 局域网使用

默认只监听 `127.0.0.1`。若客户端在另一台可信电脑上，可显式监听局域网：

```powershell
python control\server.py `
  --config control\sonicsurface.local.yaml `
  --host 0.0.0.0 --live
```

客户端指定服务端地址：

```powershell
python control\client.py `
  --service-url http://192.168.1.20:8765 status
```

当前接口没有身份认证，不应暴露到公网。需要跨机器使用时，应通过 Windows
防火墙只允许可信局域网主机访问端口 `8765`。

## HTTP 接口

```text
GET  /v1/status
POST /v1/pattern
POST /v1/modulation
POST /v1/frequency-sweep
POST /v1/off
```

`/v1/pattern` 请求：

```json
{
  "phases_rad": [0.0, 0.1, 0.2],
  "source": "outputs/example/adjoint/result_best_phases_rad.npy"
}
```

实际数组必须包含 256 项。数值表示弧度；`null` 可用于关闭单个通道。

`/v1/modulation` 在相同相位数组之外接受：

```json
{
  "phases_rad": [0.0, 0.1, 0.2],
  "frequency_hz": 200.0,
  "duty_cycle": 0.5,
  "duration_s": 3.0,
  "source": "result_best_phases_rad.npy"
}
```

实际数组同样必须包含 256 项。`duration_s` 可以省略，此时调制持续到
`POST /v1/off` 或下一次相位提交。

`/v1/frequency-sweep` 使用三角波周期改变包络频率：

```json
{
  "phases_rad": [0.0, 0.1, 0.2],
  "min_frequency_hz": 20.0,
  "max_frequency_hz": 300.0,
  "sweep_period_s": 10.0,
  "duty_cycle": 0.5,
  "duration_s": 60.0,
  "source": "result_best_phases_rad.npy"
}
```
