# 服务器代码修改与脱敏发布记录

核对日期：2026-10-16（北京时间）。GitHub 实时核对结果：`liuyuanshuai1/OpenAvatarChat:main` 与 `HumanAIGC-Engineering/OpenAvatarChat:main` 完全一致，双方提交均为 `8b7b3b45bca28ae9ab5fa72ee31bc03c5a99b08b`，ahead/behind 均为 0。

本分支基于该提交，整理当日从运行容器导出的实际源码。上传前重新执行快照校验：OAC 的 1868 项、TTS 的 280 项文件/链接全部一致，内容差异 0。再次连接服务器时 SSH 超时，因此本次发布代表 **2026-10-16 已校验快照**，不能证明快照导出后服务器未再变化。整理过程未修改服务器服务。

## 修改清单

| 部分 | 相对上游的修改 | 对应文件 |
|---|---|---|
| FlashHead | 独立 worker 进程、共享内存批次、真实预热、可选 compile、GPU uint8 转换、有界输出队列、generation 丢弃迟到结果、取消和帧节拍处理 | `src/handlers/avatar/flashhead/{avatar_handler_flashhead,flashhead_config,flashhead_processor,flashhead_worker}.py` |
| CosyVoice3 | 新增 HTTP 流式远程适配，文本合并/切分、音色缓存、取消控制线程、请求代号与 generation 校验，避免旧 PCM 进入新轮 | `src/handlers/tts/cosyvoice3_remote/tts_handler_cosyvoice3_remote.py` |
| 独立 TTS 服务 | 常驻 CV3/vLLM、NDJSON PCM、健康检查、取消接口、串行推理和断连后的生成器清理 | 本目录 `cosyvoice-service/serve_avatar.py` |
| RTC | 同步存储及合并唤醒桥接 asyncio，取消时原子清队列，发送前检查连接，硬件编码检测兼容和 NVENC 重建处理 | `src/handlers/client/rtc_client/{client_handler_rtc,thread_safe_queue}.py`、`src/service/rtc_service/rtc_stream.py` |
| 引擎诊断 | 输入/输出队列时长、等待、批次、断供统计；取消跨轮断供计数 | `src/engine_utils/media_metrics.py`、`src/chat_engine/core/chat_session.py` 及处理器 |
| 前端诊断 | `?diagnostics=1` 开启 WebRTC 接收/解码/缓冲与视频呈现诊断、标记及导出 | 本目录 `frontend-overlay/` |
| LLM | 请求超时从 5 秒改为 30 秒；此处采用容器有效版本，原工作目录仍是旧值 | `src/handlers/llm/openai_compatible/llm_handler_openai_compatible.py` |
| 旧 CV2 适配 | 将位置参数 `True` 改为显式 `stream=True` | `src/handlers/tts/cosyvoice/cosyvoice_processor.py` |
| Docker | 镜像源调整、uv 下载重试/超时参数、wheelhouse 构建目录、GPU/共享内存及 worker 启动配置 | `Dockerfile`、`run_docker_cuda128.sh` |
| 验证与说明 | 已有取消、队列、worker、音视频配对及诊断回归测试，双语 FlashHead 文档 | `tests/production_snapshot/`、`docs/` |

`cancelled = False` 在上游基准中已存在，未把它列为本次新增修复。修改范围同时包括此前部署修复与最近三项修复，不能把所有改动都归为最后一轮。

## 配置和重建边界

配置模板为 `config/chat_local_cosyvoice3_vllm_flashhead_worker.example.yaml`。使用前复制为不含 `.example` 的文件，填写自己的 LLM 地址和凭据，并准备模型、角色图片、证书与独立 TTS 服务。模板保留当时的性能参数：单会话、25 FPS、24 kHz 输出、process 模式、compile 开启、2 秒 FlashHead 缓冲。启动脚本默认选择复制后的配置。

运行树包含历史临时安装、挂载覆盖和直接注入的前端 dist；本分支没有完整 Docker 镜像、模型或离线依赖。`Dockerfile` 与启动脚本保留已观察到的构建/部署差异，**不能据此宣称从零构建即可复现当时镜像**。脚本使用的镜像标签需要自行构建或替换。`wheelhouse/.gitkeep` 只保留构建目录；没有上传二进制 wheels。

`oac-installed.txt` 和 `tts-installed.txt` 只保留可公开的版本 pin，剔除了本机路径和 URL 安装项，是环境观察记录，不是新的 lockfile。`cosyvoice-service/requirements.in` 是历史加速依赖增量，独立 TTS 还需要 CosyVoice 源码、模型、wetext 资源及正确的容器环境，不能单凭这个文件启动。

前端是独立 Git 子模块，基准为 `a6182afbda3f3b84a6608402a41e55a0c7bc6766`。本分支没有向另一个仓库写入不可访问的 gitlink，也没有上传编译 dist。`frontend-overlay/index.html` 与 `runtime-diagnostics.js` 对应开发侧补充，重新构建前可将它们分别合入子模块的 `src/renderer/index.html` 与 `src/renderer/public/runtime-diagnostics.js`，再按前端原流程构建。这两份文件不冒充容器的原始前端源码。

## 性能记录与限制

以下数值来自此次快照对应的已有测试记录，并非本次重新在线运行的结果，也不是最新官方纯净版的对照测试：

- FlashHead 完整语音批次：656 批平均约 489 ms、p95 542 ms、最大 605 ms；低于 24 帧/960 ms 的播放预算。
- 固定音频长播放记录断供 0，不代表所有真实对话场景都无卡顿。
- CV3 取消信号回调在整链路测试中小于 0.2 ms；旧计算清理仍约 2.2 秒。
- 客户端打断后旧声音尾巴仍约 1.7–1.9 秒；真实对话首声约 4–8 秒；接收端仍观察到约 300 ms 的抖动。
- process 模式只验收单会话；RTC 和引擎队列未全面有界。不能外推多会话或其他 GPU 的效果。

## 隐私处理

仅显式选择源码文本、测试和脱敏配置。未上传 `.env`、私钥/证书、容器 inspect 原文、服务器 IP/SSH 用户/本机路径、运行日志、真实聊天、音视频、模型、原始归档或备份文件。LLM 地址替换为 `llm.example.invalid`，密钥替换为占位符，启动脚本去除私人 NO_PROXY 地址。公开仓库已有的项目作者信息和公开依赖地址不属于新增服务器信息。

`publication-manifest.json` 记录发布源码及来源类别、脱敏状态和发布后 SHA256。代码文件取自容器快照；文档和测试是附加开发材料。诊断 JS 测试仅调整了源码定位路径，以便无需初始化子模块即可运行。

## 本次本地验证

详见同目录 `validation.md`。GPU 推理、完整镜像重建及当前服务器端到端验证需要可访问的服务器环境，未在此次发布中重跑。
