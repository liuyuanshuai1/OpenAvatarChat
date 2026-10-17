# PR 拆分验证

源码来源：2026-10-16 已校验的运行容器快照。拆分验证：2026-10-17，macOS arm64，通过 uv 管理隔离 Python 3.11 环境。

| 检查 | 结果 |
|---|---|
| 最终分支全部已有 Python CPU 回归测试 | 36/36 通过 |
| 浏览器诊断 Node 测试 | 3/3 通过 |
| 各中间分支新增 Python 源码语法及 diff 空白检查 | 全部通过 |
| RTC 编码分支真实模块导入 | 通过；未将模块导入计为 GPU 编码验收 |
| RTC 队列分支对应回归 | 4/4 通过 |
| FlashHead 分支对应 CPU 回归 | 27/27 通过，含实际 spawn/共享内存、预热协议、超时及清理路径 |
| CV3 分支对应回归 | 3/3 通过，使用模拟 HTTP 响应检查取消协议 |
| 诊断分支对应回归 | Python 2/2、Node 3/3 通过 |
| 通用及服务器专用启动脚本 bash 语法 | 通过 |
| 最终运行源码与原汇总 PR 对比 | 12 个运行源码文件逐字节相同 |
| 原有 Python 测试函数及参数化装饰器 | 25 个全部保留，按模块重组；参数化后为 36 项 |
| 浏览器诊断测试文件 | 与原汇总 PR 逐字节相同，迁移目录 |
| 每个中间提交及最终发布文件隐私扫描 | 未发现待处理项，见 `privacy-scan.json` |

阶段验证的机器可读结果见 `stage-validation.json`；源码和测试完整性结果见 `split-verification.json`。

首次在受限沙箱执行时共享内存被系统拒绝；允许本机共享内存后已重新执行全部测试并通过，没有通过跳过测试或替换共享内存实现来规避。完整套件报告一个第三方 websockets 弃用警告。

测试环境主要版本：Python 3.11、torch 2.14.1、numpy 1.26.4、pydantic 2.10.6、gradio 5.9.1、fastrtc 0.0.33、aiortc 1.15.0、av 15.1.0。此环境用于 CPU 回归，不能视为 Linux 线上环境复现。

在具备上述依赖的环境中，可执行：

```sh
PYTHONPATH=.:src GRADIO_ANALYTICS_ENABLED=False HF_HUB_OFFLINE=1 \
  uv run --no-project --python .venv-test/bin/python python -m pytest -q tests/regression
node --test tests/regression/runtime-diagnostics.test.cjs
```

此次拆分未修改服务器，也未重跑真实 GPU 模型推理、硬件编码性能、完整镜像构建或线上端到端验收。2026-10-16 的 SSH 复核曾超时，本次没有以重新访问服务器作为测试通过条件。
