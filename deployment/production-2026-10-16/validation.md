# 本次整理时的验证

日期：2026-10-16。

| 检查 | 结果 |
|---|---|
| GitHub fork 与上游 main 实时比较 | identical，ahead 0 / behind 0，基准 `8b7b3b45` |
| 已导出 OAC/TTS 源码快照重新校验 | 1868 + 280 项文件/链接，差异 0 |
| `node --test tests/production_snapshot/runtime-diagnostics.test.cjs` | 3/3 通过，包含不记录聊天内容/ICE 地址检查 |
| 两个已有队列测试隔离执行 | 2/2 通过：时长统计/清空，跨线程唤醒/取消后旧数据不重入 |
| Python AST 语法解析 | 220 个源码和测试文件通过 |
| `bash -n run_docker_cuda128.sh` | 通过 |
| `git diff --check` | 通过 |
| 显式发布文件隐私扫描 | 见 `privacy-scan.json`；只报告位置/类型，不输出候选敏感值 |
| 当前服务器 SSH 复核 | 连接超时，不能确认快照导出后的线上变化 |
| 完整 Python pytest/GPU/镜像/线上端到端 | 未重跑；本机虚拟环境缺少 pytest 及完整运行依赖 |

队列测试从已有测试文件提取对应函数 AST，直接使用发布代码，以标准库 `unittest.TestCase.assertRaises` 提供异常断言；没有通过伪造队列实现或模拟运行时来代替被测代码。其余依赖 GPU/完整应用栈的测试文件随分支保留，不能把语法通过计为运行通过。

Git 提交仅包含显式审查的文本文件和空 wheelhouse 目录标记；原始快照、模型、容器元数据和日志没有被加入提交。
