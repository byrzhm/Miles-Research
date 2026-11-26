# 文档索引

本目录放归档源码的专题解析。所有内容基于 `releases/<tag>/` 中的官方 tag 快照静态阅读，不执行归档代码。

| 文档 | 内容 |
| --- | --- |
| [Miles Rollout 源码解析](miles-rollout.md) | 以 v0.2.0 为基线，逐层解释 Rollout 的控制面 / 数据面、生成链路、TITO 会话、完全异步缓冲、正确性与 checkpoint 机制，并给出 v0.1.0 → v0.2.0 的逐版本演进 |

## 阅读建议

- 想先建立整体图景：读 `miles-rollout.md` 的 [第 1 节](miles-rollout.md#1-rollout-在-rl-循环中的位置) 与 [第 2 节](miles-rollout.md#2-进程与组件拓扑)。
- 想改代码 / 接自定义组件：读 [第 5 节](miles-rollout.md#5-rollout-函数-api) 与 [第 15 节](miles-rollout.md#15-配置与扩展点)。
- 想复现某个版本的行为：直接读对应 `releases/<tag>/`，再用 [第 16 节](miles-rollout.md#16-各版本演进) 对照差异。

版本级摘要（发布日期、PR 数、依赖锁定、升级提示）见仓库根目录的 [README](../README.md)。
