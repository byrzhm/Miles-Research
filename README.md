# Miles Research

本仓库归档 **Miles** 全部官方 Release 的源码，并梳理其从 v0.1.0 到 v0.2.0 的版本演进，供离线阅读、版本对比与研究使用。

## Miles 是什么

[Miles](https://github.com/radixark/miles) 是面向 LLM / VLM 大规模后训练的企业级强化学习框架，采用 Apache-2.0 许可。它源自 [slime](https://github.com/THUDM/slime) 并与其持续协同演进：以 [SGLang](https://github.com/sgl-project/sglang) 承担高吞吐 rollout，以 Megatron-LM / FSDP2 / [torchtitan](https://github.com/pytorch/torchtitan) 承担训练，覆盖全异步 RL、TITO 多轮会话、R3 路由重放、低精度训练、LoRA / Multi-LoRA、容错与多硬件后端等能力。

- 官网：<https://miles.radixark.com/>
- 文档：<https://miles.radixark.com/docs>
- v0.1 技术报告：<https://arxiv.org/abs/2609.08368>
- v0.1 发布博客：<https://lmsys.org/blog/2026-08-18-miles-v0-1>

## 归档说明

- 上游 GitHub Release 页面不附带二进制附件，因此归档对象是各 tag 的 GitHub 源码包（`https://codeload.github.com/radixark/miles/tar.gz/refs/tags/<tag>`）。解压时去掉了顶层 `miles-<version>/` 目录，文件内容未做修改。
- 下载与解压时间：**2026-11-26**。截至该日，上游共有 5 个正式 Release，无预发布版本。
- `releases/<tag>/` 是可直接阅读的 tag 快照，未编译、未安装依赖。每个版本都包含 `release-lock.json`，记录锁定的 SGLang 与 Megatron-LM commit；镜像与运行依赖见各自的 `docker/`、`requirements.txt`。
- `releases/v0.2.0/3rdparty/miles-ext-kernels/` 已按 tag 锁定的子模块 commit `34da9a8`（2026-10-22）补全，来源是 [radixark/miles-ext-kernels](https://github.com/radixark/miles-ext-kernels)。它是可选组件：Miles 默认不导入，只有显式 `--use-miles-ext-kernels`（配合 `cake` kernel 后端）时才会用到；该 commit 仅有包骨架，尚未包含实际 kernel。其余 4 个版本没有 submodule。
- 各版本目录保留上游自己的 `README.md` 与 `LICENSE`（v0.1.2 起另有 `NOTICE`）。Miles 使用 **Apache-2.0** 许可。

### 版本与目录

| 版本 | 发布日期（UTC） | 演进主题 | 本地目录 | 上游 Release | 解压后规模 |
| --- | --- | --- | --- | --- | --- |
| v0.1.0 | 2026-08-18 | 首个正式版，奠定全栈 RL 基座 | [releases/v0.1.0](releases/v0.1.0) | [v0.1.0](https://github.com/radixark/miles/releases/tag/v0.1.0) | 25 MB / 1,740 文件 |
| v0.1.1 | 2026-09-26 | 前沿模型、Multi-LoRA/Tinker、Agent 生态 | [releases/v0.1.1](releases/v0.1.1) | [v0.1.1](https://github.com/radixark/miles/releases/tag/v0.1.1) | 28 MB / 2,238 文件 |
| v0.1.2 | 2026-10-09 | torchtitan、Kubernetes、多策略训练 | [releases/v0.1.2](releases/v0.1.2) | [v0.1.2](https://github.com/radixark/miles/releases/tag/v0.1.2) | 34 MB / 2,931 文件 |
| v0.1.3 | 2026-10-23 | 原生 LoRA、P2P 重构、容错与完全异步 | [releases/v0.1.3](releases/v0.1.3) | [v0.1.3](https://github.com/radixark/miles/releases/tag/v0.1.3) | 43 MB / 3,288 文件 |
| v0.2.0 | 2026-11-12 | 多硬件、多模态、类型化配置、TPU | [releases/v0.2.0](releases/v0.2.0) | [v0.2.0](https://github.com/radixark/miles/releases/tag/v0.2.0) | 50 MB / 4,038 文件 |

### 版本节奏

| 版本 | 距上一版 | PR 数 | 贡献者数 |
| --- | --- | --- | --- |
| v0.1.0 | — | — | — |
| v0.1.1 | 39 天 | 472 | 31 |
| v0.1.2 | 13 天 | 417 | 17 |
| v0.1.3 | 14 天 | 287 | 17 |
| v0.2.0 | 20 天 | 400 | 20 |

PR 数与贡献者数取自各版本 Release Notes；v0.1.0 是首个版本，未给出对应统计。Miles 在不到三个月的时间里连续发布 5 个版本，节奏非常密集。

### 各版本锁定的上游依赖

| 版本 | sglang-miles | Megatron-LM `miles-main` |
| --- | --- | --- |
| v0.1.0 | `cb05a44f35` | `37f91617b9` |
| v0.1.1 | `880e3d2453` | `f148a32b43` |
| v0.1.2 | `b84bec5761` | `fd15ee20a4` |
| v0.1.3 | `3a938213c1` | `2fbe25a442` |
| v0.2.0 | `fd017154ad` | `809217f40d` |

完整 commit 见各版本目录下的 `release-lock.json`。v0.2.0 还锁定子模块 miles-ext-kernels 于 `34da9a8da9`（可选，默认不启用）。

## 文档

- [docs/](docs/)：归档源码的专题解析。
  - [Miles Rollout 源码解析](docs/miles-rollout.md)：以 v0.2.0 为基线，解释 Rollout 的控制面 / 数据面、生成链路、TITO 会话、完全异步缓冲、正确性与 checkpoint 机制，并给出 v0.1.0 → v0.2.0 的逐版本演进。

## 版本演进

### v0.1.0 — 首个正式版本：生产级全栈 RL 基座

**2026-08-18 · tag `v0.1.0` · commit `78527b9`**

- **定位**：Miles 的首个版本化发布，在 slime 的基础上发展为生产级后训练系统，确立 SGLang rollout + Megatron-LM / FSDP2 + Ray 编排的主线架构。
- **性能基座**：全异步 RL 解耦 rollout 与训练；GLM-5.2 744B 参考任务在 64×GB300 上保持约 4.5 分钟/训练步，前缀缓存命中率 96%。
- **正确性机制**：TITO（Token-In-Token-Out）会话服务器保证多轮 Agent 会话的 tokenization 一致；R3（Rollout Routing Replay）消除 MoE rollout 与训练之间的路由偏差；同时提供 On-Policy Distillation 与 Zero-KL 对齐。
- **权重同步与显存**：P2P RDMA 把 1T 参数模型的权重更新时间从 53.3 s 降到 7.2 s；disk-delta 把同步负载从 62.4 GB 降到 0.69–0.83 GB；支持 NVMe 优化器状态流式读写。
- **精度与 LoRA**：提供 NVFP4、MXFP4、MXFP8、FP8、INT4 QAT 训练配方；GLM-5.2 744B LoRA 只有 212 MB 可训练参数，占 BF16 基座的 0.014%。
- **模型与硬件**：支持 Kimi-K3、DeepSeek-V4、Inkling、Qwen3.8、Nemotron 3 Ultra、DeepSeek-V3.2、Kimi-K2.6、GLM-5.2、Gemma-4、GPT-OSS 等；硬件覆盖 NVIDIA H100 / H200 / B200 / B300 / GB200 / GB300 与 AMD MI300X / MI355X。
- **生态**：环境集成 Harbor、HUD、NeMo Gym、OpenEnv、Prime Intellect Verifiers；扩散模型 RL 放在独立仓库 [miles_diffusion](https://github.com/radixark/miles_diffusion)。
- **镜像**：同时发布 CUDA 13 与 CUDA 12 版本（`v0.1.0` / `v0.1.0-cu12`）。

### v0.1.1 — 前沿模型与 Agent 生态扩张

**2026-09-26 · tag `v0.1.1` · commit `2806267`**

- **规模**：472 个 PR、31 位贡献者，是 PR 数最多的一个版本。
- **模型**：Kimi-K3 从独立镜像并入主镜像；新增 Qwen3.8-Flash-Next、GLM-5.3-Flash；DeepSeek-V4 增加 Megatron 原生实现、THD packed sequence + 上下文并行，以及 MXFP4→FP8 转换；修复 Nemotron-H 的 Mamba 卷积权重加载与同步。
- **算法**：引入 sampling-support replay——top-p / top-k 采样时，训练侧只在实际被采样到的候选集上计算 log-prob，而不是全词表。
- **LoRA 服务化**：Multi-LoRA v2 与 Tinker 兼容网关，让多个隔离的训练租户共享一个基座模型，各自拥有优化器状态与 checkpoint。
- **Agent 训练**：session server 支持 Anthropic Messages API（`/v1/messages`），Claude Code 等 harness 可保持 TITO 直接训练；Harbor 任务可在 E2B、Daytona 沙箱中以进程内方式运行。
- **性能**：colocated 训练峰值内存从 324 GB 降到 257 GB；Adam 与 Muon 优化器状态均可流式落 NVMe；策略 logits 保持 BF16，前反向显存从 768 MiB 降到 304 MiB；NVFP4 fake-QAT 融合算子减少量化往返。
- **工程重构**：rollout 拆分为 InferenceController + RolloutExecutor，训练改为 cell-based TrainerController，全部权重传输协议统一到 `WeightUpdater` / `HfWeightIterator`。
- **升级提示**：版本化镜像只发布 CUDA 13；移除外部 router 模式；`--control-server-port` 更名为 `--api-server-port`；Multi-LoRA v1 及其参数被移除；覆盖写 checkpoint 会先删除旧目录。

### v0.1.2 — 多后端与平台化

**2026-10-09 · tag `v0.1.2` · commit `a22fd82`**

- **规模**：417 个 PR、17 位贡献者。
- **训练后端**：torchtitan 成为继 Megatron-LM、FSDP2 之后的第三个训练后端；三个后端共用 `training_utils`，统一数据处理、loss、metrics、replay、checkpoint 与权重同步。
- **基础设施**：引入实验性 Kubernetes 原生后端（`--cluster-backend kubernetes`），不再需要外层 Ray 集群；支持拆分部署、热重启、单任务多策略训练，以及接入外部已启动的 SGLang 引擎。
- **算法**：新增 `--loss-type score_centering`，用采样器 top-k 概率修正 rollout 策略落后于训练策略时的梯度漂移。
- **模型**：DeepSeek-V4.1 Flash 与 MiMo-V2.6-Flash-RL（309B MoE）进入主线；DeepSeek-V3.2 / GLM-5.x 支持原生 Megatron DSA；支持 NVIDIA Vera Rubin。
- **性能**：GLM-5 / DeepSeek-V3.2 / V4 / V4.1 共用一套 DSA 稀疏注意力 kernel；Qwen3.5、Kimi-K3、GLM-5.3-Flash 的线性注意力改为 TP 分头并配更快反向 kernel；DeepSeek-V4 在 CP 下均衡 indexer 负载；新增 `broadcast_packed` 权重同步模式。
- **升级提示**：启动脚本统一为 `prepare` / `train` / `full-train`；`--rollout-external` 移除；Agent 函数默认在子进程运行，参数与返回值需可 pickle；会话回复不再携带 `meta_info`，logprobs 需显式请求；停止自动构建 CUDA 12 镜像；Ray 限制在 `<2.59.0`；版权声明移入 `NOTICE`。

### v0.1.3 — 稳定性、容错与原生 LoRA

**2026-10-23 · tag `v0.1.3` · commit `74f47d3`**

- **规模**：287 个 PR、17 位贡献者。
- **原生 LoRA**：Megatron raw 模式下用一套共享实现覆盖 Llama、Qwen2/3、Qwen3.5/3.6、GLM-4/4.7/5.x、DeepSeek-V3/V3.2、Kimi-K2.5/K3、Inkling 等家族，不再依赖 Megatron-Bridge；GLM-5.3-Flash 是首个落地模型。
- **P2P 权重传输重构**：发送端不再保存完整模型的 CPU 副本，改用两个可复用的 pinned buffer；注册内存从 28.45 GiB 降到 2.00 GiB，首次更新（含建连）从 99.5 s 降到 3.9 s；写失败会使整个更新失败，并支持逐写校验和。
- **权重更新容错**：`--ft-components weight_update` 可隔离权重更新失败的 rollout cell，其余 cell 继续升级；训练 cell 卡死或中途死亡可被放弃而不阻塞全局。
- **完全异步增强**：`--fully-async --colocate` 支持生成与训练共享 GPU，最低单卡可跑；完全异步运行的 buffer、重试队列与在途工作均可 checkpoint / resume；新增 prompt KV 陈旧度指标与 namespaced radix cache。
- **性能**：融合 Triton log-prob + entropy kernel（`--log-probs-backend fused`）把 trainer 峰值内存从 117.3 GiB 降到 79.0 GiB，actor 更新吞吐提升约 20%；session server 从约 46 turns/s 提升到 75 turns/s。
- **依赖栈**：SGLang v0.5.22、torch 2.14、Miles 自构建 Transformer Engine 2.17.0+miles、transformers 5.19.0。
- **升级提示**：v0.1.2 的 rollout 侧 checkpoint 无法在 v0.1.3 续训；`DataSource` API 变更（移除 `add_samples`）；自定义 fully-async buffer 的 `get()` 改为批量接口；MTP 训练需要显式 `--enable-mtp-training`；容错默认要求 P2P 权重传输；per-token loss 的梯度尺度改为 DP 总 token 均值。

### v0.2.0 — 多硬件、多模态与平台成熟

**2026-11-12 · tag `v0.2.0` · commit `8e419cc`**

- **规模**：400 个 PR、20 位贡献者。
- **多硬件**：扩展至 NVIDIA（A100–GB300）、AMD ROCm（MI300X–MI355X）、Google TPU v7x，AWS Trainium 即将支持；torchtitan 成为成熟的第三训练后端。
- **多模态 RL**：`--multimodal-token-space-processing` 让 rollout 与 trainer 通过同一 media processor 看到完全一致的 token、mRoPE 位置与视觉特征；提供 TimeLens 视频、geo3k 图像、AVQA 音频等配方。
- **模型**：Muse Spark Open（816B omni MoE）进入主镜像；新增 MiniMax-M3 / MiniMax-M3.1-Flash；DeepSeek-V4.1-Flash 支持原生 LoRA；GLM-5.3-Flash 增加全参数 torchtitan 路径。
- **LoRA 体系**：统一的后端无关 adapter checkpoint 目录格式，可在不同 TP/PP/EP/DP 布局间恢复；原生 Multi-LoRA 接入 Tinker 网关（SDK 0.36）；新增 HTTP-LoRA 权重传输。
- **数据与传输**：`--rollout-data-yaml-config` 支持多数据集按固定比例混合并按数据集记录指标；`--use-output-store` 把 R3、indexer、sampling mask 等大数组移入 Mooncake 对象存储；NIXL 成为第二种 P2P 权重传输。
- **工程化**：所有 Miles 自有 CLI 参数改为 typed pydantic 配置，`parse_args()` 返回不可变 `AllConfig`；构建为纯 Python wheel，要求 Python 3.11+；torchtitan 升级至 v0.4.0。
- **扩散模型**：Miles-Diffusion 文档并入本仓库，新增 Krea-2、Qwen-Image 2.1、MiniMax H3 SFT、DMD2 蒸馏、异步训练与 score centering 等能力（代码仍在独立仓库）。
- **升级提示**：`--rollout-max-context-len` / `--eval-max-context-len` 被 `--max-seq-len` 取代；prompt 默认随机打乱；`rollout/raw_reward` 语义变化、MFU 改为按 HF 配置逐层计算；`--lora-type` 默认值按后端解析；LoRA checkpoint 迁移为 adapter 目录；Tinker 客户端必须使用 SDK 0.36.0；torchtitan v0.3.0 checkpoint 不能续训；MLA 模型（DeepSeek、Kimi、GLM-5.x）的 `kv_b_proj` LoRA 受到限制；DSA / GDN kernel 默认值变更；不再支持 Python 3.10。

## 演进主线

| 能力 | 引入版本 | 后续演进 |
| --- | --- | --- |
| 全异步 RL、TITO、R3、P2P RDMA、低精度训练、LoRA | v0.1.0 | 成为后续所有版本的基座 |
| Sampling-support replay（含 top-p mask replay） | v0.1.1 | 后续版本持续完善 |
| Multi-LoRA / Tinker 兼容网关 | v0.1.1 | v0.1.3 原生 raw 模式；v0.2.0 adapter 库、原生 Multi-LoRA、SDK 0.36 |
| Anthropic Messages API、沙箱化 Agent 训练 | v0.1.1 | v0.2.0 增加数据集混合与更多 Agent 配方 |
| torchtitan 训练后端 | v0.1.2 | v0.2.0 升至 v0.4.0，新增 LoRA、HTTP adapter 传输与 TPU 运行路径 |
| Kubernetes 后端、拆分部署、热重启、多策略训练 | v0.1.2（实验性） | v0.2.0 继续演进，仍为实验性 |
| score centering | v0.1.2 | v0.2.0 扩展到扩散模型 |
| 原生 Megatron LoRA（raw 模式） | v0.1.3 | v0.2.0 扩展到 DeepSeek-V4.1、Multi-LoRA 与 `lm_head` |
| 权重更新容错、完全异步 colocate | v0.1.3 | v0.2.0 新增 NIXL P2P 传输并继续完善 |
| TPU、多模态 token space、NIXL、输出存储、typed `AllConfig` | v0.2.0 | — |

概括来看，Miles 的演进可以分成四条主线：

1. **后端与硬件**：从 Megatron-LM + FSDP2 双后端、NVIDIA + AMD 双硬件，扩展到 torchtitan 三后端和 NVIDIA / AMD / TPU 多平台，并在 v0.2.0 用纯 wheel 与类型化配置收拢工程复杂度。
2. **正确性与算法**：从 TITO、R3、Zero-KL，到 sampling-support replay、score centering、top-p mask replay、长度惩罚与有界陈旧度，持续缩小 rollout 与训练之间的数值和策略偏差。
3. **规模与稳定性**：从全异步，到 Kubernetes / 拆分部署 / 热重启 / 多策略，再到权重更新容错、完全异步 colocate 与可恢复 buffer；P2P 传输也从 v0.1.0 的 RDMA 快路径演进为 v0.2.0 的 Mooncake / NIXL 双传输。
4. **LoRA 与训练服务**：从单 LoRA，到 Multi-LoRA v2 + Tinker 网关，再到原生 raw 模式 LoRA、统一 adapter checkpoint 与 HTTP-LoRA 传输，逐步把“一个基座服务多个训练租户”做成正式能力。

## 升级与兼容性提示

跨版本升级时，建议先阅读对应版本的 **Breaking Changes & Upgrade Notes**，并重点关注以下边界：

- **v0.1.0 → v0.1.1**：Multi-LoRA v1 与外部 router 模式移除；会话服务器端口语义变更；版本化镜像仅提供 CUDA 13。
- **v0.1.1 → v0.1.2**：启动脚本改为 `prepare` / `train` / `full-train` 子命令；Agent 函数默认子进程运行；会话回复裁剪；CUDA 12 镜像停止自动构建。
- **v0.1.2 → v0.1.3**：rollout 侧 checkpoint 布局变更，**v0.1.2 运行无法在 v0.1.3 续训**；`DataSource` 与 fully-async buffer API 变更；容错要求 P2P 权重传输。
- **v0.1.3 → v0.2.0**：`--max-seq-len` 取代旧上下文长度参数；LoRA checkpoint 迁移为 adapter 目录；torchtitan v0.3.0 checkpoint 不可续训；Tinker 客户端需 SDK 0.36.0；Python 最低版本升至 3.11；MLA 模型 `kv_b_proj` LoRA 受限。

对于需要长期复现的实验，建议固定到具体 tag，并同时固定 `release-lock.json` 中的 SGLang / Megatron-LM commit 与对应 Docker 镜像（`radixark/miles:vX.Y.Z`）。

## 参考链接

- 上游仓库：<https://github.com/radixark/miles>
- 全部 Release：<https://github.com/radixark/miles/releases>
- 官方文档：<https://miles.radixark.com/docs>
- v0.1 技术报告：<https://arxiv.org/abs/2609.08368>
- Miles-Diffusion：<https://github.com/radixark/miles_diffusion>
- slime：<https://github.com/THUDM/slime>
- 许可证：Apache-2.0（见各版本目录下的 `LICENSE`；v0.1.2 起另有 `NOTICE`）
