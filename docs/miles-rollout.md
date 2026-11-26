# Miles Rollout 源码解析

> 本文以 **v0.2.0** 归档源码为基线，所有 `miles/...`、`train.py` 路径默认相对于 `releases/v0.2.0/`。
> 第 16 节按版本对比 v0.1.0 → v0.2.0 的 Rollout 演进，涉及旧版本时会显式写出 `releases/<tag>/` 前缀。
> 归档源码来自官方 tag，未做修改；本文只做静态阅读，不执行归档中的代码。

## 目录

- [1. Rollout 在 RL 循环中的位置](#1-rollout-在-rl-循环中的位置)
- [2. 进程与组件拓扑](#2-进程与组件拓扑)
- [3. 控制面：InferenceController](#3-控制面inferencecontroller)
- [4. 数据面：RolloutExecutor](#4-数据面rolloutexecutor)
- [5. Rollout 函数 API](#5-rollout-函数-api)
- [6. 一次训练 rollout 的执行链](#6-一次训练-rollout-的执行链)
- [7. 生成函数与请求构造](#7-生成函数与请求构造)
- [8. TITO 会话服务器](#8-tito-会话服务器)
- [9. 数据源](#9-数据源)
- [10. 完全异步 Rollout 与数据缓冲](#10-完全异步-rollout-与数据缓冲)
- [11. 正确性机制](#11-正确性机制)
- [12. 奖励、过滤与数据转换](#12-奖励过滤与数据转换)
- [13. 评估](#13-评估)
- [14. 权重同步、Checkpoint 与容错](#14-权重同步checkpoint-与容错)
- [15. 配置与扩展点](#15-配置与扩展点)
- [16. 各版本演进](#16-各版本演进)
- [17. 关键文件索引](#17-关键文件索引)

---

## 1. Rollout 在 RL 循环中的位置

### 1.1 一条数据生产线

Miles 的一次 RL step 可以抽象成：

```text
DataSource
   │  取 prompt group
   ▼
Rollout（SGLang 生成 + reward）
   │  Sample / group
   ▼
过滤（aborted / 无 reward / 动态采样 / sample filter）
   │
   ▼
convert_samples_to_train_data（advantage、loss mask、dp 切分）
   │
   ▼
Trainer step（Megatron-LM / FSDP2 / torchtitan）
   │
   ▼
update_weights（trainer → SGLang engines）
   │
   └──────────────► 下一个 rollout
```

Rollout 并不只是“调用推理引擎”。在 Miles 里，它同时承担：

- **取数**：从 `DataSource` 拿 prompt，并按 `n_samples_per_prompt` 组成 group；
- **生成**：构造 SGLang `/generate` 请求，或通过 TITO 会话服务器驱动多轮 / Agent 轨迹；
- **奖励**：调用 reward model，或接收环境/agent 自己写入的 reward；
- **过滤**：丢掉中止、缺 reward、被动态采样过滤、被 sample filter 标记的样本；
- **转换**：把变长的 `Sample` 变成 trainer 可直接消费的训练数据，并按 DP 切分；
- **版本管理**：记录每条样本是在哪个权重版本下生成的，控制策略陈旧度；
- **状态管理**：checkpoint / resume、完全异步缓冲、评估与故障恢复。

### 1.2 同步编排：`train.py`

v0.2.0 的同步主循环在 `train.py:112-169`，核心顺序是：

```python
for rollout_id in range(args.start_rollout_id, args.num_rollout):
    await inference_controller.prepare_rollout(rollout_id)      # 引擎就绪、onload
    rollout_data_pack = await rollout_executor.get(rollout_id)  # 生成 + 转训练数据

    await actor_model.train(rollout_id, rollout_data_pack)      # 训练
    await rollout_executor.release(rollout_id)                  # 释放输出存储引用

    await update_weights(..., rollout_id=rollout_id)            # 权重同步
    await eval_dispatcher.dispatch(rollout_id, ...)             # 可选评估
```

关键点：

- `prepare_rollout` 与 `get` 是**分开的 RPC**：前者只做控制面准备，后者才真正生成。这样编排层可以在需要时插入 offload / onload / 权重更新。
- `get` 返回的 `RolloutDataPack` 不直接携带大块训练数据，而是携带 **object store 引用**（`data_ref`）和 `sample_indices`；训练进程从 object store 读取。
- `release` 只负责删除本轮训练已经消费掉的 **output store** 引用（v0.2.0 的 Mooncake 路径）；没有 output store 引用时直接返回。
- `update_weights` 发生在训练之后、下一次 rollout 之前；它会把 trainer 的新权重同步到 SGLang engines，并发布新的 weight version。

### 1.3 异步编排：`train_async.py`

`train_async.py` 的目标是让第 N+1 次 rollout 与第 N 次训练重叠：

```python
async def prepare_and_generate(rollout_id):
    await inference_controller.prepare_rollout(rollout_id)
    return await rollout_executor.get(rollout_id)

rollout_data_next_future = await eager_create_task(prepare_and_generate(args.start_rollout_id))
# 训练当前 step 的同时，下一次 rollout 已经在跑
```

几个关键变量：

- `defer_next_drain = args.fully_async and has_next_rollout and weight_update_due`：在需要更新权重的 step，完全异步模式会**推迟下一次 drain**，避免把旧权重下生成的组带进新版本。
- `--colocate` 只允许与 `--fully-async` 搭配（`train_async.py:25`），因为共享 GPU 时生成与训练必须由完全异步的缓冲机制解耦。
- `--fully-async` 要求使用类式 Rollout API（默认路径），不能与 `MILES_USE_LEGACY_ROLLOUT_V1=1` 同时使用。

---

## 2. 进程与组件拓扑

### 2.1 组件图

```text
┌─────────────────────────────────────────────────────────────────────┐
│ Driver: train.py / train_async.py                                   │
│  - 控制训练 step、保存、评估、权重更新                                │
└───────────────┬───────────────────────────────┬─────────────────────┘
                │ RPC                           │ RPC
                ▼                               ▼
┌───────────────────────────────┐   ┌─────────────────────────────────┐
│ InferenceController            │   │ RolloutExecutor                 │
│ 控制面：引擎生命周期 / 健康检查  │   │ 数据面：DataSource + RolloutFn   │
│ - prepare_rollout / prepare_eval│   │ - get(rollout_id)               │
│ - offload / onload             │   │ - convert / split by dp         │
│ - start/end_update_weights     │   │ - release / save / load         │
│ - cell reconcile / fault tol.  │   │ - set_weight_version            │
└───────────────┬───────────────┘   └───────────────┬─────────────────┘
                │ 管理                               │ HTTP
                ▼                                    ▼
┌───────────────────────────────┐   ┌─────────────────────────────────┐
│ SGLang engine cells            │◄──│ Miles Router / SGLang Router     │
│ 每 cell 一组 engine workers     │   │ 请求负载均衡                     │
└───────────────────────────────┘   └───────────────┬─────────────────┘
                                                     │ HTTP
                                      ┌──────────────▼──────────────┐
                                      │ Session Server（可选，独立进程）│
                                      │ TITO / 多轮 / Agent 会话       │
                                      └─────────────────────────────┘
```

### 2.2 两种部署抽象

v0.2.0 通过 `BaseWorkerHandle` / `BackendCapability` 把“组件跑在哪里”抽象出来：

- Ray 后端：`create_components` / `create_rollout_components`（`miles/ray/placement_group.py`）创建 Ray actor；
- Kubernetes 后端：`--cluster-backend kubernetes` 时不需要外层 Ray 集群，组件由 K8s 原生编排。

`RolloutExecutor` 本身是 `DedicatedRpcLoopWorker`（`miles/ray/rollout/rollout_executor.py:74`）：所有 rollout 相关协程共享一个专用事件循环，避免与 Ray 的默认事件循环互相阻塞。完全异步的 producer 也跑在这个循环上。

### 2.3 引擎 cell

`ServerCell`（`miles/ray/rollout/server_cell.py`）是引擎管理的最小协调单元：

- 一个 cell 包含若干 SGLang worker（rank）；
- 每个 cell 有独立的 `cell_id`，健康状态、权重版本、故障替换都以 cell 为单位；
- `InferenceController` 通过 worker provider 观察 cell（`watch_cells` / `reconcile`），支持 Ray 自建引擎，也支持外部已启动的 SGLang 引擎（`external_engine_provider.py`）。

---

## 3. 控制面：InferenceController

文件：`miles/ray/rollout/inference_controller.py`。

### 3.1 职责

`InferenceController` 只做控制面，不生成样本：

| 方法 | 作用 |
| --- | --- |
| `init` | 构建 router / session server 地址、worker provider、初始 cell、健康监控 |
| `prepare_rollout(rollout_id)` | 每次训练 rollout 前调用：确保引擎 ready、完成 onload、等待 cell 可用 |
| `prepare_eval` | 评估前准备（专用 eval fleet 或共享引擎） |
| `offload` / `onload` | 按 tag 卸载/加载 KV cache、weights、CUDA graph；支持 colocate |
| `start_update_weights(model_id)` | 冻结/暂停引擎，返回 `UpdatableEngines`（含 cell 快照与校验信息） |
| `end_update_weights(...)` | 校验权重写入结果、收集 checksum、返回失败 cell |
| `abort_update_weights()` | trainer 更新失败时中止，避免引擎停在半更新状态 |
| `check_weights` | 校验引擎权重版本与 trainer 一致 |
| `stop_cell_between_weight_updates` / `inject_fault_between_weight_updates` | 故障隔离与 CI 故障注入 |

### 3.2 ContextLock 与并发纪律

`InferenceController` 用 `ContextLock` 约束“哪些操作可以在什么上下文里并发”：

- 生成请求与权重更新不能同时发生；
- offload / onload 与 prepare 之间需要互斥；
- 评估与训练 rollout 共享引擎时需要显式 pause。

这层锁是后续完全异步、colocate、权重更新容错能够安全工作的基础。

### 3.3 外部引擎

两种接入方式：

- `--rollout-external-engine-addrs host:port ...`：Miles 不启动引擎，从每个引擎的 `/server_info` 发现拓扑；
- `--custom-inference-engine-provider-path`：自定义 `BaseWorkerProvider`，返回 cell 信息。

对应实现是 `StaticInferenceEngineWorkerProvider`（`miles/ray/rollout/external_engine_provider.py`）：启动时发现引擎、校验模型/并行配置、构造 cell，之后和内置引擎走同一套 controller 逻辑。

### 3.4 评估 fleet

评估可以复用训练引擎，也可以使用独立 eval fleet（`--eval-num-gpus`）：

- `InferenceControllerEvalFleet` 管理 eval 专用引擎；
- `RolloutExecutor` 通过 `set_eval_fleet_info` 拿到 fleet 信息；
- 评估时 `RolloutFnEvalInput.generate_state` 可以指向专用 fleet 的 `GenerateState`，从而不干扰训练 rollout。

---

## 4. 数据面：RolloutExecutor

文件：`miles/ray/rollout/rollout_executor.py`。

### 4.1 初始化

`RolloutExecutor.init()`（`rollout_executor.py:112`）做这些事：

1. 解析 router / session server 地址；
2. 初始化 tracking、object store、HTTP client；
3. 若 `--use-output-store`，连接 Mooncake output store，保存 `delete_many`；
4. 通过 `--data-source-path` 加载 `DataSource`；
5. 通过 `--rollout-function-path` / `--eval-function-path` 加载类式 RolloutFn；
   - 两条路径相同时复用同一个实例，保证 `FullyAsyncRolloutFn` 这类有状态函数不会被构造两次；
   - `MILES_USE_LEGACY_ROLLOUT_V1=1` 时回退到旧函数签名；
6. 加载自定义 reward post-process / convert 函数；
7. 创建 `MetricChecker`。

### 4.2 `get`：一次 rollout 的完整数据面

`get(rollout_id, trainer_model_id)`（`rollout_executor.py:222`）的主流程：

```python
replay = self._output_snapshotter.get(trainer_model_id, rollout_id)
if replay is None:
    data, metadata = await self._generate_rollout_data(...)
    self._output_snapshotter.capture(...)
else:
    data, metadata = replay.data, replay.metadata

self._unreleased_store_refs[rollout_id] = store_refs_of(data) | take_dropped_refs()
train_data = convert_samples_to_train_data(...)

if args.delay_split_train_data_by_dp:
    data_ref = object_store.put(train_data)
else:
    shards = split_train_data_by_dp(train_data, train_parallel_config)
    data_ref = [object_store.put(shard) for shard in shards]

return RolloutDataPack(sample_indices=..., data_ref=data_ref)
```

几个容易忽略的点：

- **output snapshot**：生成结果在进入训练前被捕获。如果 trainer 在“已生成、未训练”之间重启，下一次 `get` 会 replay 这份快照，而不是用新权重重新生成。`_MAX_RETAINED_OUTPUTS = 3`，只保留最近几份未训练输出。
- **store refs**：使用 output store 时，样本里的 R3 / indexer / sampling mask 大数组是 Mooncake 引用；`release(rollout_id)` 在训练消费完后删除这些对象。删除失败只记录日志（对象泄漏，不影响正确性）。
- **convert 时机**：`convert_samples_to_train_data` 在 executor 内完成，因此 trainer 拿到的是已经按 DP rank 切好的 shard，或者一个尚未切分的 object store 引用（`--delay-split-train-data-by-dp`）。

### 4.3 `_generate_rollout_data` 与 `_get_rollout_data`

`_generate_rollout_data`（`rollout_executor.py:276`）负责：

- 刷新 `inference_runtime_mut_state`（引擎数量、GPU 数）；
- 检查 weight version 已发布（`assert_weight_version_is_published`）；
- 调用 `_get_rollout_data`，计时并记录 metrics；
- debug dump / log。

`_get_rollout_data`（`rollout_executor.py:396`）构造输入：

```python
RolloutFnTrainInput(
    rollout_id=rollout_id,
    weight_version=self._weight_versions_of_model_id.get(trainer_model_id),
    trainer_model_id=trainer_model_id,
)
```

然后调用 `self.generate_rollout(input)`。如果启用了 legacy v1，则改用 `call_rollout_fn(...)`。

### 4.4 `release`、`save`、`load`

- `release(rollout_id)`（`:260`）：删除该 rollout 的 output store 引用；
- `save(rollout_id, completed_rollout_ids)`（`:433`）：把 rollout 侧状态写入 `<save>/rollout/<rollout_id>/`，包括 data source、generate_rollout、eval_generate_rollout、executor 的 output snapshot；
- `load(rollout_id, ..., load=...)`（`:451`）：恢复上述状态；如果 data source 配置与 checkpoint 不匹配，会抛 `RolloutDataResumeMismatch`，由 `--rollout-data-reset-on-load` 决定是否从头开始。

### 4.5 权重版本

`set_weight_version(weight_version, trainer_model_id)`（`:523`）由 `update_weights` 在权重同步完成后调用。之后新的 `RolloutFnTrainInput` 会带上这个版本号；样本生成时也会记录 `all_weight_version_spans`，用于陈旧度统计。

---

## 5. Rollout 函数 API

文件：`miles/rollout/base_types.py`。

### 5.1 输入输出契约

v0.2.0 的类式 API：

```python
@dataclass(frozen=True)
class RolloutFnConstructorInput:
    args: Namespace
    data_source: DataSource

@dataclass(frozen=True)
class RolloutFnTrainInput(RolloutFnBaseInput):
    rollout_id: int
    weight_version: int | None = None    # 首次权重更新前为 None
    trainer_model_id: str | None = None  # 多策略训练时标识策略

@dataclass(frozen=True)
class RolloutFnEvalInput(RolloutFnBaseInput):
    rollout_id: int
    invocation_id: str                    # 每次 eval 调用的唯一 id
    generate_state: GenerateState | None  # 专用 eval fleet 的生成状态
    weight_version: str | None
    hf_dir: str | None                    # checkpoint eval 的 HF 快照目录
```

输出：

```python
@dataclass
class RolloutFnTrainOutput:
    samples: list[list[Sample]]           # group 列表
    metrics: dict[str, Any] = None

@dataclass
class RolloutFnEvalOutput:
    data: dict[str, dict[str, Any]]       # 按数据集组织
    metrics: dict[str, Any] = None
```

### 5.2 `BaseRolloutFn` 生命周期

`BaseRolloutFn`（`base_types.py:95`）定义：

| 方法 | 作用 |
| --- | --- |
| `__call__(input)` | 训练或评估入口，根据 `input.evaluation` 分派 |
| `save(directory)` | 保存自身状态（完全异步的 buffer / 在途任务等） |
| `load(directory)` | 恢复状态 |
| `drop_saved_in_flight(directory)` | data source 被 reset 时，丢弃保存的在途 prompt group |
| `dispose()` | 释放后台任务 |

`RolloutExecutor.init` 会用 `load_rollout_function` 构造实例；如果 train / eval 使用同一个 `--rollout-function-path`，两边的实例是同一个对象。

### 5.3 默认实现：`InferenceRolloutFn`

`miles/rollout/inference_rollout/inference_rollout_common.py:234` 的 `InferenceRolloutFn`：

- `__call__` 根据 `input.evaluation` 分派到 `_call_train` / `_call_eval`；
- `_call_train` 调用 `generate_rollout_async`；
- `_call_eval` 调用 `run_eval_datasets`；
- 内部持有 `GenerateState`（tokenizer、processor、并发信号量、sampling params）。

### 5.4 生成函数契约

`GenerateFnInput` / `GenerateFnOutput`（`base_types.py:118`）：

```python
@dataclass(frozen=True)
class GenerateFnInput:
    state: GenerateState
    sample: Sample
    sampling_params: dict[str, Any]
    evaluation: bool

@dataclass(frozen=True)
class GenerateFnOutput:
    samples: Sample | list[Sample]   # 一次生成可以产出多个样本
```

内置生成函数在 `miles/rollout/generate_hub/`：

- `single_turn.generate`：默认，单轮请求；
- `multi_turn.generate`：多轮工具调用；
- `agentic_tool_call.generate`：把一条样本交给自定义 agent 函数，通过 TITO 会话服务器收集完整轨迹。

选择方式：

- 全局：`--custom-generate-function-path`（替换默认 `generate`）；
- 按样本/数据集：`--rollout-data-yaml-config` 里的 `generate_function_path`，或样本自身的 `generate_function_path`。

### 5.5 兼容层：legacy v1

`MILES_USE_LEGACY_ROLLOUT_V1=1` 时，rollout 函数仍然是旧式：

```python
def generate_rollout(args, rollout_id, data_source, evaluation=False):
    ...
```

`call_rollout_fn`（`base_types.py:136`）负责调用并把返回值适配成 `RolloutFnTrainOutput` / `RolloutFnEvalOutput`。这条路径主要用于兼容旧的自定义 rollout 脚本；完全异步与 v0.2.0 的新特性都要求类式 API。

---

## 6. 一次训练 rollout 的执行链

以默认 `InferenceRolloutFn` 为例，核心函数是 `generate_rollout_async`（`miles/rollout/inference_rollout/inference_rollout_train.py:95`）。

### 6.1 准备

```python
state = GenerateState(args)
# tokenizer / processor
# semaphore = args.sglang_server_concurrency * engine_count
# sampling_params = compute_sampling_params(...)
```

`GenerateState` 是长生命周期状态：tokenizer、processor、信号量、sampling params、当前是否 aborted。`reset()` 在每次 rollout 结束时清理 aborted 标记，避免影响下一次 rollout / eval。

### 6.2 主循环

```python
target_data_size = args.rollout_batch_size          # 以 group 为单位
scheduler = make_submission_scheduler(args, default="group")
pendings = set()
data = []

while len(data) < target_data_size:
    while scheduler.has_capacity(pending_groups=len(pendings),
                                 group_budget=target_data_size - len(data)):
        samples = data_source(args.over_sampling_batch_size)
        stamp_kv_cache_namespace(samples, namespace=kv_cache_namespace)
        scheduler.on_submit(samples)
        pendings.update(submit_generate_tasks(state, samples, scheduler.sample_done_callback))

    done, pendings = await scheduler.wait_for_progress(pendings)

    for task in done:
        group = task.result()
        all_data.append(group)
        if not apply_preput_filters(args, dynamic_filter, group).keep:
            continue
        if len(data) < target_data_size:
            data.append(group)
```

几个关键设计：

- **batch 单位是 group**：`rollout_batch_size` 是 prompt group 数；每组包含 `n_samples_per_prompt` 条样本。
- **oversampling**：`--over-sampling-batch-size` 控制每次从 data source 取多少组。被动态过滤掉的组不会占用最终 batch 名额，因此会继续提交新组。
- **提交粒度**：
  - `--rollout-submission-granularity group`（同步默认）：整个 group 完成才释放提交槽位；
  - `--rollout-submission-granularity sample`（完全异步默认）：每条样本完成就释放自己的槽位，新的 group 可以在旧 group 尚未全部完成时提交。
- **`stamp_kv_cache_namespace`**：开启 `--namespaced-radix-cache` 时，给样本打上本次 rollout 调用的 KV cache namespace，防止旧权重下算出的前缀 KV 被新样本复用。

### 6.3 组级生成

`submit_generate_tasks` 把一个 group 包成一个 `asyncio.Task`，调用 `generate_and_rm_group`（`inference_rollout_common.py:160`）：

- 给每个样本分配 `routing_key`（router 按 key 路由时会话粘性）；
- 设置 deterministic sampling seed（如果引擎启用 deterministic inference）；
- 并发调用 `generate_and_rm`；
- 通过 `sample_done_callback` 通知 scheduler 某条样本已完成（sample 粒度提交时使用）。

### 6.4 单样本生成与奖励

`generate_and_rm`（`inference_rollout_common.py:71`）：

1. 如果是 partial rollout 且 `--mask-offpolicy-in-partial-rollout`，先把旧 response 的 loss mask 置 0；
2. 如果样本已完成/已截断，直接返回；
3. 根据样本的 `max_response_len` 调整 `max_new_tokens`；
4. 获取 `GenerateState.generate_fn_semaphore`（并发上限 = `sglang_server_concurrency × engine_count`）；
5. 调用生成函数（`single_turn` / `multi_turn` / `agentic`）；
6. 如果 `group_rm=False`，调用 reward model：
   - 单样本：`async_rm(args, sample)`；
   - 多样本（multi-agent）：`batched_async_rm(...)`；
   - 如果 agent / 环境已经写入 reward，则跳过。

奖励既可以是标量，也可以是 dict（配合 `--reward-key` 取值）；`group_rm=True` 时由组级 reward model 统一计算。

### 6.5 收尾

生成到 `rollout_batch_size` 个有效 group 后：

1. `abort(state, pendings, rollout_id)`：
   - 向所有 worker 发送 `/abort_request`；
   - 调用 agent abort hook，让 agent 环境停止运行；
   - `--partial-rollout` 时收集已生成的部分样本，否则丢弃并记录 dropped；
2. 按 `sample.index` 排序，保证数据顺序稳定；
3. `state.reset()`；
4. 应用 `--rollout-sample-filter-path`（例如 `mask_truncated_zero_reward`）；
5. 可选 `--rollout-all-samples-process-path`：对包括被过滤样本在内的全量数据做处理；
6. `recompute_samples_rollout_logprobs_via_prefill`：对需要重算的样本，用一次 prefill 重新计算 rollout logprobs；
7. 返回 `RolloutFnTrainOutput(samples=data, metrics=...)`。

### 6.6 从 `Sample` 到训练数据

回到 `RolloutExecutor.get`：

- `_get_rollout_data` 先做 `postprocess_rollout_data`（自定义 postprocess、长度截断等）；
- `assert_samples_weight_version_sane` 校验样本的 weight version 合法；
- 可选 `RolloutDataInjectionUtil`（debug / CI 注入数据）；
- `convert_samples_to_train_data`：
  - 按 group 做 advantage normalization；
  - 生成 `loss_mask`、`sample_indices`；
  - 应用自定义 `--custom-convert-samples-to-train-data-path`；
- `split_train_data_by_dp` 按训练并行配置切分，或延迟到 trainer 侧切分。

---

## 7. 生成函数与请求构造

### 7.1 `single_turn`

`miles/rollout/generate_hub/single_turn.py`：

- `compute_prompt_ids_from_sample`：从样本的 prompt / messages 构造 token ids；
- 如果样本已有部分 response（partial rollout resume），直接把已有 tokens 作为输入，并扣减 `max_new_tokens`；
- `multimodal_token_space_processing` 开启时，媒体 token 由 SGLang 在 token 空间展开；
- `compute_request_payload` 构造请求；
- POST 到 `http://<router>/generate`；
- `update_sample_from_response` 回填 tokens、logprobs、reward 相关字段。

### 7.2 `multi_turn`

`miles/rollout/generate_hub/multi_turn.py`：

- 循环最多 `--generate-max-turns` 轮；
- 每轮生成后解析 tool call，执行工具，把 observation 作为下一轮输入；
- 维护 `loss_mask`：只对模型生成的 token 计算 loss，工具返回的 token 置 0；
- 合并多轮的 sampling mask / top-k logprobs；
- 到最后仍没有结束，则标记 `TRUNCATED`。

### 7.3 `agentic_tool_call`

`miles/rollout/generate_hub/agentic_tool_call.py`：

1. 通过 `OpenAIEndpointTracer` / session client 创建 TITO 会话；
2. 调用 `--custom-agent-function-path` 指定的 agent 函数；
3. agent 通过 session server 的 OpenAI / Anthropic 接口与模型交互；
4. 会话结束后从 session server `/samples` 收集 worker 组装的训练样本；
5. 应用 agent metadata / session metadata。

Agent 函数的执行模式（`miles/rollout/agentic/agent_function.py`）：

- `subproc`：每次 agent 调用起一个子进程。一个 episode 的阻塞、崩溃、状态泄漏不会影响其它 episode；子进程通过 pickle 管道回传结果，支持 `report_to_caller` 中途上报；
- `inline`：直接在 rollout 事件循环上执行，开销小但会阻塞其它协程。

v0.1.2 起默认 `subproc`；子进程使用 `PR_SET_PDEATHSIG` 保证 rollout 进程退出时 agent 不会残留。

### 7.4 请求载荷

`compute_request_payload`（`miles/rollout/generate_utils/generate_endpoint_utils.py:121`）构造的字段：

| 字段 | 含义 |
| --- | --- |
| `input_ids` | prompt token ids（多模态时是未展开的占位符） |
| `sampling_params` | temperature / top_p / top_k / max_new_tokens 等 |
| `return_logprob` | 始终为 `True`，训练需要 rollout logprobs |
| `return_routed_experts` | `--use-rollout-routing-replay` 时请求 MoE 路由（R3） |
| `return_indexer_topk` | `--use-rollout-indexer-replay` 时请求 indexer top-k |
| `return_sampling_mask` | sampling-support replay 时请求实际采样支撑集 |
| `lora_path` | LoRA rollout 时指定 adapter |
| `image_data` / 媒体字段 | 多模态输入；token-space 模式下改传媒体引用 |
| `mm_token_expansion_start_len` | token-space 下从哪个位置开始展开媒体占位符 |
| `extra_key` | KV cache namespace（namespaced radix cache） |
| output store 字段 | `--use-output-store` 时让 SGLang 把大数组写入 Mooncake，只返回 ref |

### 7.5 响应回填

`update_sample_from_response`：

- 首次请求时初始化 `sample.tokens`；
- `adopt_expanded_prompt_ids`：如果请求要求返回 `prompt_token_ids`，用 SGLang 展开后的 prompt 替换本地 tokens（多模态 token 空间）；
- 追加 response tokens、`rollout_log_probs`；
- 解码 routed experts / indexer top-k / sampling mask；
- 追加 top-k logprobs（`append_rollout_topk_logprobs`）；
- `truncate_to_max_seq_len`：把超过 `--max-seq-len` 的输出裁掉并标记 `TRUNCATED`。

---

## 8. TITO 会话服务器

### 8.1 为什么需要 TITO

多轮 / Agent 场景下，如果每一轮都把文本重新 tokenize，会出现：

- chat template 差异导致 token 序列与模型实际生成的不一致；
- 工具返回、思考 token、特殊 token 在重新编码时被改写；
- 训练侧看到的 token 与 rollout 侧实际采样的 token 不同，logprob 对不上。

TITO（Token-In-Token-Out）的做法是：**会话服务器保存模型实际生成/看到的 token 序列**，下一轮直接以 token 形式送回引擎，训练样本也直接由这些 token 组装，不再经过文本重编码。

### 8.2 独立进程

`miles/rollout/session/server.py`：

- 独立的 FastAPI 进程（进程标题 `miles-session-server`），拥有自己的事件循环；
- `SessionServer.do_proxy` 用 aiohttp 把请求转发给推理 router（SGLang Router 或 Miles Router），由 router 做负载均衡；
- 因为不依赖 Miles Router，也可以配合 SGLang Rust Router、Kubernetes 原生部署；
- 支持 OpenAI 兼容接口与 Anthropic Messages API（`anthropic_adapter.py`），Claude Code 等 harness 可以直接接入。

### 8.3 v1 与 v2 会话模型

| 维度 | v1 | v2 |
| --- | --- | --- |
| 核心类 | `SessionCore` / `LinearTrajectory` / `SessionRegistry` | `SessionCoreV2` / `SessionStateV2` |
| 轨迹结构 | 线性 turn 序列 | 树状 / 多分支轨迹（`tree_trajectory.py`） |
| 分支 | 单分支 | 支持 attach point、回滚、多分支采样 |
| 样本 | `samples/codec.py` 编解码 | `samples/merge.py` 合并多分支样本 |
| 指标 | 基础 metrics | `v2/metrics.py` 更细的会话指标 |

v2 的 `SessionStateV2` 记录每个 turn 的 attach point 与提交状态，支持“同一 prompt 多个分支”以及 picker 策略（如丢弃重复 prompt 的重试）。

### 8.4 样本收集与输出存储

- 会话结束时，session server 通过 `/samples` 返回组装好的训练样本：tokens、loss_mask、logprobs、reward、metadata；
- v0.2.0 的 `session/store_refs.py` 管理 output store 引用：
  - 每个响应可能携带一个 `output_store_ref`；
  - 成功的 `/samples` 把引用交给客户端；
  - 会话 DELETE 时删除未被交出的引用（中止的 turn、回滚的分支、被 picker 丢弃的叶子）；
  - 如果响应声称使用 output store 却没有 ref，抛出 `StorePlanError`（非可重试错误）。

---

## 9. 数据源

文件：`miles/rollout/data_source.py`。

### 9.1 抽象

```python
class DataSource(abc.ABC):
    def get_samples(self, num_samples: int) -> list[list[Sample]]: ...
    def save(self, directory: Path) -> None: ...
    def load(self, directory: Path) -> None: ...
    def get_buffer_length(self) -> int | None: ...
```

`get_samples(num_samples)` 返回的是 **group 列表**，每个 group 的长度是 `n_samples_per_prompt`。

### 9.2 `RolloutDataSource`

最常用的实现：

- 读取全局数据集（`--rollout-global-dataset`）；
- 维护 `epoch_id`、`sample_group_index`、`sample_index`、`sample_offset`；
- 每个 epoch 按 `--rollout-seed` 打乱（`--no-rollout-shuffle` 可关闭）；
- 支持 `--rollout-max-prompt-len` 过滤超长 prompt；
- `save/load` 保存 epoch / offset / 样本 id，保证 resume 后从正确位置继续。

### 9.3 `MixedRolloutDataSource`

v0.2.0 新增，对应 `--rollout-data-yaml-config`：

```yaml
datasets:
  - name: math
    path: /data/math.jsonl
    weight: 2
    max_seq_len: 8192
    agent_function_path: ...
  - name: code
    path: /data/code.jsonl
    weight: 1
```

实现要点（`data_source.py:278`）：

- **按权重平滑轮转**：`split_by_weight` 把每个 batch 的取数名额按权重分配，避免大权重数据集长期霸占；
- **每个数据集独立 offset / epoch / shuffle**；
- **sha256 pinning**：记录每个数据集文件的内容哈希，resume 时校验；不匹配则抛 `RolloutDataResumeMismatch`；
- **`_make_group`**：给样本打上 `dataset_name`、`agent_config`、`agent_function_path`、`max_seq_len`、`max_response_len`、`generate_function_path`、`reward_spec` 等 per-dataset 元数据。

这样一条 run 可以同时训练多个异构数据集/Agent 环境，并按数据集记录指标。

### 9.4 恢复语义

- `--rollout-data-reset-on-load`：当 checkpoint 的 data source 配置与当前配置不兼容时，允许从头开始；
- 如果 data source 被 reset，`RolloutExecutor.load` 会调用 `generate_rollout.drop_saved_in_flight(...)`，丢弃保存的在途 prompt group；
- 不 reset 时，data source 从保存的 offset 继续，output snapshot 也可以 replay 未训练 batch。

---

## 10. 完全异步 Rollout 与数据缓冲

### 10.1 动机

同步 rollout 的每步耗时近似：

```text
rollout_time + train_time + weight_sync_time
```

完全异步让一个**常驻 producer** 持续生成，训练 step 只从 buffer 中 drain 已完成的 group：

```text
max(rollout_time, train_time) + 少量同步开销
```

### 10.2 生产者：`FullyAsyncRolloutFn`

文件：`miles/rollout/fully_async_rollout.py`。

- 首次 train 调用时懒启动 `_worker_loop`；
- producer 始终保持最多 `rollout_batch_size` 个 group 在途（或由 `--async-max-concurrent-samples` 控制）；
- `_submit_one_group`：优先从 retry buffer 取，否则从 data source 取；打 KV cache namespace；
- `_generate_group`：调用 `generate_and_rm_group`；
- 完成的 group 放进 `DataBuffer`（`put`），由消费者 `_drain` 取走；
- `_handle_unused`：根据 `--async-unused-samples-handler` 选择 `retry`（回收 prompt 重新生成）或 `drop`。

### 10.3 消费者与 buffer

`DataBuffer` 契约（`miles/rollout/fully_async_data_buffer.py:95`）：

```python
async def put(input: DataBufferInput) -> None
async def get(*, num_groups: int, current_version: int | None = None, ...) -> list[DataBufferInput]
def get_metrics(trainer_model_id) -> dict[str, float]
def state_dict() / load_state_dict(state)
def admits(*, in_flight: int) -> bool
def drop_saved_state(state, *, reason: str)
```

`DefaultDataBuffer`（`:141`）：

- FIFO 队列；
- `put` 时做 preput 过滤：aborted、缺 reward、动态采样过滤失败；
- `get` 时做**陈旧度过滤**：如果 group 的最老生成 token 版本与当前引擎版本差距超过 `--max-weight-staleness`，拒绝并回收；
- 容量：`floor(--async-data-buffer-capacity-factor × rollout_batch_size)` 个 group，默认 factor = 2.0；满时 producer 阻塞；
- `--async-submit-within-staleness`：在 producer 提交阶段就用 `admits()` 控制，使在途 + 缓冲的 group 数不超过 N × batch；
- `--async-unused-samples-handler drop|retry`：训练不用的 group 是丢弃还是回收 prompt；
- 指标：`staleness`、`buffer_size`、drop 原因等。

`DefaultMultiDataBuffer`（`:378`）：多策略训练时按 `trainer_model_id` 为每个策略维护一个内层 buffer；可用 `--custom-async-data-buffer-path-per-model MODEL_ID=PATH` 为某个策略指定自定义 buffer。

### 10.4 提交粒度与并发

- 完全异步默认 `--rollout-submission-granularity sample`：每条样本完成即释放槽位，避免长尾样本拖住整个 group；
- `--async-max-concurrent-samples`：把生成并发与训练 batch 解耦；
- `--max-weight-staleness`：控制 off-policy 程度；
- `--fully-async --colocate`：生成与训练共享 GPU，显存通过 offload / KV cache 释放协调；v0.1.3 起支持。

### 10.5 checkpoint / resume

`FullyAsyncRolloutFn.save/load` 保存 `_FullyAsyncRolloutState`：

```python
@dataclass
class _FullyAsyncRolloutState:
    retry_buffer: deque[list[Sample]]
    running: list[_RunningTask]
    output: DataBuffer
```

- `retry_buffer`：等待重试的 prompt group；
- `running`：在途任务及其 prompt group；
- `output`：已完成但未被训练的 group。

恢复时 producer 和在途任务会重新建立，避免 checkpoint 后丢失已生成的样本。`drop_saved_in_flight` 在 data source reset 时只丢弃在途组，保留已完成 buffer。

### 10.6 与 `train_async.py` 的配合

- `--fully-async` 选择 `FullyAsyncRolloutFn`；
- `train_async.py` 在权重更新 step 使用 `defer_next_drain`，避免下一次 drain 取到旧版本 group；
- 如果 `--colocate`，权重更新与 KV cache 的 onload/offload 需要严格顺序，编排层会先释放训练显存再加载 rollout 权重；
- `--fully-async` 不支持 `--partial-rollout`、`--recompute-logprobs-via-prefill`、`--rollout-all-samples-process-path`、`--pause-generation-mode abort` 等会与常驻 producer 冲突的选项；`--fully-async --colocate` 若卸载 rollout 权重，还要求 `--update-weights-interval 1`（启动校验见 `miles/utils/arguments.py:138-157`）。

---

## 11. 正确性机制

Rollout 与训练之间的数值偏差是 RL 后训练的核心问题。Miles 在 rollout 侧提供了多种“重放”机制。

### 11.1 Weight version 与陈旧度

- 每次权重同步后，`InferenceController` 让引擎发布新版本，`RolloutExecutor.set_weight_version` 记录；
- 样本在生成时记录 `all_weight_version_spans`（可能跨多个版本，例如长样本在权重更新期间生成）；
- 完全异步的 `DefaultDataBuffer` 用最老版本计算 staleness；
- `--max-weight-staleness` 限制允许的最大差距；
- `--namespaced-radix-cache` 防止旧权重下的 prefix KV 被新样本复用。

### 11.2 R3：Rollout Routing Replay

MoE 模型在 rollout 与训练时的路由可能不同，导致同一 token 在两侧走不同的 expert。R3 的做法：

- rollout 请求设置 `return_routed_experts=True`（`--use-rollout-routing-replay`）；
- SGLang 返回每个 token 的 routed experts；
- 样本保存 `rollout_routed_experts`；
- trainer 在 forward 时强制使用 rollout 的路由。

样本级字段与训练侧 replay 逻辑在 `miles/utils/types.py` 与各训练后端中；rollout 侧只负责请求与回填。

### 11.3 Indexer replay

DeepSeek 等模型的 DSA（稀疏注意力）indexer 也会在两侧产生不同的 top-k。`--use-rollout-indexer-replay`：

- 请求 `return_indexer_topk=True`；
- 样本保存 indexer top-k；
- trainer 重放 rollout 的 indexer 决策。

### 11.4 Sampling-support replay

当 rollout 使用 `top_p < 1` 或 `top_k > 0` 时，实际采样分布被截断。如果训练侧仍在全词表上计算 log-prob，会高估低概率 token 的概率。Sampling-support replay 的做法：

- rollout 请求 `return_sampling_mask=True`；
- SGLang 返回每个生成 token 的**实际候选支撑集**与对应 logprob；
- 样本保存 `RolloutSamplingMask`；
- trainer 只在支撑集上计算 log-prob / importance ratio。

约束（`miles/rollout/generate_utils/sampling_mask.py`）：

- top_p 必须在 `(0, 1]`；
- replay 需要显式设置 `top_p`、`top_k`、`temperature`，且 temperature 必须等于训练 temperature；
- `frequency_penalty`、`presence_penalty`、`repetition_penalty`、`logit_bias`、`custom_logit_processor` 不被支持，因为 trainer 无法复现这些 logit 变换；
- 工具/环境插入的非采样 token 用 singleton support 标记（`append_forced_sampling_tokens`）。

### 11.5 Top-k logprobs

`--rollout-top-logprobs-num K` 记录每个生成 token 的 top-K 候选：

- 用于 score centering、重要性采样修正、诊断 rollout 与训练的分布差异；
- `--rollout-sampling-logprobs-mode support`：SGLang support 模式返回实际行为分布的候选；
- 非 support 模式使用 `top_logprobs_num` / base64 flat 数组，避免逐候选 detokenize，且 `-inf` 不会在 JSON 中变成 `null`；
- 样本字段：`rollout_topk_token_ids`、`rollout_topk_log_probs`，形状 `[response_len, K]`，不可用槽位填 `-1` / `-inf`；
- `validate_rollout_topk_logprobs_sample` 在 `--ci-test` 下做全量校验（候选与 sampling support 一致、采样 token 的概率与 `rollout_log_probs` 一致）。

### 11.6 Prefill logprob 重算

某些场景（例如 partial rollout、权重更新后继续生成）需要重新计算 rollout logprobs。`recompute_samples_rollout_logprobs_via_prefill` 用一次 prefill 请求重算，而不是重新采样。`--recompute-logprobs-via-prefill` 控制是否启用；完全异步不支持该选项。

### 11.7 Output store

v0.2.0 的 `--use-output-store` 把 R3 / indexer / sampling mask 等大数组从 HTTP 响应中移出：

- SGLang 把数组写入 Mooncake，返回 `output_store_ref`；
- rollout 侧只传引用，不读大数组；
- trainer 读取对象并把 `StorePlan` materialize 成训练列；
- 样本合并、过滤、截断时只编辑 `StorePlan`，不搬运数组；
- 训练消费完后由 `RolloutExecutor.release` 删除对象。

这既降低 rollout 进程的内存/网络压力，也避免大数组在 session server、router、driver 之间反复序列化。

---

## 12. 奖励、过滤与数据转换

### 12.1 Reward

- 逐样本 reward：`async_rm(args, sample)`；
- 组级 reward：`args.group_rm=True` 时由组级 reward model 计算；
- 多样本 / multi-agent：`batched_async_rm`；
- agent / 环境可以自己写 `sample.reward`，此时不再调用 reward model；
- reward 可以是标量或 dict；dict 用 `--reward-key` 取值；
- `--custom-reward-post-process-path` 在 reward 计算后、转换前做自定义处理。

### 12.2 Preput filters

`apply_preput_filters`（`miles/rollout/filter_hub/common_filters.py`）在样本进入最终 batch 前依次检查：

1. **aborted**：样本被中止；
2. **missing reward**：没有 reward；
3. **dynamic sampling filter**：例如 `check_reward_nonzero_std`，要求 group 内 reward 有非零方差（DAPO 风格）。

任一失败，group 被丢弃并记录原因；主循环继续提交新 group。

### 12.3 Sample filters

`--rollout-sample-filter-path` 指向 `fn(args, data)`，直接修改样本：

- `mask_truncated_zero_reward`：把“被长度/时间截断且 reward ≤ 0”的样本标记为 `remove_sample=True`。这类轨迹没有完成任务，0 reward 不能说明动作好坏；正奖励的截断样本保留。
- `remove_sample=True` 的样本不进入 loss，但仍参与 group 的 reward normalization。

### 12.4 Length penalty

v0.2.0 新增 `miles/rollout/reward_shaping.py`，用于在 advantage normalization 前对正确轨迹做长度惩罚：

```python
# 仅对 reward == 1.0 的 rollout
reference = max(min_tokens, min(length of correct rollouts))
multiplier = max(1.0, length / reference) ** (-coeff)
shaped_reward = reward * multiplier
```

参数：

- `--length-penalty-coeff`：非负有限值；0 表示关闭；
- `--length-penalty-min-tokens`：reference 长度的下限；
- 不能与 `--custom-reward-post-process-path` 或 `--custom-convert-samples-to-train-data-path` 同时使用，避免两次整形叠加；
- 支持同一 prompt 下多个 rollout segment（多轮/多分支）合并计算长度；
- 只改 reward，不改样本原始 reward，便于日志对比。

### 12.5 转换为训练数据

`convert_samples_to_train_data`（`miles/ray/rollout/train_data_conversion.py`）：

- 按 group 计算 advantage（GRPO 等）；
- 生成 `loss_mask`：只对模型生成的 token 计算 loss；
- 处理 partial rollout 的 off-policy mask；
- 生成 `sample_indices`，用于把训练结果映射回样本；
- 应用 `--custom-convert-samples-to-train-data-path`；
- 按 DP 切分或放入 object store。

---

## 13. 评估

### 13.1 评估入口

`EvalDispatcher`（`miles/ray/rollout/eval_dispatch.py`）负责：

- 按 `--eval-interval` 触发；
- 选择评估目标：共享 rollout 引擎、专用 eval fleet、或 checkpoint snapshot；
- 调用 `RolloutExecutor.eval(...)`；
- 汇总指标、处理 `EvalSkip`。

`RolloutExecutor.eval` 根据 `weight_version` / `--eval-uses-snapshots` 决定走普通 eval 还是 `_eval_checkpoint`。

### 13.2 评估输入

`RolloutFnEvalInput` 携带：

- `rollout_id`；
- `invocation_id`：每次 eval 调用唯一，用于 KV cache namespace；
- `generate_state`：专用 eval fleet 的 tokenizer / processor / 并发状态；
- `weight_version`：要评估的权重版本；
- `hf_dir`：checkpoint eval 的 HF 快照目录。

`RolloutFnEvalOutput.data` 是按数据集组织的指标，`metrics` 是全局指标。

### 13.3 专用 eval fleet

`--eval-num-gpus > 0` 时，Miles 会启动独立 eval 引擎：

- 训练 rollout 与 eval 不共享引擎，避免 eval 的长尾请求拖慢训练；
- `InferenceController.prepare_eval` 负责 onload / ready；
- eval fleet 的 `GenerateState` 通过 `set_eval_fleet_info` 传给 `RolloutExecutor`。

### 13.4 Checkpoint eval

`CheckpointEvalFn`（`miles/rollout/checkpoint_eval.py`）：

- 用于“评估某个已保存的 checkpoint”，而不是当前引擎权重；
- 需要 `train_async.py` + snapshot source；
- 可以从 HF 快照启动独立引擎，或把 snapshot 加载到 eval fleet；
- 不支持时抛 `EvalSkip`，由 dispatcher 记录跳过原因。

---

## 14. 权重同步、Checkpoint 与容错

### 14.1 Weight update handshake

`miles/ray/placement_group.py:382` 的 `update_weights`：

```python
info = await inference_controller.start_update_weights(model_id=trainer_model_id)
try:
    output = await actor_model.update_weights(info=info, rollout_id=rollout_id)
except BaseException:
    await inference_controller.abort_update_weights()
    raise

await inference_controller.end_update_weights(
    snapshot_cell_id_to_hashes=info.snapshot_cell_id_to_hashes,
    failed_cell_ids=output.failed_cell_ids,
    checksum_model_id=trainer_model_id,
    collect_checksums=...,
)

if output.weight_version is not None:
    await rollout_executor.set_weight_version(output.weight_version,
                                              trainer_model_id=trainer_model_id)
```

三个阶段：

1. **start**：controller 暂停生成、返回可更新的 cell 与权重快照哈希；
2. **update**：trainer 把权重写到引擎（broadcast / p2p / disk-delta / http-lora 等）；
3. **end**：controller 校验写入结果、收集 checksum、恢复生成；失败的 cell 被隔离，其余 cell 继续服务。

### 14.2 传输方式

`--update-weight-transfer-mode`：

| 模式 | 说明 |
| --- | --- |
| `broadcast` | 默认，trainer 向所有引擎广播完整权重 |
| `broadcast_packed` | 打包后广播，减少小 tensor 的通信开销 |
| `p2p` | 点对点传输；v0.2.0 支持 Mooncake / NIXL 两种 transport |
| `disk-delta` | 只发布相对上一版的差异字节，引擎从本地 checkpoint 增量应用 |
| `ici` | TPU 共享 slice 的设备到设备传输 |
| `http-lora` | LoRA-only，通过 HTTP 把 adapter 交给引擎 |

`--pause-generation-mode` 控制权重更新时在途请求的处理：

- `abort`：直接中止；
- `retract`：放回等待队列，更新后重算 KV；
- `in_place`：冻结请求，保留 KV cache 继续。

### 14.3 容错

- `--ft-components weight_update`：权重更新失败的 rollout cell 被隔离，其余 cell 继续升级；
- 训练 cell 卡死/死亡可被放弃，不阻塞全局；
- `InferenceController` 的健康监控与 cell reconcile 会自动替换故障 cell；
- 权重写入支持逐写 checksum（`--log-inference-engine-weight-checksums`）；
- `abort_update_weights` 防止 trainer 失败后引擎停在半更新状态。

### 14.4 Rollout checkpoint 布局

`<save>/rollout/<rollout_id>/`：

```text
rollout/<rollout_id>/
├── data_source/            # DataSource 状态（epoch / offset / sample ids）
├── generate_rollout/       # 训练 rollout fn 状态（如 fully-async buffer）
├── eval_generate_rollout/  # eval rollout fn 状态（如果与 train 不同）
└── executor/               # output snapshot（未训练 batch）
```

`RolloutExecutor.save` 使用 `atomic_save_folder`，先写临时目录再原子替换；`load` 按相同布局恢复。

### 14.5 Output snapshotter 的语义

`miles/ray/rollout/output_snapshotter.py`：

- `capture`：`get` 生成完、训练前捕获 batch；
- `get`：replay 未训练 batch（标记 `REPLAYED`，只能 replay 一次）；
- `discard_completed`：训练完成的 rollout 不再保留；
- `save/load`：随 rollout checkpoint 一起持久化；
- 最多保留 3 份未训练输出，避免内存无限增长；
- 如果 data source 被 reset，`drop_loaded` 丢弃这些 snapshot 并删除对应的 output store 引用。

这保证了“trainer 重启后不会用新权重重新生成旧 step 的数据”，对异步训练和 checkpoint resume 都是必要的正确性保障。

---

## 15. 配置与扩展点

### 15.1 Typed `AllConfig`

v0.2.0 起，Miles 自有 CLI 参数改为 pydantic 类型化配置：

- `miles/utils/args/runtime.py:286` 定义 `AllConfig`；
- `miles/utils/args/configs/` 按领域拆分：`rollout.py`、`data.py`、`algo.py`、`reward_model.py`、`train.py` 等；
- `parse_args()` 返回不可变配置；
- 每个字段带 `Arg(help=...)`，参数语义直接写在代码里。

例如 `RolloutRelatedConfig`（`miles/utils/args/configs/rollout.py`）包含：

- `rollout_function_path`、`eval_function_path`；
- `rollout_temperature` / `rollout_top_p` / `rollout_top_k` / `rollout_max_response_len`；
- `fully_async`、`update_weights_interval`、`pause_generation_mode`；
- `use_output_store`、`object_store_backend`；
- `update_weight_transfer_mode`、`p2p_transport`；
- `rollout_external_engine_addrs`、`custom_inference_engine_provider_path`。

`RolloutRelatedRolloutOnlyConfig` 还包含：

- `max_seq_len`、`rollout_max_prompt_len`；
- `rollout_shuffle`、`rollout_seed`；
- `over_sampling_batch_size`、`dynamic_sampling_filter_path`；
- `rollout_submission_granularity`；
- `max_weight_staleness`、`async_data_buffer_capacity_factor`、`async_max_concurrent_samples`、`custom_async_data_buffer_path`；
- `partial_rollout`、`mask_offpolicy_in_partial_rollout`。

### 15.2 关键参数速查

| 参数 | 作用 |
| --- | --- |
| `--rollout-function-path` | Rollout 函数类路径（默认 `InferenceRolloutFn`） |
| `--eval-function-path` | 评估函数路径；与训练路径相同则复用实例 |
| `--data-source-path` | DataSource 类路径 |
| `--custom-generate-function-path` | 替换默认 `generate` 函数 |
| `--rollout-batch-size` | 每步训练的 prompt group 数 |
| `--n-samples-per-prompt` | 每个 prompt 生成的样本数 |
| `--over-sampling-batch-size` | 每次从 data source 取多少组 |
| `--rollout-submission-granularity` | `group` / `sample` 提交粒度 |
| `--dynamic-sampling-filter-path` | 动态采样过滤器 |
| `--rollout-sample-filter-path` | 样本级过滤器（可设 `remove_sample`） |
| `--partial-rollout` | 收集被中止的部分样本 |
| `--max-weight-staleness` | 完全异步允许的最大权重版本差距 |
| `--fully-async` | 启用完全异步 producer/consumer |
| `--use-rollout-routing-replay` | R3 |
| `--use-rollout-indexer-replay` | indexer top-k replay |
| `--rollout-top-logprobs-num` | 记录 top-K 候选 logprobs |
| `--use-output-store` | 大数组写入 Mooncake，返回引用 |
| `--rollout-data-yaml-config` | 多数据集混合配置 |
| `--length-penalty-coeff` | 长度惩罚系数 |
| `--max-seq-len` | 单条轨迹总 token 上限（v0.2.0 起替代旧上下文长度参数） |

### 15.3 扩展点

| 扩展点 | 参数 | 接口 |
| --- | --- | --- |
| 自定义 Rollout 函数 | `--rollout-function-path` | `BaseRolloutFn.__call__` |
| 自定义生成函数 | `--custom-generate-function-path` | `GenerateFnInput -> GenerateFnOutput` |
| 自定义 DataSource | `--data-source-path` | `DataSource` |
| 自定义异步 buffer | `--custom-async-data-buffer-path` | `DataBuffer` |
| 自定义 filter | `--dynamic-sampling-filter-path` / `--rollout-sample-filter-path` | `fn(args, group)` |
| 自定义 reward | `--custom-reward-post-process-path` | `fn(args, samples)` |
| 自定义训练数据转换 | `--custom-convert-samples-to-train-data-path` | `fn(args, samples, metadata)` |
| 自定义 Agent 函数 | `--custom-agent-function-path` | `async fn(args, sample, ...)` |
| 自定义引擎 provider | `--custom-inference-engine-provider-path` | `BaseWorkerProvider` |
| 自定义 eval | `--eval-function-path` | `BaseRolloutFn` 或 `CheckpointEvalFn` |

所有路径通过 `miles/utils/function_registry.py` 的 `load_function` 解析，支持 `module.path:function` 形式。

---

## 16. 各版本演进

### 16.0 总览

| 版本 | 发布日期 | Rollout 主题 | 关键变化 |
| --- | --- | --- | --- |
| v0.1.0 | 2026-08-18 | 全栈 RL 基座 | `RolloutManager` 单体 actor；类式 API 已是默认；TITO v1/v2；R3；Multi-LoRA v1；完全异步与 submission scheduler 已有雏形 |
| v0.1.1 | 2026-09-26 | 前沿模型 / Agent 生态 | 控制面/数据面拆分：`InferenceController` + `RolloutExecutor`；sampling-support replay；Agent 包与 Anthropic adapter；移除 Multi-LoRA v1 rollout |
| v0.1.2 | 2026-10-09 | 多后端 / 平台化 | Agent 子进程模式；top-k logprobs；外部 SGLang 引擎；多策略 `trainer_model_id`；eval fleet 信息接入 executor |
| v0.1.3 | 2026-10-23 | 稳定性 / 容错 / 完全异步 | sample filters；output snapshotter；完全异步 checkpoint / colocate；权重更新容错；DataSource API 变更；namespaced radix cache |
| v0.2.0 | 2026-11-12 | 多硬件 / 多模态 / 平台成熟 | 多数据集混合 DataSource；长度惩罚；output store；多模态 token space；typed `AllConfig`；NIXL；serving registry |

`miles/rollout` 的代码量变化（Python 行数）：

| 版本 | 行数 |
| --- | --- |
| v0.1.0 | 8,324 |
| v0.1.1 | 9,093 |
| v0.1.2 | 9,836 |
| v0.1.3 | 10,426 |
| v0.2.0 | 11,653 |

### 16.1 v0.1.0：单体 `RolloutManager`

**架构**

- `miles/ray/rollout/rollout_manager.py` 是一个 `@ray.remote` actor，同时承担：
  - DataSource 与 RolloutFn 的构造；
  - SGLang server 的启动（`start_rollout_servers`）；
  - session server 启动；
  - health monitor；
  - eval fleet；
  - `generate` / `eval` / `save` / `load` / `offload` / `onload`；
  - `get_updatable_engines_and_lock` / `recover_updatable_engines` / `set_weight_version`。
- 配套文件：`rollout_server.py`、`server_group.py`、`server_engine.py`、`addr_allocator.py`。
- `train.py` 直接调用 `rollout_manager.generate.remote(rollout_id)`，然后 `actor_model.update_weights(rollout_id=...)`。

**Rollout 能力**

- 类式 API 已经存在：`InferenceRolloutFn`、`BaseRolloutFn`、`FullyAsyncRolloutFn`、`fully_async_data_buffer.py`、`submission_scheduler.py`；
- legacy v1 仍可通过 `MILES_USE_LEGACY_ROLLOUT_V1=1` 使用；
- TITO session server v1 / v2 均在；
- R3（`--use-rollout-routing-replay`）已在；
- partial rollout、dynamic sampling filter、debug data injection、checkpoint eval 已有；
- `multi_lora/` 包提供 Multi-LoRA rollout 与 data source。

**尚未具备**

- sampling-support replay（v0.1.1）；
- top-k logprobs（v0.1.2）；
- common preput filters / sample filters（v0.1.1 / v0.1.3）；
- 外部引擎、output snapshotter、完全异步 checkpoint / colocate（v0.1.2 / v0.1.3）；
- output store、多数据集混合、长度惩罚（v0.2.0）。

### 16.2 v0.1.1：控制面 / 数据面拆分

**架构重构**

- 删除 `rollout_manager.py`、`server_group.py`、`server_engine.py`、`addr_allocator.py`；
- 新增 `inference_controller.py`（控制面）与 `rollout_executor.py`（数据面）；
- `server_cell.py` 扩展为 cell 生命周期管理；新增 `cell_state.py`；
- `train.py` 改为：

```python
inference_controller, rollout_executor, num_rollout_per_epoch = await create_rollout_components(args)
...
await inference_controller.prepare_rollout(rollout_id)
rollout_data_pack = await rollout_executor.get.remote(rollout_id)
...
await update_weights(actor_model, rollout_executor, rollout_id=rollout_id)
```

这次拆分让控制面与数据面可以独立扩展：控制面管引擎/权重/故障，数据面管数据源/生成/转换。

**正确性**

- 新增 `miles/rollout/generate_utils/sampling_mask.py`：
  - `should_return_sampling_mask` / `validate_sampling_support_request`；
  - `RolloutSamplingMask`；
  - `append_sampling_metadata`；
  - 工具插入 token 的 singleton support。
- 新增 `miles/rollout/filter_hub/common_filters.py`：aborted / missing reward / dynamic filter 的统一 preput 顺序。

**Agent 生态**

- 新增 `miles/rollout/agentic/`（`credentials.py`、`session.py`）；
- 新增 `session/anthropic_adapter.py`、`session/config.py`、`session/request_args.py`、`session/v2/metrics.py`；
- 新增 picker 策略 `session/v2/picker_hub/drop_same_prompt_retries.py` 与 `drop_rolled_back_leaves.py`，替代 v0.1.0 的 `drop_retries.py`。

**移除**

- 删除 `miles/rollout/multi_lora/async_rollout.py`、`multi_lora/data_source.py`；
- Multi-LoRA v1 及其参数被移除，后续由 Multi-LoRA v2 / Tinker 网关接管。

**差异规模**

- `miles/rollout`：48 个文件，+1,949 / −1,180；
- `miles/ray/rollout`：15 个文件，+1,077 / −1,168。

### 16.3 v0.1.2：Agent 子进程、top-k logprobs、外部引擎

**Agent**

- 新增 `miles/rollout/agentic/agent_function.py`：
  - `subproc` / `inline` 两种模式；
  - 子进程 pickle 管道、`report_to_caller`、取消宽限、`PR_SET_PDEATHSIG`；
  - 默认 `subproc`。
- 新增 `agentic/rollout_limits.py`：Agent 调用的资源/长度限制。
- `generate_hub/agentic_tool_call.py` 改用 `call_agent_function`，并新增 `--custom-agent-function-mode`。

**正确性**

- 新增 `generate_utils/rollout_topk_logprobs.py`：
  - `configure_rollout_topk_logprobs_request`；
  - support / selected 两种模式；
  - base64 flat 数组解码；
  - `rollout_topk_token_ids` / `rollout_topk_log_probs`。
- 这是后续 score centering、重要性采样修正的基础。

**平台化**

- 新增 `miles/ray/rollout/external_engine_provider.py`：接入外部已启动的 SGLang 引擎；
- 新增 `engine_env_reporter.py`：上报引擎环境信息；
- `RolloutExecutor` 增加 `trainer_model_id` 参数与 `set_eval_fleet_info` / `get_init_state`，为多策略训练与 eval fleet 做准备；
- `rollout_executor.py` 从 283 行增长到 338 行，`inference_controller.py` 从 336 行增长到 416 行。

**差异规模**

- `miles/rollout`：24 个文件，+815 / −72；
- `miles/ray/rollout`：13 个文件，+899 / −227。

### 16.4 v0.1.3：sample filters、output snapshotter、完全异步 checkpoint

**过滤**

- 新增 `miles/rollout/filter_hub/sample_filters.py`：
  - `mask_truncated_zero_reward`；
  - `CUT_OFF_EXIT_STATUSES`（`TimeLimitExceeded` / `SequenceLengthLimitExceeded`）；
  - `remove_sample` 语义：不进入 loss，但仍参与 group reward normalization。

**Checkpoint / resume**

- 新增 `miles/ray/rollout/output_snapshotter.py`：
  - `capture` / `get` / `discard_completed` / `save` / `load`；
  - `CAPTURED` / `LOADED` / `REPLAYED` 三阶段；
  - 最多保留 3 份未训练输出；
  - `--ci-inject-missing-prefetched-batch-bug` 用于 CI 验证。
- `RolloutExecutor.save/load` 改为按 `rollout_id` 分目录：
  - `data_source/`、`generate_rollout/`、`eval_generate_rollout/`、`executor/`；
  - `compute_rollout_checkpoint_dir`。
- `FullyAsyncRolloutFn` 增加 `save/load/drop_saved_in_flight`：
  - `_FullyAsyncRolloutState(retry_buffer, running, output)`；
  - `--fully-async --colocate` 支持；
  - `--async-submit-within-staleness`、`admits()`；
  - `--async-max-concurrent-samples`；
  - `--async-data-buffer-capacity-factor` 等缓冲参数完善。
- `DataSource` API 变更：
  - `save/load` 从 `rollout_id` 改为 `directory: Path`；
  - 基类移除 `add_samples`，仅 legacy buffer 保留；
  - v0.1.2 的 rollout checkpoint 无法在 v0.1.3 续训。

**权重更新容错**

- `InferenceController` 增加 `stop_cell_between_weight_updates`、`inject_fault_between_weight_updates`、cell 状态查询；
- `--ft-components weight_update` 隔离权重更新失败的 cell；
- 逐写 checksum、失败 cell 上报。

**性能 / 正确性**

- 新增 `--namespaced-radix-cache`（v0.1.3 起完善）：按 rollout 调用隔离 prefix KV；
- session server 线程/协程优化（发布说明约 46 → 75 turns/s）；
- `sglang_rollout.py` 增加 sampling mask 相关校验与安全断言（全零 routed experts 等）。

**差异规模**

- `miles/rollout`：30 个文件，+832 / −242；
- `miles/ray/rollout`：12 个文件，+705 / −195。

### 16.5 v0.2.0：多数据集、长度惩罚、output store、多模态

**数据源**

- `data_source.py` 从 206 行增长到 493 行；
- 新增 `MixedRolloutDataSource`：
  - `--rollout-data-yaml-config`；
  - 按权重平滑轮转；
  - 每个数据集独立 offset / epoch / shuffle；
  - sha256 pinning；
  - `RolloutDataResumeMismatch` + `--rollout-data-reset-on-load`；
  - `_make_group` 注入 per-dataset 元数据。

**奖励整形**

- 新增 `miles/rollout/reward_shaping.py`：
  - `--length-penalty-coeff` / `--length-penalty-min-tokens`；
  - 只惩罚正确轨迹；
  - 支持多 segment rollout；
  - 与 custom reward post-process / convert 互斥。

**Output store**

- 新增 `miles/rollout/session/store_refs.py`；
- `--use-output-store` + `--object-store-backend mooncake`：
  - SGLang 把 R3 / indexer / sampling mask 写入 Mooncake；
  - 响应只返回 `output_store_ref`；
  - trainer 侧 materialize `StorePlan`；
  - `RolloutExecutor.release` 删除对象；
  - 新增 `miles/utils/output_store/`（`plan.py`、`dropped.py`、`trainer_columns.py` 等）。

**多模态**

- `--multimodal-token-space-processing`：
  - rollout 与 trainer 使用同一个 media processor；
  - SGLang 在 token 空间展开媒体占位符；
  - `mm_token_expansion_start_len` / `return_prompt_token_ids`；
  - `truncate_to_max_seq_len` 处理展开后的超长；
- `--max-seq-len` 取代旧的 `--rollout-max-context-len` / `--eval-max-context-len`。

**异步 / 多策略**

- `DefaultMultiDataBuffer` 支持按 `trainer_model_id` 分策略缓冲；
- `--custom-async-data-buffer-path-per-model MODEL_ID=PATH`；
- `--async-submit-within-staleness`、`--async-max-concurrent-samples` 完善；
- `RolloutExecutor` 按 `trainer_model_id` 维护 weight version / train parallel config / output snapshot。

**平台化**

- Typed `AllConfig`（`miles/utils/args/`）；
- `serving_registry.py`：把引擎注册/注销抽象为 `ServingRegistry`；
- NIXL 成为第二种 P2P transport（`--p2p-transport nixl`）；
- `http-lora` 权重传输；
- Python 3.11+、纯 Python wheel。

**差异规模**

- `miles/rollout`：41 个文件，+1,560 / −333；
- `miles/ray/rollout`：13 个文件，+470 / −159。

### 16.6 演进主线

1. **从单体到分层**：v0.1.0 的 `RolloutManager` 同时管引擎、数据源、生成、转换；v0.1.1 拆成 `InferenceController`（控制面）与 `RolloutExecutor`（数据面），后续所有平台化能力都建立在这个边界上。
2. **从同步到完全异步**：v0.1.0 已有 `FullyAsyncRolloutFn` 与 buffer 雏形；v0.1.3 补齐 checkpoint / colocate / staleness 控制；v0.2.0 支持多策略 buffer 与更强的并发控制。
3. **正确性机制持续加码**：TITO → R3 → sampling-support replay（v0.1.1）→ top-k logprobs（v0.1.2）→ namespaced radix cache / output snapshotter（v0.1.3）→ output store（v0.2.0）。
4. **平台与生态**：Agent 内联 → 子进程（v0.1.2）→ Anthropic / OpenAI 适配与多数据集混合（v0.1.1 / v0.2.0）；外部引擎（v0.1.2）→ typed config / K8s / TPU / NIXL（v0.2.0）。

---

## 17. 关键文件索引

| 文件 | 作用 |
| --- | --- |
| `train.py` | 同步训练编排 |
| `train_async.py` | 异步 / 完全异步训练编排 |
| `miles/ray/placement_group.py` | 组件创建、`update_weights` handshake |
| `miles/ray/rollout/inference_controller.py` | 引擎控制面 |
| `miles/ray/rollout/rollout_executor.py` | Rollout 数据面 |
| `miles/ray/rollout/server_cell.py` | 引擎 cell 管理 |
| `miles/ray/rollout/output_snapshotter.py` | 未训练 batch 的捕获 / replay |
| `miles/ray/rollout/eval_dispatch.py` | 评估调度 |
| `miles/ray/rollout/external_engine_provider.py` | 外部 SGLang 引擎接入 |
| `miles/rollout/base_types.py` | Rollout 函数契约 |
| `miles/rollout/inference_rollout/inference_rollout_common.py` | `GenerateState`、`generate_and_rm`、`InferenceRolloutFn` |
| `miles/rollout/inference_rollout/inference_rollout_train.py` | `generate_rollout_async` |
| `miles/rollout/inference_rollout/inference_rollout_eval.py` | 评估生成 |
| `miles/rollout/fully_async_rollout.py` | 完全异步 producer / consumer |
| `miles/rollout/fully_async_data_buffer.py` | 异步数据缓冲 |
| `miles/rollout/submission_scheduler.py` | group / sample 提交调度 |
| `miles/rollout/data_source.py` | DataSource 与混合数据源 |
| `miles/rollout/generate_hub/` | single_turn / multi_turn / agentic 生成函数 |
| `miles/rollout/generate_utils/generate_endpoint_utils.py` | 请求构造与响应回填 |
| `miles/rollout/generate_utils/sampling_mask.py` | sampling-support replay |
| `miles/rollout/generate_utils/rollout_topk_logprobs.py` | top-k logprobs |
| `miles/rollout/generate_utils/prefill_logprobs.py` | prefill logprob 重算 |
| `miles/rollout/filter_hub/common_filters.py` | preput filters |
| `miles/rollout/filter_hub/sample_filters.py` | sample filters |
| `miles/rollout/reward_shaping.py` | 长度惩罚 |
| `miles/rollout/session/` | TITO 会话服务器 v1/v2 |
| `miles/rollout/agentic/agent_function.py` | Agent 子进程 / inline 执行 |
| `miles/rollout/checkpoint_eval.py` | checkpoint 评估 |
| `miles/utils/args/` | typed `AllConfig` |
| `miles/utils/output_store/` | Mooncake output store 与 `StorePlan` |
