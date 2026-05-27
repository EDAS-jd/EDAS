# EDPO (branch_adv): Easy-R1 → verl 迁移说明

> 本文档记录把 `/mnt/public/users/liuwenpu/Easy-R1-branchadv` 上跑通的 **GRPO / DAPO + branch_adv (EDPO)** 完整训练流程移植到 `/mnt/public/users/liuwenpu/EDPO/verl` 框架的全部改动、原因、用法与未覆盖项。

- 两个仓库都做 DAPO，但底层框架不同：
  - **Easy-R1-branchadv** = EasyR1 派生的轻量单 yaml 框架，jinja + format_prompt 在 dataset 加载时注入。
  - **EDPO/verl** = 上游 verl，用 Hydra config + `tokenizer.apply_chat_template`，reward 走 RewardManager 工厂。
- 迁移的关键四件事：①核心算法 `branch_adv (EDPO)`、②动态 max_tokens / 长度日志、③reward + LLM judge + jinja 1:1 对齐、④启动脚本（含 SwanLab）打平 CLI。
- 算法本体的语义、三分支公式、零和性证明见 `Easy-R1-branchadv/branch_adv_implementation.md`，本文档**不重复推导**，只讲迁移。

---

## 0. 一句话总结

把 Easy-R1 上的 EDPO 训练命令照搬来用：

```bash
cd /mnt/public/users/liuwenpu/EDPO/verl

bash examples/branch_adv/run_8b_non_thinking_04_30_20_wo_format_17k.sh
```

或者用完整 CLI（与原 Easy-R1 入口 1:1 对齐）：

```bash
cd /mnt/public/users/liuwenpu/EDPO/verl

bash examples/branch_adv/run_dapo_branch_adv.sh \
  --model_path /mnt/public/users/liuwenpu/model/Qwen/Qwen3-8B \
  --train_data /mnt/public/users/liuwenpu/data/DAPO-Math-17k-Processed/train.formatted.parquet \
  --val_data   /mnt/public/users/liuwenpu/data/val_merge/val_merged.formatted.parquet \
  --task_name  8b_non_thinking_04_30_20_wo_format_17k \
  --output_path /mnt/public/users/liuwenpu/EDPO/verl/ckpt \
  --gpus 8 \
  --max_prompt_length 2048 \
  --max_response_length 8192 \
  --swanlab_project new_math-2k-4k \
  --swanlab_mode cloud
```

⚠️ 注意 `*.formatted.parquet` 是 Easy-R1 `*.parquet` 用 `examples/branch_adv/preprocess_easyR1_parquet.py` 预渲染过 jinja 之后的版本（见 §4）。

---

## 1. 改动总览（按层）

### 1.1 核心算法层（`verl/trainer/...`）

| 文件 | 改动 | 关键行 |
|---|---|---|
| `verl/trainer/config/algorithm.py` | `AlgoConfig` 上加 9 个 `branch_adv_*` 字段，Easy-R1 一一对应。 | L671-686 |
| `verl/trainer/config/ppo_trainer.yaml` | `algorithm:` 节加同名键 + Hydra 默认值，便于命令行 override。 | L112-142 |
| `verl/trainer/ppo/core_algos.py` | 新增 `_apply_branch_advantage_adjustment(...)`（A 跳过 / B 模式坍缩统一惩罚 / C 健康发散按相对惊奇度调整 + κ-clip + 日志）；`compute_grpo_outcome_advantage` 在标准化后、`unsqueeze` 前调用，in-place 修改 `scores`；`mathruler.grader.grade_answer` 用作 canonical 去重，缺包回落到字符串相等。 | L40-46, L278-460, L465-560 |
| `verl/trainer/ppo/ray_trainer.py` | 新增 `_compute_branch_adv_warmup_scale`（延迟期 / 爬坡期 / 稳定期，与 Easy-R1 等价）；`compute_advantage` 新增 `branch_adv_kwargs` 参数；`fit()` 在 GRPO 算 advantage 前从 `batch.non_tensor_batch[accuracy / answer_pred / format]` 取值、按 warmup scale 缩放 α/β、把日志路径拼成相对 `trainer.default_local_dir` 的绝对路径，调用 `compute_advantage(..., branch_adv_kwargs=...)`，并把返回的统计量写入 `metrics["branch_adv/*"]`。 | L136-160, L163, L209-211, L1600-1664 |

**关键约定**：reward function 的返回 dict 必须包含 `accuracy`、`answer_pred`（branch_adv 必读）和可选的 `format`（≥0.5 才参与，缺省视为通过）。verl 的 reward 流程会自动把这三个 key 写进 `non_tensor_batch`，trainer 直接读。

| Easy-R1 字段 | verl 字段 | 默认值 | 说明 |
|---|---|---|---|
| `algorithm.branch_adv_enabled` | `algorithm.branch_adv_enabled` | `False` | 总开关 |
| `algorithm.branch_adv_alpha` | `algorithm.branch_adv_alpha` | `0.2` | 分支 C 探索奖励系数 |
| `algorithm.branch_adv_beta` | `algorithm.branch_adv_beta` | `0.2` | 分支 B 坍缩惩罚系数 |
| `algorithm.branch_adv_kappa` | `algorithm.branch_adv_kappa` | `2.0` | κ 熔断系数 |
| `algorithm.branch_adv_log_path` | `algorithm.branch_adv_log_path` | `null` | 调整明细日志相对路径（相对 `trainer.default_local_dir`） |
| `algorithm.branch_adv_warmup_start` | `algorithm.branch_adv_warmup_start` | `0` | 延迟期终点（步数） |
| `algorithm.branch_adv_warmup_steps` | `algorithm.branch_adv_warmup_steps` | `0` | 爬坡步数 |
| —（默认 hard-coded） | `algorithm.branch_adv_acc_key` | `accuracy` | reward dict 中的准确率字段名 |
| —（默认 hard-coded） | `algorithm.branch_adv_answer_key` | `answer_pred` | reward dict 中的预测答案字段名 |
| —（默认 hard-coded） | `algorithm.branch_adv_format_key` | `format` | reward dict 中的格式分字段名 |

### 1.2 动态 max_tokens / 长度日志层（rollout）

> ⚠️ **重要前提**：Easy-R1 的 `ENABLE_DYNAMIC_MAX_TOKENS=True` 是在 `verl/workers/rollout/vllm_rollout_spmd.py:230` 改的（SPMD/sync 路径）。而 **上游 verl 在 PR #4411 已经把 SPMD 模式完全废弃**：
>
> - verl 没有 `vllm_rollout_spmd.py` 这个文件；
> - verl 的同步壳子 `verl/workers/rollout/vllm_rollout/vllm_rollout.py::ServerAdapter.generate_sequences()` 直接抛 `NotImplementedError`（见该文件第 201-220 行），明确告诉调用方走 async server；
> - `verl/workers/rollout/base.py:84` 的 `_ROLLOUT_REGISTRY` 里 `("vllm", "async") → ServerAdapter`，**没有** `("vllm", "sync")` 的注册项；
> - `verl/trainer/config/rollout/rollout.yaml` 第 8 行 `mode: async` 是默认值，本次两个启动脚本也都 **没有** `rollout.mode=...` 的 override。
>
> 所以 verl 上 vLLM 真正活跃的代码路径就是 `vllm_async_server.py`，**这正是本次打补丁的位置**。SPMD 那条路径在 verl 里已经不存在，不需要再改。

| 文件 | 改动 |
|---|---|
| `verl/workers/config/rollout.py` | `RolloutConfig` 新增 `enable_dynamic_max_tokens: bool = False`、`length_log_interval: int = 0`、`length_log_path: Optional[str] = None`（L280-286）。|
| `verl/trainer/config/rollout/rollout.yaml` | 同步声明上面三个键（L354 / L358 / L361），让 Hydra CLI 可以 override。|
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | `generate(...)` 的 max_tokens 兜底分支（L481-493）改写：`enable_dynamic_max_tokens=True` 时返回 `max_possible_tokens = max_model_len - len(prompt_ids)`（去掉 `response_length` 上界，仅受 `max_model_len` 限制）；False 时维持 verl 原行为 `min(response_length, prompt_length + response_length - len(prompt_ids))`。|

#### Easy-R1 ↔ verl async 语义对照

两边公式实质完全等价（用同一个 `max_model_len` 上限）：

| Easy-R1 SPMD (`vllm_rollout_spmd.py:230`) | verl async (`vllm_async_server.py:486`) |
|---|---|
| `total_capacity = self.config.max_model_len` | `max_possible_tokens = self.config.max_model_len - len(prompt_ids)` |
| `dynamic_max = total_capacity - p_len` | `max_tokens = max_possible_tokens` |
| `if dynamic_max <= 0: continue`（跳过该 prompt） | `if max_possible_tokens < 0: raise ValueError`（更严格，直接拒绝） |
| `sp.max_tokens = dynamic_max`，per-prompt 一个独立 `SamplingParams` | 每条请求各自走 async `generate`，per-request 独立 `SamplingParams`，天然等效 |
| `enable_dynamic_max_tokens=False` 时 `max_tokens = response_length`（全局） | `enable_dynamic_max_tokens=False` 时 `max_tokens = min(response_length, max_model_len - len(prompt_ids))` |

**净效果完全一致**：开启后 prompt 越短，留给 response 的空间越大，可以一路撑到 `max_model_len`，绕开 `response_length` 的硬上限；关闭时退回 verl 原版 `response_length` 上界。

#### 验证你跑的就是这条路径

启动后第一段 ray 日志里会有：

```
[ServerAdapter] launching vLLM async server (mode=async) ...
```

如果你看见 `NotImplementedError: ServerAdapter does not support synchronous generate_sequences()`，说明有人在 CLI 里强行 `actor_rollout_ref.rollout.mode=sync` —— 把这个 override 去掉即可。

> 备注：length_log 的**实时写文件**功能（min/mean/max 摘要）目前只暴露了配置位（`length_log_interval` / `length_log_path`），没有在 `vllm_async_server.generate` 里回填写盘逻辑（vLLM 自带 stats 已能覆盖大部分场景）。如需 Easy-R1 风格逐 step 长度日志，把 Easy-R1 `vllm_rollout_spmd.py::_append_length_log` 抠出来作为纯函数 import 即可。

### 1.3 Reward / Jinja / LLM judge 层（`examples/branch_adv/`）

| 路径 | 内容 | 与 Easy-R1 的关系 |
|---|---|---|
| `examples/branch_adv/reward_function/math_no_format.py` | 提供两个入口：①`compute_score(...)` per-sample（NaiveRewardManager 用，无 LLM judge，作 fallback / debug）；②`compute_score_batch(...)` batch（`BatchRewardManager` 用，**1:1 复刻 Easy-R1 `math_no_format.compute_score`** 的流程：rule (`mathruler.grade_answer`) → sympy → **LLM judge fan-out**）。返回 `{score, overall, accuracy, length_penalty, answer_pred, format}`，`score = accuracy + overlong_penalty_factor * length_penalty`。同时根据 `extra_info["split"] == "val"` 自动跳过 sympy + LLM judge + length_penalty，对齐 Easy-R1 的 `worker.val_reward`。 | 行为 1:1，多了 split 路由 |
| `examples/branch_adv/reward_function/llm_judge.py` | **从 Easy-R1 verbatim 移植**（仅调整 `PROMPT_DIR` 相对路径）。HTTP 客户端 / 端口轮询 / `ThreadPoolExecutor` / debug log 全部保留；默认 `host=10.119.97.103, ports=9000-9007, model=/mnt/public/users/zhuyongfu/model/openai/gpt-oss-20b, endpoint=/v1/chat/completions, timeout=30s, max_retries=2, max_workers=512, max_tokens=1024`，**与 Easy-R1 一致**；env 或 `--llm_judge_*` CLI 都可覆盖。 | verbatim |
| `examples/branch_adv/llm_judge_prompt/math.jinja` | **从 Easy-R1 `examples/llm_judge_prompt/math.jinja` verbatim 拷贝**。 | verbatim |
| `examples/branch_adv/llm_judge_prompt/sqa.jinja` | 同上，`verify_type=llava` 自动归一为 `sqa`，与 Easy-R1 一致。 | verbatim |
| `examples/branch_adv/reward_function/val_wo_format.py` | 验证集打分：`extract_boxed_content + grade_answer`，**不调 judge**。Easy-R1 val reward 本来就不调 judge，行为一致。 | 1:1 |
| `examples/branch_adv/format_prompt/math_wo_format.jinja` | 与 Easy-R1 同名模板原样保留。verl 不消费 `data.format_prompt`，由 `preprocess_easyR1_parquet.py` 在数据预处理阶段渲染（见 §4）。 | verbatim（用法不同） |
| `examples/branch_adv/format_prompt/val_wo_format.jinja` | 同上，验证用。 | verbatim |
| `examples/branch_adv/preprocess_easyR1_parquet.py` | 把 Easy-R1 风格 `*.parquet` 用 jinja 渲染后写出 `*.formatted.parquet`，并把 `extra_info["split"]` 注入到 `val` 行，让 reward 流程知道用 val 路径。 | 新增 |

#### LLM judge 对应表

| 维度 | Easy-R1 | verl 侧 |
|---|---|---|
| 入口 | `examples/reward_function/math_no_format.py::compute_score`（batch API） | `examples/branch_adv/reward_function/math_no_format.py::compute_score_batch`（verl `BatchRewardManager` API） |
| 触发条件 | rule → sympy → 入 judge 队列 | 同上，逻辑一字不改 |
| 并发 | `ThreadPoolExecutor(max_workers=512)` + 端口轮询 | 同上（同一份 `llm_judge.py`） |
| HTTP 客户端 | `requests.Session()` thread-local | 同上 |
| 默认 host / ports / model / endpoint / timeout / retries | `10.119.97.103 / 9000-9007 / /mnt/public/users/zhuyongfu/model/openai/gpt-oss-20b / /v1/chat/completions / 30 / 2` | **完全一致**（env 默认 verbatim） |
| Prompt 模板 | `examples/llm_judge_prompt/{math,sqa}.jinja` | `examples/branch_adv/llm_judge_prompt/{math,sqa}.jinja`（verbatim） |
| `verify_type=llava` 别名 | 自动归一为 `sqa` | 同上 |
| Score 解析 | 正则 `\b([TF])\b(?![\s\S]*\b[TF]\b)` 取最后一个 T/F | 同上 |
| Debug 日志 | `LLM_JUDGE_DEBUG=1 + LLM_JUDGE_DEBUG_LOG_PATH=...` 写 JSONL | 同上 |
| `prompt` 字段（仅 `sqa.jinja` 用到） | Easy-R1 batch reward 透传 `reward_input.get("prompt")` | verl 侧从 `extra_info.get("question") or extra_info.get("prompt")` 取；如果 dataset 里有这个 key，自动透传，否则置 `None`（math.jinja 不用，无影响） |
| 开关 | reward_function_kwargs 传 `enable_llm_judge=False` | `--enable_llm_judge False` 或 `+reward.custom_reward_function.reward_kwargs.enable_llm_judge=False` |

覆盖默认值的方式：

- **环境变量（推荐）**：`export LLM_JUDGE_HOST=...`、`LLM_JUDGE_PORTS=9000-9007`、`LLM_JUDGE_MODEL=...`，启动脚本会原样 `export` 给 ray actor。这与 Easy-R1 的做法一致。
- **CLI**：`--llm_judge_host / --llm_judge_ports / --llm_judge_model / --llm_judge_endpoint / --llm_judge_timeout / --llm_judge_max_retries / --llm_judge_max_workers / --llm_judge_max_tokens`，脚本会 `export` 同名环境变量。
- **Hydra override**：直接在脚本后追加 `+reward.custom_reward_function.reward_kwargs.llm_judge_host=...`。

> ⚠️ Sanity：`llm_judge.py` 模块顶端的 `_DEFAULT_*` 常量是 import-time 读取的，所以 `export` 必须在 `python3 -m recipe.dapo.main_dapo`（或 `verl.trainer.main_ppo`）**之前** 完成。脚本里已经处理。

### 1.4 启动脚本层（含 SwanLab）

| 脚本 | 入口 | 适配的算法 |
|---|---|---|
| `examples/branch_adv/run_grpo_branch_adv.sh` | `python3 -m verl.trainer.main_ppo` | 纯 GRPO + branch_adv（无 online_filtering） |
| `examples/branch_adv/run_dapo_branch_adv.sh` | `python3 -m recipe.dapo.main_dapo` (`RayDAPOTrainer.fit`) | **DAPO + branch_adv**：开 `algorithm.filter_groups.enable`、`clip_ratio_low/high=0.2/0.28`、`disable_kl=True`、`gen_batch_size` 对齐 Easy-R1 `rollout_batch_size`，`max_num_gen_batches=20` 对齐 `max_try_make_batch` |
| `examples/branch_adv/run_8b_non_thinking_04_30_20_wo_format_17k.sh` | 同上 | **直接对应原 Easy-R1 调用**的便捷 wrapper（参数预填好，剩下的 override 透传给 `run_dapo_branch_adv.sh`） |

SwanLab 在启动脚本里的对应关系：

| Easy-R1（`run_multinode_*.sh`） | verl 侧（`run_dapo_branch_adv.sh`） |
|---|---|
| `SWANLAB_API_KEY` env / `--swanlab_api_key` | 同名 env / CLI（脚本头部 `export`） |
| `SWANLAB_PROJECT` env / `--swanlab_project` | 同名 env / CLI；`trainer.project_name=${SWANLAB_PROJECT}` |
| `SWANLAB_MODE` env / `--swanlab_mode` | 同名 env / CLI（`cloud` / `local` / `offline`） |
| `EXPERIMENT_NAME = TASK_NAME_YYYYMMDDhhmm` | 同上 → `trainer.experiment_name=${EXPERIMENT_NAME}` |
| `trainer.logger=['swanlab']` | `trainer.logger='[swanlab]'`（Hydra 列表语法） |

SwanLab 默认值（CLI 不显式传时）：

```
SWANLAB_API_KEY  = (empty — pass --swanlab_api_key or export SWANLAB_API_KEY)
SWANLAB_PROJECT  = verl-dapo-branch-adv  (DAPO)  /  verl-grpo-branch-adv  (GRPO)
SWANLAB_MODE     = cloud
```

#### 1.4.1 `gen_batch_size` vs `train_batch_size` 的选择 ⚠️

DAPO 流程里这两个值不一样会显著改变 rollout 行为，先说三方对齐：

| 框架 | 参数 / 文件 | 默认关系 |
|---|---|---|
| **Easy-R1** | `examples/config_2k-8k_wo_format_04_03_20.yaml`：`rollout_batch_size: 256`，`mini_rollout_batch_size: null` | `mini_rollout_batch_size=null` 时 `verl/trainer/data_loader.py:54-57` fallback 到 `rollout_batch_size` ⇒ **gen = train = 256** |
| **verl 官方 dapo recipe** | `recipe/dapo/run_dapo_qwen3_moe_30b_megatron_npu.sh:27-28`：`train_prompt_bsz=16`、`gen_prompt_bsz=train_prompt_bsz*2=32` | **gen = train × 2**（over-rollout，让 `filter_groups` 丢掉不合格 group 之后仍能凑齐 train_batch） |
| **本仓库 `run_dapo_branch_adv.sh`** | L88-95：`TRAIN_BATCH_SIZE=256`、`GEN_BATCH_SIZE=$((TRAIN_BATCH_SIZE * 2))=512` | **gen = train × 2 = 512**（跟随 verl 官方，比 Easy-R1 快） |

为什么默认选 **gen = train × 2**（verl 官方风格）？

- DAPO `filter_groups` 会动态丢弃 acc 全 0 或全 1 的 group。如果 gen=train，过滤后凑不齐就要重新 rollout（消耗 `algorithm.filter_groups.max_num_gen_batches=20` 中的一次）；retry 越多，整体训练越慢。
- 提前 over-rollout 2 倍，单次 rollout 里就有冗余样本顶上，绝大多数 step 一发即过，不再触发 retry，训练显著加速。
- 代价是单步 rollout 的 GPU 消耗增加约 2×，但 verl 官方 dapo recipe 已在 Qwen3-MoE-30B 验证过这是合理 trade-off。
- Easy-R1 那一版用 1:1 是因为它没有这个 over-rollout 开关；既然 verl 提供了，迁移到 verl 后默认享受加速。

CLI 行为：

```bash
# 默认走 2:1
bash examples/branch_adv/run_dapo_branch_adv.sh ...   # GEN_BATCH_SIZE=512, TRAIN_BATCH_SIZE=256

# 改 train，gen 自动重算成 train * 2
bash examples/branch_adv/run_dapo_branch_adv.sh ... --train_batch_size 128   # gen=256

# 想退回 Easy-R1 风格 1:1，显式传 --gen_batch_size
bash examples/branch_adv/run_dapo_branch_adv.sh ... --gen_batch_size 256

# 自定义 train + gen
bash examples/branch_adv/run_dapo_branch_adv.sh ... --train_batch_size 128 --gen_batch_size 192
```

实现细节（脚本 L206-207 + L250-254）：

- `--train_batch_size N` 只改 `TRAIN_BATCH_SIZE`，**不再**联动改 `GEN_BATCH_SIZE`；
- `--gen_batch_size N` 设置 `GEN_BATCH_SIZE` 并打上 `GEN_BATCH_SIZE_EXPLICIT=1` 标志；
- CLI 解析结束后，如果 `GEN_BATCH_SIZE_EXPLICIT` 未设，自动 `GEN_BATCH_SIZE = TRAIN_BATCH_SIZE * 2`；
- 这样不管 `--train_batch_size` 和 `--gen_batch_size` 谁先谁后传，行为都一致。

- `VAL_BATCH_SIZE=1024` 固定，对齐 Easy-R1 yaml `val_batch_size: 1024`。

> 一句话：**默认 = verl 官方 2:1（快）**；想完全复现 Easy-R1 曲线就 `--gen_batch_size 256`（1:1）。

---

## 2. 推荐用法

### 2.1 跑用户原 Easy-R1 命令的等价版

用户原 Easy-R1 调用：

```bash
cd /mnt/public/users/liuwenpu/Easy-R1-branchadv
bash examples/run_multinode_1_02_20_2k_4k_dapo.sh \
  --model_path /mnt/public/users/liuwenpu/model/Qwen/Qwen3-8B \
  --train_data /mnt/public/users/liuwenpu/data/DAPO-Math-17k-Processed/train.parquet \
  --val_data   /mnt/public/users/liuwenpu/data/val_merge/val_merged.parquet \
  --task_name  8b_non_thinking_04_30_20_wo_format_17k \
  --output_path /mnt/public/users/liuwenpu/Easy-R1-branchadv/ckpt \
  --gpus 8 \
  --max_prompt_length 2048 --max_response_length 8192 \
  --config_path examples/config_2k-8k_wo_format_04_03_20.yaml \
  --swanlab_project new_math-2k-4k --swanlab_mode cloud
```

verl 等价命令（两步）：

**Step 1：一次性预渲染 jinja（只需做一次，后续复用 `*.formatted.parquet`）**

```bash
cd /mnt/public/users/liuwenpu/EDPO/verl

python3 examples/branch_adv/preprocess_easyR1_parquet.py \
  --input  /mnt/public/users/liuwenpu/data/DAPO-Math-17k-Processed/train.parquet \
  --output /mnt/public/users/liuwenpu/data/DAPO-Math-17k-Processed/train.formatted.parquet \
  --template examples/branch_adv/format_prompt/math_wo_format.jinja \
  --split train

python3 examples/branch_adv/preprocess_easyR1_parquet.py \
  --input  /mnt/public/users/liuwenpu/data/val_merge/val_merged.parquet \
  --output /mnt/public/users/liuwenpu/data/val_merge/val_merged.formatted.parquet \
  --template examples/branch_adv/format_prompt/val_wo_format.jinja \
  --split val
```

> 如果原 parquet 里 `prompt` 已经是 chat-list `[{role, content}]`，脚本会自动抽 `content` 字段；如果没有 `reward_model` 列（Easy-R1 val_merged.parquet 只有 `answer`），加 `--answer-key answer` 让它复制成 `reward_model.ground_truth`；缺 `data_source` 时加 `--data-source math_no_format`。

**Step 2：启动 DAPO + branch_adv**

```bash
cd /mnt/public/users/liuwenpu/EDPO/verl

bash examples/branch_adv/run_8b_non_thinking_04_30_20_wo_format_17k.sh
```

或者用完整 CLI：

```bash
bash examples/branch_adv/run_dapo_branch_adv.sh \
  --model_path /mnt/public/users/liuwenpu/model/Qwen/Qwen3-8B \
  --train_data /mnt/public/users/liuwenpu/data/DAPO-Math-17k-Processed/train.formatted.parquet \
  --val_data   /mnt/public/users/liuwenpu/data/val_merge/val_merged.formatted.parquet \
  --task_name  8b_non_thinking_04_30_20_wo_format_17k \
  --output_path /mnt/public/users/liuwenpu/EDPO/verl/ckpt \
  --gpus 8 \
  --max_prompt_length 2048 \
  --max_response_length 8192 \
  --enable_dynamic_max_tokens True \
  --branch_adv_enabled True \
  --branch_adv_alpha 0.4 \
  --branch_adv_beta 0.3 \
  --branch_adv_kappa 2.0 \
  --branch_adv_log_path logs/branch_adv.log \
  --swanlab_project new_math-2k-4k \
  --swanlab_mode cloud
```

CLI 与 Easy-R1 对照（语义完全一致，参数名一致）：

| Easy-R1 | verl 脚本 |
|---|---|
| `--model_path` | `--model_path` |
| `--train_data` | `--train_data`（指向 `*.formatted.parquet`） |
| `--val_data` | `--val_data`（指向 `*.formatted.parquet`） |
| `--task_name` | `--task_name`（拼到 `EXPERIMENT_NAME` 和 swanlab） |
| `--output_path` | `--output_path` → `trainer.default_local_dir` |
| `--gpus` | `--gpus` → `trainer.n_gpus_per_node` |
| `--max_prompt_length` | `--max_prompt_length` |
| `--max_response_length` | `--max_response_length` |
| `--config_path examples/config_2k-8k_wo_format_04_03_20.yaml` | **accepted for parity, ignored**（verl 用 `recipe/dapo/config/dapo_trainer.yaml` + Hydra override） |
| `--swanlab_project` | `--swanlab_project` |
| `--swanlab_mode` | `--swanlab_mode` |
| `--swanlab_api_key` | `--swanlab_api_key` |

verl 独有 / Easy-R1 默认开启的新参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--enable_dynamic_max_tokens True\|False` | True | 与 Easy-R1 同义；False 退回 verl 原版本（`min(response_length, capacity-prompt_len)`） |
| `--penalty_max_length <int>` / `--overlong_buffer_length <int>` | 50000 / 0 | 透传到 `reward.custom_reward_function.reward_kwargs`，对齐 Easy-R1 的 `worker.reward.reward_function_kwargs` |
| `--branch_adv_enabled / _alpha / _beta / _kappa / _warmup_start / _warmup_steps / _log_path` | True / 0.4 / 0.3 / 2.0 / 0 / 0 / `logs/branch_adv.log` | DAPO 默认值已经对齐 Easy-R1 yaml |
| `--enable_llm_judge True\|False` | True | 关掉就退回纯 rule + sympy |
| `--reward_fn_path / _name / --reward_manager` | `examples/branch_adv/reward_function/math_no_format.py` / `compute_score_batch` / `batch` | 退回 NaiveRewardManager：`--reward_fn_name compute_score --reward_manager naive` |

### 2.2 纯 GRPO（无 DAPO online_filtering）

```bash
bash examples/branch_adv/run_grpo_branch_adv.sh \
  --model_path /mnt/public/users/xieweichu/Qwen3-4B-Base \
  --train_data /mnt/public/users/liuwenpu/data/DAPO-Math-17k-Processed/train.formatted.parquet \
  --val_data   /mnt/public/users/liuwenpu/data/val_merge/val_merged.formatted.parquet \
  --task_name  4b_base_04_20_20_grpo_kl-loss \
  --output_path /mnt/public/users/liuwenpu/EDPO/verl/ckpt \
  --gpus 8 \
  --max_prompt_length 2048 \
  --max_response_length 4096 \
  --branch_adv_enabled True \
  --branch_adv_alpha 0.4 --branch_adv_beta 0.2 --branch_adv_kappa 2.0 \
  --branch_adv_log_path logs/branch_adv.log \
  --swanlab_project new_math-2k-4k --swanlab_mode cloud
```

入口走 `python3 -m verl.trainer.main_ppo`；不会启用 `algorithm.filter_groups`、`clip_ratio_high=0.28` 等 DAPO 专属知识点。

---

## 3. 数据流（GRPO + branch_adv，DAPO 同理多一步 filter_groups）

```
RLHFDataset(prompt.formatted.parquet)
  → rollout (vLLM, enable_dynamic_max_tokens 控制 per-prompt max_tokens)
  → BatchRewardManager.compute_rm_score
       (custom_reward_function = examples/branch_adv/reward_function/math_no_format.py::compute_score_batch)
       → rule (mathruler.grade_answer) → sympy → LLM judge fan-out
       → 返回 {score, accuracy, answer_pred, format, length_penalty, ...}
  → batch.non_tensor_batch.update(reward_extra_infos_dict)
  → ray_trainer.fit() 触发 compute_advantage(...)
       if algorithm.branch_adv_enabled and warmup_scale > 0:
         branch_adv_kwargs = {
           acc_list, answer_list, format_list,
           branch_adv_alpha = α * scale,
           branch_adv_beta  = β * scale,
           branch_adv_kappa = κ,           # κ 不受 scale 影响
           branch_adv_log_path = <save_dir>/<log>,
           branch_adv_step = global_steps,
           branch_adv_out_metrics = {}
         }
       core_algos.compute_grpo_outcome_advantage(..., **branch_adv_kwargs)
           → GRPO 组内标准化
           → _apply_branch_advantage_adjustment in-place 改写 scores
              Branch A: N_w ≤ 1                    → 跳过
              Branch B: 全错答案一致 (K = 1)         → Δ = -β * base
              Branch C: 错答案多样 (K > 1)          → Δ = α * base * C_i + κ-clip
       returns / values 用修改后的 scores
       metrics["branch_adv/*"] ← branch_adv_out_metrics
       → swanlab logger 上传
```

---

## 4. 数据集格式注意事项 ⚠️

Easy-R1 通过 `data.format_prompt: math_wo_format.jinja` 在加载阶段把每条 prompt 包一层 system 指令；**verl 没有这一层**，它直接用 HF `tokenizer.apply_chat_template` 把 dataset 里的 `prompt` 列 tokenize。

两个迁移选项：

1. **预处理 parquet（推荐，本仓库默认走这条）**：用 `examples/branch_adv/preprocess_easyR1_parquet.py` 把渲染后的字符串写入 `prompt` 列。该脚本会：
   - 渲染 `format_prompt/math_wo_format.jinja`（或 `val_wo_format.jinja`）；
   - 把结果包成 `[{"role": "user", "content": <rendered>}]`，与 verl `apply_chat_template` 的输入对齐；
   - 可选地把 `extra_info["split"]` 注入到 `train` / `val` 行，让 `compute_score_batch` 对 val 行走 rule-only（与 Easy-R1 `worker.val_reward` 等价）；
   - 缺 `reward_model` 列时按 `--answer-key` 补；缺 `data_source` 时按 `--data-source` 补。

   一次预处理，长期复用，下次 fine-tune 只跑 `Step 2`。

2. **走 chat-template + system prompt**：直接利用 Qwen3 base 的默认 chat_template，传 `data.apply_chat_template_kwargs.system_message="..."`（取决于具体 tokenizer）。本仓库不走这条，避免破坏其它依赖。

---

## 5. 暂未迁移 / 已知差异

- **`_log_group_error_stats` JSONL 日志（Easy-R1 调试用）**：未迁移。如需，复制 Easy-R1 `verl/trainer/ppo/ray_trainer.py` 里的同名方法到 `EDPO/verl/verl/trainer/ppo/ray_trainer.py` 即可。
- **`length_log_*` 实际写盘**：配置位已加（`enable_dynamic_max_tokens` / `length_log_interval` / `length_log_path`），未在 `vllm_async_server.generate` 里挂钩写盘。如需，把 Easy-R1 `vllm_rollout_spmd.py` 的 `_append_length_log` 拆成纯函数 import。
- **入口路径**：
  - GRPO 入口 = `verl.trainer.main_ppo`（deprecated 但功能完整）；
  - DAPO 入口 = `recipe.dapo.main_dapo` (`RayDAPOTrainer.fit`)，**已经透传 `branch_adv_kwargs`**（recipe.dapo 内部复用 `compute_advantage` 时不会丢字段）。
  - 如果将来切到推荐入口 `main_ppo_sync`（multi-trajectory），需要在 `main_ppo_sync.py::_compute_advantage` 内复刻 `ray_trainer.fit` 里同一段 warmup-scale + 取 acc/answer/format 的逻辑（13 行）。
- **Validation reward 独立配置**：Easy-R1 用单独的 `worker.val_reward.reward_function`；verl 只支持单一 `custom_reward_function`，所以我们让 `compute_score_batch` 自己读 `extra_info["split"]` 来分流。**预处理时必须把 `--split val` 加上**，否则 val 也会走 sympy + LLM judge（精度无误，但慢且占额度）。

---

## 6. 改动清单（diff 视图）

```
verl/trainer/config/algorithm.py                              # +branch_adv_* fields on AlgoConfig
verl/trainer/config/ppo_trainer.yaml                          # +algorithm.branch_adv_* defaults
verl/trainer/config/rollout/rollout.yaml                      # +enable_dynamic_max_tokens / length_log_*
verl/trainer/ppo/core_algos.py                                # +imports, +_apply_branch_advantage_adjustment, modified compute_grpo_outcome_advantage
verl/trainer/ppo/ray_trainer.py                               # +_compute_branch_adv_warmup_scale, branch_adv_kwargs in compute_advantage, fit() integration
verl/workers/config/rollout.py                                # +enable_dynamic_max_tokens / length_log_* on RolloutConfig
verl/workers/rollout/vllm_rollout/vllm_async_server.py        # honor enable_dynamic_max_tokens in generate()
examples/branch_adv/format_prompt/math_wo_format.jinja        # verbatim (used by preprocess only)
examples/branch_adv/format_prompt/val_wo_format.jinja         # verbatim (used by preprocess only)
examples/branch_adv/reward_function/math_no_format.py         # compute_score + compute_score_batch (1:1 Easy-R1 + split routing)
examples/branch_adv/reward_function/val_wo_format.py          # rule-only val reward (Easy-R1 1:1)
examples/branch_adv/reward_function/llm_judge.py              # ← verbatim from Easy-R1
examples/branch_adv/llm_judge_prompt/math.jinja               # ← verbatim from Easy-R1
examples/branch_adv/llm_judge_prompt/sqa.jinja                # ← verbatim from Easy-R1
examples/branch_adv/preprocess_easyR1_parquet.py              # 新增：一次性渲染 Easy-R1 jinja → verl 兼容 parquet
examples/branch_adv/run_grpo_branch_adv.sh                    # 入口 verl.trainer.main_ppo（GRPO + branch_adv）
examples/branch_adv/run_dapo_branch_adv.sh                    # 入口 recipe.dapo.main_dapo（DAPO + branch_adv，SwanLab plumbing）
examples/branch_adv/run_8b_non_thinking_04_30_20_wo_format_17k.sh  # 用户原 Easy-R1 调用的便捷 wrapper
examples/branch_adv/branch_adv_README.md                      # 早期版本说明（可忽略，已合并入本文件）
EDPO_MIGRATION.md                                             # ← 本文档
```

---

## 7. 常见问题

**Q1：训练起来之后 SwanLab 上看不到 `branch_adv/*` 曲线？**

A：在 `branch_adv_enabled=True` 且当前 step 已过 `warmup_start` 之后才会出现。脚本默认 `warmup_start=0 / warmup_steps=0` ⇒ 第 0 步起即满强度，应当立刻看到 `branch_adv/scale = 1.0`、`branch_adv/groups_processed`、`branch_adv/branch_b_count`、`branch_adv/branch_c_count` 等键。如果 `groups_processed` 一直 0，回去检查 reward function 是不是返回了 `accuracy` 和 `answer_pred` 这两个键，且这两个键被合并进 `non_tensor_batch`。

**Q2：训练 N 步后 LLM judge 把端口打爆 / timeout 飙升？**

A：调小 `LLM_JUDGE_MAX_WORKERS`（默认 512）或扩 `LLM_JUDGE_PORTS`；也可以临时 `--enable_llm_judge False` 退回 rule+sympy。

**Q3：我能换数据集吗，譬如不是 DAPO-Math-17k？**

A：可以。只要保证 dataset 满足：① `prompt` 列是 chat-list `[{role, content}]`（或用 `preprocess_easyR1_parquet.py` 转一道），② `reward_model.ground_truth` 列存在（同样可以用预处理脚本补），③ math/llava 类问题就用 `math.jinja`，其它 SQA 类问题改用 `sqa.jinja` 并把 `extra_info["verify_type"]` 设成 `sqa`。

**Q4：要把 `branch_adv_log_path` 关掉只看 SwanLab 指标？**

A：脚本里加 `--branch_adv_log_path ""` 即可，落到 yaml 默认 `null`，调整明细就不写盘。SwanLab 上的 `branch_adv/*` 不受影响。

**Q5：跑出的 ckpt 在哪里？**

A：`${OUTPUT_PATH}/${EXPERIMENT_NAME}`，其中 `EXPERIMENT_NAME = ${TASK_NAME}_${TASK_ID}`、`TASK_ID` 是启动时的 `YYYYMMDDhhmm`。SwanLab 实验名一致。
