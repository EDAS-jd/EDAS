# EDPO (branch_adv) — Easy-R1 → verl 迁移说明

把 `Easy-R1-branchadv` 上跑通的 **GRPO + branch_adv (EDPO)** 算法、reward、jinja、动态 max-tokens 等流程移植到 `EDPO/verl` 框架。本文档记录改动点、启动方式、未覆盖项。

> 算法本体的语义见 `Easy-R1-branchadv/branch_adv_implementation.md`，本文档不重复推导。

---

## 1. 改动概览

### 1.1 核心算法（对齐原 Easy-R1）

| 文件 | 改动 |
|---|---|
| `verl/trainer/config/algorithm.py` | `AlgoConfig` 新增 `branch_adv_enabled / alpha / beta / kappa / log_path / warmup_start / warmup_steps / acc_key / answer_key / format_key`。 |
| `verl/trainer/config/ppo_trainer.yaml` | `algorithm:` 节加上同名键，Hydra 默认值与 dataclass 一致。 |
| `verl/trainer/ppo/core_algos.py` | 新增 `_apply_branch_advantage_adjustment(...)`（三分支：A 跳过 / B 模式坍缩统一惩罚 / C 健康发散按相对惊奇度调整 + κ-clip + 日志）。`compute_grpo_outcome_advantage` 接收 `acc_list / answer_list / format_list / branch_adv_*` 参数，在标准化之后、`unsqueeze` 之前调用上述函数 in-place 修改 `scores`。`mathruler.grader.grade_answer` 用于 canonical 去重，缺包时回落到字符串相等。 |
| `verl/trainer/ppo/ray_trainer.py` | 新增 `_compute_branch_adv_warmup_scale`（与 Easy-R1 一致的延迟期 / 爬坡期 / 稳定期）。`compute_advantage` 增 `branch_adv_kwargs`；`fit()` 在 GRPO 算 advantage 前从 `batch.non_tensor_batch[accuracy / answer_pred / format]` 取值，按 warmup scale 缩放 α/β，把日志路径拼成相对 `trainer.default_local_dir` 的绝对路径，调用 `compute_advantage(..., branch_adv_kwargs=...)`，把返回的统计量写入 `metrics["branch_adv/*"]`。|

> 关键约定：reward function 的返回 dict 必须包含 `accuracy`、`answer_pred`（branch_adv 必读）和可选的 `format`（≥0.5 才参与 branch_adv，缺省视为通过）。verl 的 reward 流程会自动把这三个 key 放进 `non_tensor_batch`，trainer 直接读取。

### 1.2 动态 max_tokens / 长度日志

| 文件 | 改动 |
|---|---|
| `verl/workers/config/rollout.py` | `RolloutConfig` 新增 `enable_dynamic_max_tokens: bool = False`、`length_log_interval: int = 0`、`length_log_path: Optional[str] = None`。 |
| `verl/trainer/config/rollout/rollout.yaml` | 同步声明上面三个键。 |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py` | `generate(...)` 默认分支：`enable_dynamic_max_tokens=True` 时 `max_tokens = prompt_length + response_length - len(prompt_ids)`（去掉 `response_length` 上界），与 Easy-R1 同义；False 时维持 verl 原行为 `min(response_length, capacity-prompt_len)`。 |

> 备注：verl 默认就在能力范围内动态裁剪 max_tokens；`enable_dynamic_max_tokens=True` 的真正语义是“**允许 response 比 `response_length` 长**，只受 `max_model_len` 限制”，与 Easy-R1 行为一致。length log 的实时写文件功能（min/mean/max 摘要）目前仅暴露了配置位，未在 vllm_async_server 内回填写盘逻辑（vLLM 自带 stats 已能覆盖大部分场景；如需完整 Easy-R1 风格逐 step 日志，可后续把 Easy-R1 `vllm_rollout_spmd.py` 里的 `_append_length_log` 拆成纯函数复用）。

### 1.3 Reward / Jinja

| 路径 | 内容 |
|---|---|
| `examples/branch_adv/reward_function/math_no_format.py` | 提供两个入口：<br>① `compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)` per-sample，**不带 LLM judge**（`NaiveRewardManager` 用，作 fallback / debug）。<br>② `compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos, **kwargs)` batch，**1:1 复刻 Easy-R1 `math_no_format.compute_score`** 的流程：rule (`mathruler.grade_answer`) → sympy → **LLM judge fan-out**（线程池 + 端口轮询，通过 verl 的 `BatchRewardManager` 走）。返回 `{score, overall, accuracy, length_penalty, answer_pred, format}`，`score = accuracy + overlong_penalty_factor * length_penalty`。 |
| `examples/branch_adv/reward_function/llm_judge.py` | **从 Easy-R1 verbatim 移植**（仅调整 `PROMPT_DIR` 的相对路径，因为 verl 侧把 `llm_judge_prompt/` 放进 `examples/branch_adv/` 而不是 `examples/`）。HTTP 客户端 / 端口轮询 / `ThreadPoolExecutor` / debug log 全部保留；默认 `host=10.119.97.103, ports=9000-9007, model=/mnt/public/users/zhuyongfu/model/openai/gpt-oss-20b, endpoint=/v1/chat/completions, timeout=30s, max_retries=2, max_workers=512, max_tokens=1024`，**与 Easy-R1 一致**；env 或 `--llm_judge_*` CLI 都可覆盖。 |
| `examples/branch_adv/llm_judge_prompt/math.jinja` | **从 Easy-R1 `examples/llm_judge_prompt/math.jinja` verbatim 拷贝**。判分规则 / scoring rules / 示例完全一致。 |
| `examples/branch_adv/llm_judge_prompt/sqa.jinja` | 同上，verbatim。`verify_type=llava` 自动归一为 `sqa`，与 Easy-R1 一致。 |
| `examples/branch_adv/reward_function/val_wo_format.py` | 验证集打分，仅做 `extract_boxed_content + grade_answer`（Easy-R1 val reward 本来就不调 judge，保持一致）。 |
| `examples/branch_adv/format_prompt/math_wo_format.jinja` | 与 Easy-R1 同名模板原样保留，**仅作参考**：verl 直接用 HF `tokenizer.apply_chat_template`，不消费 `data.format_prompt`。如要复用 Easy-R1 的 system prompt，请把它提前写进 parquet 的 `prompt` 列（见 §3）。 |
| `examples/branch_adv/format_prompt/val_wo_format.jinja` | 同上，验证用。 |

> ⚠️ Reward manager 切换：`compute_score_batch` 走 verl 的 `BatchRewardManager`，启动脚本默认已经把 `reward.reward_manager.name=batch` 加上了。如果你想退回到无 judge 的 per-sample 模式，命令行追加 `--reward_fn_name compute_score --reward_manager naive` 即可。

#### 1.3.1 LLM judge 是怎么对应的

| 维度 | Easy-R1 | verl 侧（本目录） |
|---|---|---|
| 入口 | `examples/reward_function/math_no_format.py::compute_score`（batch API，受 `worker.reward.reward_function_kwargs` 控制） | `examples/branch_adv/reward_function/math_no_format.py::compute_score_batch`（verl `BatchRewardManager` API，受 `reward.custom_reward_function.reward_kwargs` 控制） |
| 触发条件 | rule (`grade_answer`) 失败 → sympy 失败 → 入 `judge_jobs` 队列 | 同上，逻辑一字不改 |
| 并发 | `ThreadPoolExecutor(max_workers=512)` + 端口轮询 | 同上（同一份 `llm_judge.py`） |
| HTTP 客户端 | `requests.Session()` thread-local | 同上 |
| 默认 host / ports / model / endpoint / timeout / retries | `10.119.97.103 / 9000-9007 / /mnt/public/users/zhuyongfu/model/openai/gpt-oss-20b / /v1/chat/completions / 30 / 2` | **完全一致**（同一份代码，env 默认值 verbatim 保留） |
| Prompt 模板 | `examples/llm_judge_prompt/{math,sqa}.jinja` | `examples/branch_adv/llm_judge_prompt/{math,sqa}.jinja`（verbatim 拷贝） |
| `verify_type=llava` 别名 | 自动归一为 `sqa` | 同上 |
| Score 解析 | 正则 `\b([TF])\b(?![\s\S]*\b[TF]\b)` 取最后一个 T/F | 同上 |
| Debug 日志 | `LLM_JUDGE_DEBUG=1 + LLM_JUDGE_DEBUG_LOG_PATH=...` 写 JSONL | 同上 |
| `prompt` 字段（仅 `sqa.jinja` 用到） | Easy-R1 batch reward 透传 `reward_input.get("prompt")` | verl 侧从 `extra_info.get("question") or extra_info.get("prompt")` 取；如果 dataset 里有这个 key，自动透传，否则置 `None`（math.jinja 不用，无影响） |
| 关 / 开 | reward_function_kwargs 传 `enable_llm_judge=False` | `--enable_llm_judge False` 或 `+reward.custom_reward_function.reward_kwargs.enable_llm_judge=False` |

覆盖默认值（host/port/model/...）的方式：

- **环境变量**（推荐）：`export LLM_JUDGE_HOST=...`、`LLM_JUDGE_PORTS=9000-9007`、`LLM_JUDGE_MODEL=...`，启动脚本会原样 `export` 给 ray actor。这与 Easy-R1 的做法一致。
- **CLI**：`--llm_judge_host / --llm_judge_ports / --llm_judge_model / --llm_judge_endpoint / --llm_judge_timeout / --llm_judge_max_retries / --llm_judge_max_workers / --llm_judge_max_tokens`。脚本会 `export` 同名环境变量。
- **Hydra override**：直接在命令尾加 `+reward.custom_reward_function.reward_kwargs.llm_judge_host=...`，会以参数形式传进 `compute_score_batch`，覆盖 env 默认。

> Sanity：`llm_judge.py` 模块顶端的 `_DEFAULT_*` 常量是 import-time 读取的，所以 `export` 必须在 `python3 -m verl.trainer.main_ppo` **之前** 完成。脚本里这步已经处理。

### 1.4 启动脚本

`examples/branch_adv/run_grpo_branch_adv.sh` —— 命令行参数尽量对齐 Easy-R1：
`--model_path / --train_data / --val_data / --task_name / --output_path / --gpus / --max_prompt_length / --max_response_length / --enable_dynamic_max_tokens / --penalty_max_length / --overlong_buffer_length / --swanlab_*`，并新增 `--branch_adv_*` 一组开关。`--config_path` 仍然接受但忽略（verl 用 Hydra `ppo_trainer.yaml` 默认 + CLI override，跟 Easy-R1 单 yaml 模型不同）。

---

## 2. 启动命令（与原 Easy-R1 调用 1:1 对应）

```bash
cd /mnt/public/users/liuwenpu/EDPO/verl

bash examples/branch_adv/run_grpo_branch_adv.sh \
  --model_path /mnt/public/users/xieweichu/Qwen3-4B-Base \
  --train_data /mnt/public/users/liuwenpu/data/DAPO-Math-17k-Processed/train.parquet \
  --val_data   /mnt/public/users/liuwenpu/data/val_merge/val_merged.parquet \
  --task_name  4b_base_04_20_20_grpo_kl-loss \
  --output_path /mnt/public/users/liuwenpu/EDPO/verl/ckpt \
  --gpus 8 \
  --max_prompt_length 2048 \
  --max_response_length 4096 \
  --enable_dynamic_max_tokens True \
  --branch_adv_enabled True \
  --branch_adv_alpha 0.4 \
  --branch_adv_beta 0.2 \
  --branch_adv_kappa 2.0 \
  --branch_adv_log_path logs/branch_adv.log \
  --swanlab_project new_math-2k-4k \
  --swanlab_mode cloud
```

可选 override 直接接在脚本后面（Hydra 风格），脚本会原样转发：

```bash
# 例：临时关掉 branch_adv 跑纯 GRPO baseline
bash examples/branch_adv/run_grpo_branch_adv.sh ... --branch_adv_enabled False
```

### 对照表（Easy-R1 ↔ verl）

| Easy-R1 入口 | verl 入口 |
|---|---|
| `examples/run_multinode_1_02_20_2k_4k_grpo.sh` | `examples/branch_adv/run_grpo_branch_adv.sh` |
| `examples/config_2k-4k_wo_format_04_02_20_grpo_kl-loss.yaml` | `verl/trainer/config/ppo_trainer.yaml` + `algorithm.branch_adv_*` CLI override |
| `examples/format_prompt/math_wo_format.jinja` | `examples/branch_adv/format_prompt/math_wo_format.jinja`（仅文档） |
| `examples/reward_function/math_no_format.py`（batch API） | `examples/branch_adv/reward_function/math_no_format.py::compute_score_batch`（batch API，**已包含 LLM judge fan-out**） |
| `examples/reward_function/llm_judge.py` | `examples/branch_adv/reward_function/llm_judge.py`（verbatim） |
| `examples/llm_judge_prompt/{math,sqa}.jinja` | `examples/branch_adv/llm_judge_prompt/{math,sqa}.jinja`（verbatim） |
| `algorithm.branch_adv_*` (Easy-R1 dataclass) | `algorithm.branch_adv_*` (verl `AlgoConfig`) |
| `worker.rollout.enable_dynamic_max_tokens` | `actor_rollout_ref.rollout.enable_dynamic_max_tokens` |
| `worker.reward.reward_function_kwargs.{penalty_max_length, overlong_buffer_length}` | `reward.custom_reward_function.reward_kwargs.{penalty_max_length, overlong_buffer_length}` |

---

## 3. 数据集格式注意事项 ⚠️

Easy-R1 通过 `data.format_prompt: math_wo_format.jinja` 在加载阶段把每条 prompt 包一层 system 指令；**verl 没有这一层**，它直接用 HF `tokenizer.apply_chat_template` 把 dataset 里的 `prompt` 列 tokenize。

两个迁移选项任选其一：

1. **预处理 parquet（推荐）**：把 `examples/branch_adv/format_prompt/math_wo_format.jinja` 渲染后的字符串写入 `prompt` 列，原始题目放到 `extra_info`/`raw_prompt`。  
   ```python
   import pandas as pd, jinja2
   tpl = jinja2.Template(open("examples/branch_adv/format_prompt/math_wo_format.jinja").read())
   df = pd.read_parquet("DAPO-Math-17k-Processed/train.parquet")
   df["prompt"] = df["problem"].map(lambda c: [{"role": "user", "content": tpl.render(content=c)}])
   df.to_parquet("DAPO-Math-17k-Processed/train.formatted.parquet")
   ```
2. **走 chat-template + system prompt**：直接利用 Qwen3 base 的默认 chat_template，传 `data.apply_chat_template_kwargs.system_message="..."`（取决于具体 tokenizer）。

> 我没有动 verl 的 dataset 加载逻辑，避免破坏其它依赖；预处理一次比改框架更稳。

---

## 4. 数据流（以 GRPO + branch_adv 为例）

```
RLHFDataset → rollout (vLLM, enable_dynamic_max_tokens 控制 per-prompt max_tokens)
            → RewardLoopManager.compute_rm_score
                (custom_reward_function = examples/branch_adv/reward_function/math_no_format.py)
                返回 {score, accuracy, answer_pred, format, length_penalty, ...}
            → batch.non_tensor_batch.update(reward_extra_infos_dict)
            → ray_trainer.fit() 触发 compute_advantage:
                if algorithm.branch_adv_enabled and warmup_scale > 0:
                    branch_adv_kwargs = {acc_list, answer_list, format_list,
                                         branch_adv_alpha=α*scale,
                                         branch_adv_beta =β*scale,
                                         branch_adv_kappa=κ,
                                         branch_adv_log_path=<save_dir>/<log>,
                                         branch_adv_step=global_steps,
                                         branch_adv_out_metrics={}}
                core_algos.compute_grpo_outcome_advantage(..., **branch_adv_kwargs)
                    → GRPO 组内标准化
                    → _apply_branch_advantage_adjustment in-place 改写 scores
                metrics["branch_adv/*"] ← out_metrics
```

---

## 5. 暂未迁移 / 已知差异

- **Online filtering（DAPO 风格组过滤）**：Easy-R1 在 `_make_batch_data` 里用 `algorithm.online_filtering` 过滤 group。verl 用 `algorithm.filter_groups` 走另一条路径（在 main_ppo_sync 里），未把 Easy-R1 的实现一比一搬过来。如需：默认关；启用走 verl 自带 `filter_groups`。
- **`_log_group_error_stats` JSONL 日志**：调试用，未迁移。如需，复制 Easy-R1 ray_trainer.py 里的同名方法到 verl ray_trainer 即可。
- **`length_log_*` 实际写盘**：配置位已加，未在 `vllm_async_server.generate` 里挂钩；需要的话把 Easy-R1 `vllm_rollout_spmd.py` 的 `_append_length_log` 直接 import 用即可。
- **入口建议**：本次改动针对 `verl.trainer.main_ppo`（即 deprecated 但功能完整的旧入口）。verl 推荐的 `main_ppo_sync` 走 multi-trajectory 流程（`compute_advantage_for_multi_trajectories`），同样调用同一个 `compute_advantage`，**默认仍能透传 branch_adv_kwargs**，但 `_compute_advantage` 里没显式构造它；如果你切到 sync 入口跑，且需要 branch_adv 生效，请在 `main_ppo_sync.py:_compute_advantage` 内复刻同一段 warmup-scale + 取 acc/answer/format 的逻辑（实现就在 `ray_trainer.fit` 里，可直接拷贝）。
- **Validation reward**：Easy-R1 用单独的 `worker.val_reward.reward_function`；verl 暂只支持单一 `custom_reward_function`。如需对验证集做不同打分，可在 `compute_score` 内根据 `extra_info.get("split")` 分流，或扩 verl reward 流程。

---

## 6. 改动清单（diff 视图）

```
verl/trainer/config/algorithm.py                    # +branch_adv_* fields on AlgoConfig
verl/trainer/config/ppo_trainer.yaml                # +algorithm.branch_adv_* defaults
verl/trainer/config/rollout/rollout.yaml            # +enable_dynamic_max_tokens / length_log_*
verl/trainer/ppo/core_algos.py                      # +imports, +_apply_branch_advantage_adjustment, modified compute_grpo_outcome_advantage
verl/trainer/ppo/ray_trainer.py                     # +_compute_branch_adv_warmup_scale, branch_adv_kwargs in compute_advantage, fit() integration
verl/workers/config/rollout.py                      # +enable_dynamic_max_tokens / length_log_* on RolloutConfig
verl/workers/rollout/vllm_rollout/vllm_async_server.py  # honor enable_dynamic_max_tokens in generate()
examples/branch_adv/format_prompt/math_wo_format.jinja
examples/branch_adv/format_prompt/val_wo_format.jinja
examples/branch_adv/reward_function/math_no_format.py
examples/branch_adv/reward_function/val_wo_format.py
examples/branch_adv/reward_function/llm_judge.py        # ← verbatim from Easy-R1
examples/branch_adv/llm_judge_prompt/math.jinja         # ← verbatim from Easy-R1
examples/branch_adv/llm_judge_prompt/sqa.jinja          # ← verbatim from Easy-R1
examples/branch_adv/run_grpo_branch_adv.sh
examples/branch_adv/branch_adv_README.md            # ← 本文档
```

---

## 7. Sanity check 建议

1. 关 branch_adv 跑一轮：`--branch_adv_enabled False`，确认 GRPO 基线复现。
2. 开 branch_adv 单步：把 `total_training_steps=1`，看 `metrics["branch_adv/groups_b/c/skip"]` 是否合理；`branch_adv_log_path` 指向的文件应有 `=== step 0 (branch_adv) ===` header。
3. 切几次 `branch_adv_warmup_start / warmup_steps`，验证 `branch_adv/scale` 指标的爬坡曲线。
4. 把 `--enable_dynamic_max_tokens False`，对比 max length 直方图是否退回 `response_length` 的硬上界。
