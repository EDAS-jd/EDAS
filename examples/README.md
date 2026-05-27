# verl 示例

本目录收录了精心整理、依赖最少的示例,用于驱动 `verl.trainer.main_ppo`(基于当前的 Hydra API)。
针对算法的扩展、研究基线以及非平凡的入口脚本均放在 `recipe/` 目录下;
如果你需要超出本目录示例范围的自定义损失函数或奖励函数,请优先使用该目录。

## 约定

所有运行脚本都遵循相同的结构:

1. 规范化的文件名格式:

   ```
   run_<model>_<train-backend>.sh
   ```

   - `<model>`:每个模型族使用一个统一的规范化尺寸标识。例如:
     `qwen3_8b`、`qwen3_30b_a3b`、`qwen3_235b_a22b`、`qwen3_vl_8b`、
     `deepseek_v3`、`mimo_7b`、`nemotron_nano_v3`。
   - `<train-backend>`:取值为 `fsdp`、`fsdp2`、`megatron`、`mindspeed`、
     `automodel` 或 `veomni` 之一。**必须是 `.sh` 之前最后一个用下划线分隔的 token**。

   `<train-backend>` 之后不允许再添加任何内容。每个示例的*特性*——包括
   推理后端(`vllm`/`sglang`/`trtllm`)、平台(`DEVICE=gpu|npu`)、GPU 机型
   (`MACHINE=gb200`/`b200`/`blackwell`)、Liger kernel、LoRA、FP8 量化、序列并行
   大小、server 与 sync rollout 等——**不应**出现在文件名中。
   它们应当作为环境变量开关暴露在唯一的规范化脚本内部。
   不要新增 `_npu`、`_amd`、`_vllm`、`_sglang`、`_trtllm` 或 `_fp8` 这类脚本变体。
   例如,`sft/gsm8k/run_qwen2_5_0_5b_fsdp.sh` 通过环境变量同时覆盖了普通 SFT
   及其 `USE_LIGER=1`、`SP_SIZE=2`、`USE_PEFT=1` 等变体;
   `grpo_trainer/run_qwen3_8b_fsdp.sh` 则通过开关覆盖了 vLLM、SGLang、TRT-LLM rollout、
   CUDA/NPU 平台以及 `MACHINE=gb200`(Blackwell)等场景。

   该命名规则由 `check-example-naming` pre-commit 钩子强制校验
   (详见 `tests/special_sanity/check_example_naming.py`)。

2. 每个脚本都会在文件顶部一个明确的"用户可调整区域"中暴露重要参数。
   派生的默认值以及与设备/后端相关的实现细节应放在
   "no user adjustment needed below" / "derived defaults" 分界线之下。
   面向用户的可调参数请使用大写环境变量,例如:

   ```bash
   # ---- user-adjustable ----
   DEVICE=${DEVICE:-gpu}
   INFER_BACKEND=${INFER_BACKEND:-vllm}
   MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
   NNODES=${NNODES:-1}
   NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-}
   TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
   ROLLOUT_TP=${ROLLOUT_TP:-2}
   ROLLOUT_N=${ROLLOUT_N:-5}
   PROJECT_NAME=${PROJECT_NAME:-verl_grpo_gsm8k_math}
   EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_8b_grpo_vllm_fsdp}
   # ---- end user-adjustable ----
   ...
   ```

   所有你关心的参数都可以在命令行中覆盖:

   ```bash
   DEVICE=npu MODEL_PATH=/my/local/qwen3-8b NDEVICES_PER_NODE=4 bash examples/grpo_trainer/run_qwen3_8b_fsdp.sh
   ```

   GPU 与 NPU 路径应共享相同的 `PROJECT_NAME` / `EXPERIMENT_NAME` 形式。
   不要仅仅因为选择了 `DEVICE=npu` 就在 project 或 experiment 名称后追加 `_npu`。

3. 默认配置(除非某个目录明确说明了不同的约定):

   - 文本 LLM 的 `data.train_files` + `data.val_files` = GSM8K + MATH
     (视觉模型使用 `geo3k`;面向规模演示的 235B / 671B 脚本使用
     `dapo-math-17k` / `aime-2024`)。
   - `actor_rollout_ref.actor.use_dynamic_bsz=True`
   - `trainer.balance_batch=True`
   - `trainer.logger=["console","wandb"]`。

4. 不再使用已弃用的 Hydra 参数:

   - `ppo_megatron_trainer.yaml` → 改用 `actor_rollout_ref.actor.model_engine=megatron`。
   - `actor_rollout_ref.rollout.mode=async` → 已移除;示例脚本中不再通过这种方式选择异步 rollout。
   - `actor_rollout_ref.hybrid_engine=True` → 已移除;trainer 现在会在内部强制使用受支持的 hybrid-engine 路径。
   - `ppo_micro_batch_size` / `log_prob_micro_batch_size` → 改用带 `_per_gpu` 后缀的版本。
   - `data.val_batch_size` → 已移除。
   - 顶层的 `reward_model.*` → 视情况改用 `reward_model.reward_model.*` / `reward.reward_model.*`。
   - `actor.ulysses_sequence_parallel_size` → 改用
     `actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size`。

## 目录结构

### 算法 trainer

每个目录包含一种训练算法的规范化实现。如果你想新增一种算法,
且它需要独立的 trainer 入口或奖励代码,请将其放到 `recipe/` 目录下。

| 目录                                | 算法                            | `algorithm.adv_estimator` / `policy_loss.loss_mode` |
|------------------------------------|--------------------------------|-----------------------------------------------------|
| `ppo_trainer/`                     | PPO(actor + critic)            | `adv_estimator=gae`                                 |
| `grpo_trainer/`                    | GRPO                           | `adv_estimator=grpo`                                |
| `rloo_trainer/`                    | RLOO                           | `adv_estimator=rloo`                                |
| `remax_trainer/`                   | ReMax                          | `adv_estimator=remax`                               |
| `reinforce_plus_plus_trainer/`     | REINFORCE++ / baseline         | `adv_estimator=reinforce_plus_plus[_baseline]`      |
| `cispo_trainer/`                   | CISPO                          | `loss_mode=cispo`                                   |
| `dppo_trainer/`                    | DPPO(TV / KL 变体)             | `loss_mode=dppo_tv \| dppo_kl`                      |
| `gdpo_trainer/`                    | GDPO                           | `adv_estimator=gdpo`                                |
| `gmpo_trainer/`                    | GMPO                           | `loss_mode=geo_mean`                                |
| `gpg_trainer/`                     | GPG                            | `adv_estimator=gpg`,`loss_mode=gpg`                 |
| `gspo_trainer/`                    | GSPO                           | `loss_mode=gspo`                                    |
| `sapo_trainer/`                    | SAPO                           | `loss_mode=sapo`                                    |
| `otb_trainer/`                     | OTB                            | `adv_estimator=optimal_token_baseline`              |
| `mtp_trainer/`                     | DAPO + MTP(MiMo-7B)            | `adv_estimator=grpo`,MTP 相关 flag                   |
| `on_policy_distillation_trainer/`  | on-policy 蒸馏                  | GRPO + 蒸馏损失                                       |
| `flowgrpo_trainer/`                | Flow-GRPO(扩散模型)            | 图像生成专用                                          |

### 特性 / 基础设施

| 目录                  | 用途                                                                                       |
|----------------------|------------------------------------------------------------------------------------------|
| `tuning/`            | LoRA(`tuning/lora/`)以及 scaling 演示(`tuning/scaling/`)。                                |
| `profile/`           | NPU profiler / torch-memory profiler 相关运行。                                             |
| `sft/`               | 监督微调示例。                                                                              |
| `generation/`        | 仅推理(rollout-only)的启动脚本。                                                            |
| `vllm_omni/`         | vLLM omni 后端示例。                                                                        |
| `data_preprocess/`   | 用于生成运行脚本所期望的 `$HOME/data/<dataset>/*.parquet` 数据布局的脚本。                       |
| `prefix_grouper/`    | Prefix-grouped rollout 示例。                                                              |
| `rollout_correction/`| Rollout correction(rollout 校正)示例。                                                     |
| `router_replay/`     | Router replay 示例。                                                                       |
| `tutorial/`          | 教程及集群启动器(`ray/`、`slurm/`、`skypilot/`、`agent_loop_get_started/`)。                |

### 算法研究变体在哪里?

请查看 `recipe/` 目录——例如 `recipe/dapo`、`recipe/prime`、`recipe/retool`、
`recipe/r1`、`recipe/spin`、`recipe/gvpo`、`recipe/flowrl` 等。
它们各自带有独立的 trainer 入口和奖励代码。
