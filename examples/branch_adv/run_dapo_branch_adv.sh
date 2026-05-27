#!/bin/bash
# verl DAPO + branch_adv (EDPO) launcher.
# Mirrors the CLI surface of Easy-R1's run_multinode_1_02_20_2k_4k_dapo.sh
# so the same invocation works against the verl framework via recipe/dapo.
#
# What this script wires up that the GRPO variant does NOT:
#   - Entry point: python3 -m recipe.dapo.main_dapo (RayDAPOTrainer.fit)
#   - algorithm.filter_groups.enable=True (DAPO dynamic sampling, equivalent to
#     Easy-R1 online_filtering)
#   - DAPO clip ratios (0.2 / 0.28) and disable_kl semantics
#   - data.gen_batch_size mirroring Easy-R1's rollout_batch_size, with
#     max_num_gen_batches = Easy-R1 max_try_make_batch
#
# Reward path is identical to the GRPO variant: BatchRewardManager + the
# Easy-R1 1:1 compute_score_batch with LLM judge fan-out.
#
# IMPORTANT: To match Easy-R1's split-specific reward (train uses rule + sympy +
# LLM judge, val uses rule-only), the val parquet must be preprocessed with
# `--split val` so each row carries extra_info.split='val'. compute_score_batch
# reads that flag and skips sympy + LLM judge + length_penalty for val samples.

set -euo pipefail
set -x

# ==================== 0. SwanLab defaults ====================
python3 -c "import swanlab" 2>/dev/null || pip install -U swanlab
SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-verl-dapo-branch-adv}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"

export VERL_DEBUG_PADDING="${VERL_DEBUG_PADDING:-1}"

# ==================== 1. Distributed env ====================
target_master_addr="${MASTER_ADDR:-127.0.0.1}"
target_master_port="${MASTER_PORT:-29500}"
target_nnodes="${WORLD_SIZE:-1}"
target_node_rank="${RANK:-0}"
ray_port="${RAY_PORT:-6379}"

echo "------------------------------------------------"
echo "verl (DAPO + branch_adv) Distributed Config"
echo "Master Addr: $target_master_addr"
echo "Master Port: $target_master_port"
echo "Ray Port:    $ray_port"
echo "NNodes:      $target_nnodes"
echo "Node Rank:   $target_node_rank"
echo "------------------------------------------------"

# ==================== 2. Defaults (CLI overridable) ====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ID="$(date +%Y%m%d%H%M)"
NUM_GPUS=8

# DAPO core hyperparameters (match Easy-R1 config_2k-8k_wo_format_04_03_20.yaml
# after the CLI overrides used in run_multinode_1_02_20_2k_4k_dapo.sh)
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
DISABLE_KL=True                # -> use_kl_in_reward=False + actor.use_kl_loss=False
ONLINE_FILTERING=True          # -> algorithm.filter_groups.enable
FILTER_METRIC=accuracy         # binary, std-based filter is equivalent to Easy-R1's 0.01<mean<0.99
MAX_TRY_MAKE_BATCH=20          # -> filter_groups.max_num_gen_batches (Easy-R1 default)

SAVE_FREQ=10
TEST_FREQ=5
SAVE_LIMIT=20

# Lengths (CLI overridable)
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=8192
ENABLE_DYNAMIC_MAX_TOKENS=True
PENALTY_MAX_LENGTH=50000
OVERLONG_BUFFER_LENGTH=0

# Rollout / actor (mirrors Easy-R1 config_2k-8k_wo_format_04_03_20.yaml)
ROLLOUT_N=10
ROLLOUT_TP=2
ROLLOUT_GPU_MEM_UTIL=0.8
ROLLOUT_MAX_NUM_BATCHED_TOKENS=32768

ACTOR_PPO_MINI_BATCH_SIZE=64
ACTOR_PPO_MICRO_BSZ_PER_GPU=1
ACTOR_LR=1e-6
ACTOR_WEIGHT_DECAY=1.0e-2
ACTOR_ULYSSES=1
ACTOR_GRAD_NORM=1.0

# Easy-R1's rollout_batch_size = 256, val_batch_size = 1024.
TRAIN_BATCH_SIZE=256
VAL_BATCH_SIZE=1024
# Follow verl's official DAPO recipe (recipe/dapo/run_dapo_qwen3_moe_30b_megatron_npu.sh):
# gen_batch_size = train_batch_size * 2. Over-rollout 2x so DAPO filter_groups
# can discard unqualified groups without triggering as many retry rollouts; faster
# than Easy-R1's 1:1 default. CLI --gen_batch_size overrides; --train_batch_size N
# re-derives GEN_BATCH_SIZE = N * 2 unless --gen_batch_size is also passed.
GEN_BATCH_SIZE=$((TRAIN_BATCH_SIZE * 2))

# Branch-adv (EDPO) defaults — match the user's Easy-R1 yaml.
BRANCH_ADV_ENABLED=True
BRANCH_ADV_ALPHA=0.4
BRANCH_ADV_BETA=0.3
BRANCH_ADV_KAPPA=2.0
BRANCH_ADV_LOG_PATH="logs/branch_adv.log"
BRANCH_ADV_WARMUP_START=0
BRANCH_ADV_WARMUP_STEPS=0

# Required (CLI)
MODEL_PATH=""
TRAINDATA_PATH=""
VALDATA_PATH=""
TASK_NAME=""
OUTPUT_PATH=""

# Reward function (verl-side, has LLM judge fan-out)
REWARD_FN_PATH="${SCRIPT_DIR}/reward_function/math_no_format.py"
REWARD_FN_NAME="compute_score_batch"
REWARD_MANAGER="batch"

# LLM judge defaults — read by examples/branch_adv/reward_function/llm_judge.py
LLM_JUDGE_HOST="${LLM_JUDGE_HOST:-10.119.97.103}"
LLM_JUDGE_PORTS="${LLM_JUDGE_PORTS:-9000-9007}"
LLM_JUDGE_MODEL="${LLM_JUDGE_MODEL:-/mnt/public/users/zhuyongfu/model/openai/gpt-oss-20b}"
LLM_JUDGE_ENDPOINT="${LLM_JUDGE_ENDPOINT:-/v1/chat/completions}"
LLM_JUDGE_TIMEOUT="${LLM_JUDGE_TIMEOUT:-30}"
LLM_JUDGE_MAX_RETRIES="${LLM_JUDGE_MAX_RETRIES:-2}"
LLM_JUDGE_MAX_WORKERS="${LLM_JUDGE_MAX_WORKERS:-512}"
LLM_JUDGE_MAX_TOKENS="${LLM_JUDGE_MAX_TOKENS:-1024}"
ENABLE_LLM_JUDGE="${ENABLE_LLM_JUDGE:-True}"

# Validation override (Easy-R1 worker.rollout.val_override_config)
VAL_TEMPERATURE=0.7
VAL_TOP_P=0.8
VAL_TOP_K=20
VAL_N=1

TOTAL_EPOCHS=2

usage() {
  cat <<EOF
Usage:
  bash run_dapo_branch_adv.sh \\
    --model_path <path> \\
    --train_data <path>        # *.formatted.parquet (preprocessed via preprocess_easyR1_parquet.py) \\
    --output_path <path> \\
    [--val_data <path>] \\
    [--task_name <name>] \\
    [--config_path <path>]     # accepted for parity, ignored (verl uses dapo_trainer.yaml) \\
    [--gpus <int>] \\
    [--clip_ratio_low <float>] [--clip_ratio_high <float>] \\
    [--disable_kl True|False] \\
    [--online_filtering True|False] [--filter_metric accuracy|overall|score|...] \\
    [--max_try_make_batch <int>] \\
    [--save_freq <int>] [--save_limit <int>] [--test_freq <int>] \\
    [--max_prompt_length <int>] [--max_response_length <int>] \\
    [--enable_dynamic_max_tokens True|False] \\
    [--penalty_max_length <int>] [--overlong_buffer_length <int>] \\
    [--actor_ulysses <int>] [--rollout_n <int>] [--rollout_tp <int>] \\
    [--train_batch_size <int>] [--gen_batch_size <int>] \\
    [--ppo_mini_batch_size <int>] \\
    [--total_epochs <int>] \\
    [--branch_adv_enabled True|False] \\
    [--branch_adv_alpha <float>] [--branch_adv_beta <float>] [--branch_adv_kappa <float>] \\
    [--branch_adv_warmup_start <int>] [--branch_adv_warmup_steps <int>] \\
    [--branch_adv_log_path <path>] \\
    [--reward_fn_path <path>] [--reward_fn_name <name>] [--reward_manager batch|naive|dapo|prime] \\
    [--enable_llm_judge True|False] \\
    [--llm_judge_host <ip>] [--llm_judge_ports <spec>] [--llm_judge_model <path>] \\
    [--llm_judge_endpoint <path>] [--llm_judge_timeout <sec>] \\
    [--llm_judge_max_retries <int>] [--llm_judge_max_workers <int>] [--llm_judge_max_tokens <int>] \\
    [--swanlab_project <name>] [--swanlab_mode cloud|local|offline] [--swanlab_api_key <key>]
EOF
}

# ==================== 3. Parse CLI ====================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path) MODEL_PATH="$2"; shift 2 ;;
    --train_data) TRAINDATA_PATH="$2"; shift 2 ;;
    --val_data) VALDATA_PATH="$2"; shift 2 ;;
    --task_name) TASK_NAME="$2"; shift 2 ;;
    --output_path) OUTPUT_PATH="$2"; shift 2 ;;
    --config_path) shift 2 ;;            # accepted for parity, ignored
    --gpus) NUM_GPUS="$2"; shift 2 ;;

    --clip_ratio_low) CLIP_RATIO_LOW="$2"; shift 2 ;;
    --clip_ratio_high) CLIP_RATIO_HIGH="$2"; shift 2 ;;
    --disable_kl) DISABLE_KL="$2"; shift 2 ;;
    --online_filtering) ONLINE_FILTERING="$2"; shift 2 ;;
    --filter_metric) FILTER_METRIC="$2"; shift 2 ;;
    --max_try_make_batch) MAX_TRY_MAKE_BATCH="$2"; shift 2 ;;

    --save_freq) SAVE_FREQ="$2"; shift 2 ;;
    --save_limit) SAVE_LIMIT="$2"; shift 2 ;;
    --test_freq) TEST_FREQ="$2"; shift 2 ;;

    --max_prompt_length) MAX_PROMPT_LENGTH="$2"; shift 2 ;;
    --max_response_length) MAX_RESPONSE_LENGTH="$2"; shift 2 ;;
    --enable_dynamic_max_tokens) ENABLE_DYNAMIC_MAX_TOKENS="$2"; shift 2 ;;
    --penalty_max_length) PENALTY_MAX_LENGTH="$2"; shift 2 ;;
    --overlong_buffer_length) OVERLONG_BUFFER_LENGTH="$2"; shift 2 ;;

    --actor_ulysses) ACTOR_ULYSSES="$2"; shift 2 ;;
    --critic_ulysses) shift 2 ;;        # accepted for parity, ignored
    --rollout_n) ROLLOUT_N="$2"; shift 2 ;;
    --rollout_tp) ROLLOUT_TP="$2"; shift 2 ;;
    --rollout_gpu_memory_utilization) ROLLOUT_GPU_MEM_UTIL="$2"; shift 2 ;;
    --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --gen_batch_size) GEN_BATCH_SIZE="$2"; GEN_BATCH_SIZE_EXPLICIT=1; shift 2 ;;
    --ppo_mini_batch_size) ACTOR_PPO_MINI_BATCH_SIZE="$2"; shift 2 ;;
    --total_epochs) TOTAL_EPOCHS="$2"; shift 2 ;;

    --branch_adv_enabled) BRANCH_ADV_ENABLED="$2"; shift 2 ;;
    --branch_adv_alpha) BRANCH_ADV_ALPHA="$2"; shift 2 ;;
    --branch_adv_beta) BRANCH_ADV_BETA="$2"; shift 2 ;;
    --branch_adv_kappa) BRANCH_ADV_KAPPA="$2"; shift 2 ;;
    --branch_adv_warmup_start) BRANCH_ADV_WARMUP_START="$2"; shift 2 ;;
    --branch_adv_warmup_steps) BRANCH_ADV_WARMUP_STEPS="$2"; shift 2 ;;
    --branch_adv_log_path) BRANCH_ADV_LOG_PATH="$2"; shift 2 ;;

    --reward_fn_path) REWARD_FN_PATH="$2"; shift 2 ;;
    --reward_fn_name) REWARD_FN_NAME="$2"; shift 2 ;;
    --reward_manager) REWARD_MANAGER="$2"; shift 2 ;;

    --enable_llm_judge) ENABLE_LLM_JUDGE="$2"; shift 2 ;;
    --llm_judge_host) LLM_JUDGE_HOST="$2"; shift 2 ;;
    --llm_judge_ports) LLM_JUDGE_PORTS="$2"; shift 2 ;;
    --llm_judge_model) LLM_JUDGE_MODEL="$2"; shift 2 ;;
    --llm_judge_endpoint) LLM_JUDGE_ENDPOINT="$2"; shift 2 ;;
    --llm_judge_timeout) LLM_JUDGE_TIMEOUT="$2"; shift 2 ;;
    --llm_judge_max_retries) LLM_JUDGE_MAX_RETRIES="$2"; shift 2 ;;
    --llm_judge_max_workers) LLM_JUDGE_MAX_WORKERS="$2"; shift 2 ;;
    --llm_judge_max_tokens) LLM_JUDGE_MAX_TOKENS="$2"; shift 2 ;;

    --swanlab_project) SWANLAB_PROJECT="$2"; shift 2 ;;
    --swanlab_mode) SWANLAB_MODE="$2"; shift 2 ;;
    --swanlab_api_key) SWANLAB_API_KEY="$2"; shift 2 ;;

    -h|--help) usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown arg: $1"
      usage
      exit 1
      ;;
  esac
done

export SWANLAB_API_KEY
export SWANLAB_PROJECT
export SWANLAB_MODE

# Re-derive GEN_BATCH_SIZE = TRAIN_BATCH_SIZE * 2 (verl official DAPO default)
# unless the user explicitly passed --gen_batch_size.
if [[ -z "${GEN_BATCH_SIZE_EXPLICIT:-}" ]]; then
  GEN_BATCH_SIZE=$((TRAIN_BATCH_SIZE * 2))
fi

# Export LLM judge env so reward_function/llm_judge.py picks them up at import.
export LLM_JUDGE_HOST
export LLM_JUDGE_PORTS
export LLM_JUDGE_MODEL
export LLM_JUDGE_ENDPOINT
export LLM_JUDGE_TIMEOUT
export LLM_JUDGE_MAX_RETRIES
export LLM_JUDGE_MAX_WORKERS
export LLM_JUDGE_MAX_TOKENS

TASK_NAME="${TASK_NAME:-dapo_branch_adv}"
if [[ -z "$MODEL_PATH" || -z "$TRAINDATA_PATH" || -z "$OUTPUT_PATH" ]]; then
  echo "[ERROR] Missing required args: --model_path --train_data --output_path"
  usage
  exit 1
fi

# ==================== 3.1 Derived: disable_kl → kl knobs ====================
if [[ "${DISABLE_KL,,}" == "true" ]]; then
  USE_KL_IN_REWARD=False
  USE_KL_LOSS=False
  KL_LOSS_COEF=0.0
  KL_COEF=0.0
else
  USE_KL_IN_REWARD=False
  USE_KL_LOSS=True
  KL_LOSS_COEF=1.0e-2
  KL_COEF=1.0e-2
fi

# ==================== 4. Ray cluster ====================
ray stop --force || true
if [[ "$target_node_rank" -eq 0 ]]; then
  ray start --head \
    --port="$ray_port" \
    --num-gpus="$NUM_GPUS" \
    --dashboard-host=0.0.0.0 \
    --disable-usage-stats
else
  ray start --address="$target_master_addr:$ray_port" \
    --num-gpus="$NUM_GPUS" \
    --disable-usage-stats \
    --block
  exit 0
fi

# ==================== 5. Output paths ====================
mkdir -p "${OUTPUT_PATH}"
EXPERIMENT_NAME="${TASK_NAME}_${TASK_ID}"
SAVE_DIR="${OUTPUT_PATH}/${EXPERIMENT_NAME}"
export TENSORBOARD_DIR="${SAVE_DIR}/tensorboard_log"
mkdir -p "${TENSORBOARD_DIR}"

LENGTH_LOG_PATH="${SAVE_DIR}/lengthlog48k"

# ==================== 6. Hydra args ====================
PY_ARGS=(
  algorithm.adv_estimator=grpo
  algorithm.use_kl_in_reward=${USE_KL_IN_REWARD}
  algorithm.kl_ctrl.kl_coef=${KL_COEF}
  algorithm.kl_penalty=low_var_kl

  # DAPO dynamic sampling (verl recipe.dapo.RayDAPOTrainer reads these)
  algorithm.filter_groups.enable=${ONLINE_FILTERING}
  algorithm.filter_groups.metric=${FILTER_METRIC}
  algorithm.filter_groups.max_num_gen_batches=${MAX_TRY_MAKE_BATCH}

  # branch_adv (EDPO)
  algorithm.branch_adv_enabled=${BRANCH_ADV_ENABLED}
  algorithm.branch_adv_alpha=${BRANCH_ADV_ALPHA}
  algorithm.branch_adv_beta=${BRANCH_ADV_BETA}
  algorithm.branch_adv_kappa=${BRANCH_ADV_KAPPA}
  algorithm.branch_adv_warmup_start=${BRANCH_ADV_WARMUP_START}
  algorithm.branch_adv_warmup_steps=${BRANCH_ADV_WARMUP_STEPS}
  algorithm.branch_adv_log_path=${BRANCH_ADV_LOG_PATH}

  # data
  data.train_files=${TRAINDATA_PATH}
  data.max_prompt_length=${MAX_PROMPT_LENGTH}
  data.max_response_length=${MAX_RESPONSE_LENGTH}
  data.train_batch_size=${TRAIN_BATCH_SIZE}
  data.gen_batch_size=${GEN_BATCH_SIZE}
  data.val_batch_size=${VAL_BATCH_SIZE}
  data.filter_overlong_prompts=True
  data.truncation=error
  data.shuffle=True
  +data.apply_chat_template_kwargs.enable_thinking=False

  # model
  actor_rollout_ref.model.path=${MODEL_PATH}
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True

  # actor
  actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
  actor_rollout_ref.actor.optim.weight_decay=${ACTOR_WEIGHT_DECAY}
  actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}
  actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}
  actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}
  actor_rollout_ref.actor.grad_clip=${ACTOR_GRAD_NORM}
  actor_rollout_ref.actor.ppo_mini_batch_size=${ACTOR_PPO_MINI_BATCH_SIZE}
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ACTOR_PPO_MICRO_BSZ_PER_GPU}
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=${ACTOR_ULYSSES}
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True

  # rollout (vLLM)
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.n=${ROLLOUT_N}
  actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
  actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
  actor_rollout_ref.rollout.enforce_eager=False
  actor_rollout_ref.rollout.enable_chunked_prefill=True
  actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.enable_dynamic_max_tokens=${ENABLE_DYNAMIC_MAX_TOKENS}
  actor_rollout_ref.rollout.length_log_path=${LENGTH_LOG_PATH}
  actor_rollout_ref.rollout.length_log_interval=1
  actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}
  actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}
  actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K}
  actor_rollout_ref.rollout.val_kwargs.n=${VAL_N}
  actor_rollout_ref.rollout.val_kwargs.do_sample=True

  # ref
  actor_rollout_ref.ref.fsdp_config.param_offload=True

  # reward — use our compute_score_batch (LLM judge fan-out, batch manager).
  # The DAPO recipe's default reward_manager=dapo is overridden here on
  # purpose — we want Easy-R1 reward parity, not DAPO-paper overlong buffer.
  reward.custom_reward_function.path=${REWARD_FN_PATH}
  reward.custom_reward_function.name=${REWARD_FN_NAME}
  reward.reward_manager.name=${REWARD_MANAGER}
  +reward.custom_reward_function.reward_kwargs.penalty_max_length=${PENALTY_MAX_LENGTH}
  +reward.custom_reward_function.reward_kwargs.overlong_buffer_length=${OVERLONG_BUFFER_LENGTH}
  +reward.custom_reward_function.reward_kwargs.enable_llm_judge=${ENABLE_LLM_JUDGE}
  # Make sure recipe/dapo's own overlong_buffer is OFF — our reward already
  # bakes the length penalty into 'score'.
  reward.reward_kwargs.overlong_buffer_cfg.enable=False

  # trainer
  trainer.experiment_name=${EXPERIMENT_NAME}
  trainer.project_name=${SWANLAB_PROJECT}
  trainer.logger='[swanlab]'
  trainer.n_gpus_per_node=${NUM_GPUS}
  trainer.nnodes=${target_nnodes}
  trainer.save_freq=${SAVE_FREQ}
  trainer.test_freq=${TEST_FREQ}
  trainer.max_actor_ckpt_to_keep=${SAVE_LIMIT}
  trainer.total_epochs=${TOTAL_EPOCHS}
  trainer.val_before_train=True
  trainer.default_local_dir=${SAVE_DIR}
)

if [[ -n "${VALDATA_PATH}" ]]; then
  PY_ARGS+=( "data.val_files=${VALDATA_PATH}" )
fi

echo "[INFO] Effective PY_ARGS:"
printf '  %s\n' "${PY_ARGS[@]}"

# ==================== 7. Launch ====================
# recipe/ is a git submodule (no __init__.py); `python -m recipe.dapo.main_dapo`
# discovers it as a namespace package only when cwd contains `recipe/`.
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${VERL_ROOT}"
python3 -m recipe.dapo.main_dapo \
  "${PY_ARGS[@]}"

ray stop --force || true
