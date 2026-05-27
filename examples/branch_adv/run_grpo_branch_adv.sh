#!/bin/bash
# verl GRPO + branch_adv (EDPO) launcher.
# Mirrors the CLI surface of Easy-R1's run_multinode_1_02_20_2k_4k_grpo.sh
# so the same invocation works against the verl framework.

set -euo pipefail
set -x

# ==================== 0. SwanLab defaults ====================
python3 -c "import swanlab" 2>/dev/null || pip install -U swanlab
SWANLAB_API_KEY="${SWANLAB_API_KEY:-}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-verl-branch-adv}"
SWANLAB_MODE="${SWANLAB_MODE:-cloud}"

export VERL_DEBUG_PADDING="${VERL_DEBUG_PADDING:-1}"

# ==================== 1. Distributed env ====================
target_master_addr="${MASTER_ADDR:-127.0.0.1}"
target_master_port="${MASTER_PORT:-29500}"
target_nnodes="${WORLD_SIZE:-1}"
target_node_rank="${RANK:-0}"
ray_port="${RAY_PORT:-6379}"

echo "------------------------------------------------"
echo "verl (GRPO + branch_adv) Distributed Config"
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

CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.2
DISABLE_KL=False
SAVE_FREQ=5
TEST_FREQ=5

# Lengths
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=4096
ENABLE_DYNAMIC_MAX_TOKENS=True
PENALTY_MAX_LENGTH=50000
OVERLONG_BUFFER_LENGTH=0

# Branch-adv (EDPO) defaults
BRANCH_ADV_ENABLED=True
BRANCH_ADV_ALPHA=0.4
BRANCH_ADV_BETA=0.2
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

# Format/reward defaults — point at the new verl-side reward function & jinja.
# The jinja is informational only (verl uses HF chat_template natively); see
# branch_adv_README.md for how to bake the system prompt into the parquet.
REWARD_FN_PATH="${SCRIPT_DIR}/reward_function/math_no_format.py"
# Default to the batch entry point so LLM judge fan-out (Easy-R1 parity) works.
REWARD_FN_NAME="compute_score_batch"
REWARD_MANAGER="batch"

# LLM judge defaults — read by examples/branch_adv/reward_function/llm_judge.py
# at module load via os.getenv. These match Easy-R1's defaults verbatim.
LLM_JUDGE_HOST="${LLM_JUDGE_HOST:-10.119.97.103}"
LLM_JUDGE_PORTS="${LLM_JUDGE_PORTS:-9000-9007}"
LLM_JUDGE_MODEL="${LLM_JUDGE_MODEL:-/mnt/public/users/zhuyongfu/model/openai/gpt-oss-20b}"
LLM_JUDGE_ENDPOINT="${LLM_JUDGE_ENDPOINT:-/v1/chat/completions}"
LLM_JUDGE_TIMEOUT="${LLM_JUDGE_TIMEOUT:-30}"
LLM_JUDGE_MAX_RETRIES="${LLM_JUDGE_MAX_RETRIES:-2}"
LLM_JUDGE_MAX_WORKERS="${LLM_JUDGE_MAX_WORKERS:-512}"
LLM_JUDGE_MAX_TOKENS="${LLM_JUDGE_MAX_TOKENS:-1024}"
ENABLE_LLM_JUDGE="${ENABLE_LLM_JUDGE:-True}"

ROLLOUT_N=10
ROLLOUT_TP=2
ROLLOUT_GPU_MEM_UTIL=0.8

ACTOR_LR=1e-6
KL_LOSS_COEF=1.0e-2
USE_KL_LOSS=True

usage() {
  cat <<EOF
Usage:
  bash run_grpo_branch_adv.sh \\
    --model_path <path> \\
    --train_data <path> \\
    --output_path <path> \\
    [--val_data <path>] \\
    [--task_name <name>] \\
    [--gpus <int>] \\
    [--max_prompt_length <int>] \\
    [--max_response_length <int>] \\
    [--enable_dynamic_max_tokens True|False] \\
    [--penalty_max_length <int>] \\
    [--overlong_buffer_length <int>] \\
    [--branch_adv_enabled True|False] \\
    [--branch_adv_alpha <float>] \\
    [--branch_adv_beta <float>] \\
    [--branch_adv_kappa <float>] \\
    [--branch_adv_warmup_start <int>] \\
    [--branch_adv_warmup_steps <int>] \\
    [--branch_adv_log_path <path>] \\
    [--reward_fn_path <path>] \\
    [--reward_fn_name <name>] \\
    [--reward_manager naive|batch|dapo|prime] \\
    [--enable_llm_judge True|False] \\
    [--llm_judge_host <ip>] \\
    [--llm_judge_ports <spec>] \\
    [--llm_judge_model <path|name>] \\
    [--llm_judge_endpoint <path>] \\
    [--llm_judge_timeout <sec>] \\
    [--llm_judge_max_retries <int>] \\
    [--llm_judge_max_workers <int>] \\
    [--llm_judge_max_tokens <int>] \\
    [--rollout_n <int>] \\
    [--rollout_tp <int>] \\
    [--save_freq <int>] \\
    [--test_freq <int>] \\
    [--swanlab_project <name>] \\
    [--swanlab_mode cloud|local|offline] \\
    [--swanlab_api_key <key>]
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
    --config_path) shift 2 ;;            # accepted for parity, ignored (verl uses Hydra defaults)
    --gpus) NUM_GPUS="$2"; shift 2 ;;

    --clip_ratio_low) CLIP_RATIO_LOW="$2"; shift 2 ;;
    --clip_ratio_high) CLIP_RATIO_HIGH="$2"; shift 2 ;;
    --disable_kl) DISABLE_KL="$2"; shift 2 ;;
    --save_freq) SAVE_FREQ="$2"; shift 2 ;;
    --test_freq) TEST_FREQ="$2"; shift 2 ;;
    --save_limit) shift 2 ;;             # not directly mapped in verl; ignored

    --max_prompt_length) MAX_PROMPT_LENGTH="$2"; shift 2 ;;
    --max_response_length) MAX_RESPONSE_LENGTH="$2"; shift 2 ;;
    --enable_dynamic_max_tokens) ENABLE_DYNAMIC_MAX_TOKENS="$2"; shift 2 ;;
    --penalty_max_length) PENALTY_MAX_LENGTH="$2"; shift 2 ;;
    --overlong_buffer_length) OVERLONG_BUFFER_LENGTH="$2"; shift 2 ;;

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

    --rollout_n) ROLLOUT_N="$2"; shift 2 ;;
    --rollout_tp) ROLLOUT_TP="$2"; shift 2 ;;
    --rollout_gpu_memory_utilization) ROLLOUT_GPU_MEM_UTIL="$2"; shift 2 ;;

    --actor_lr) ACTOR_LR="$2"; shift 2 ;;
    --kl_loss_coef) KL_LOSS_COEF="$2"; shift 2 ;;
    --use_kl_loss) USE_KL_LOSS="$2"; shift 2 ;;

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

# Export LLM judge env so reward_function/llm_judge.py picks them up at import.
export LLM_JUDGE_HOST
export LLM_JUDGE_PORTS
export LLM_JUDGE_MODEL
export LLM_JUDGE_ENDPOINT
export LLM_JUDGE_TIMEOUT
export LLM_JUDGE_MAX_RETRIES
export LLM_JUDGE_MAX_WORKERS
export LLM_JUDGE_MAX_TOKENS

TASK_NAME="${TASK_NAME:-grpo_branch_adv}"
if [[ -z "$MODEL_PATH" || -z "$TRAINDATA_PATH" || -z "$OUTPUT_PATH" ]]; then
  echo "[ERROR] Missing required args: --model_path --train_data --output_path"
  usage
  exit 1
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
  algorithm.use_kl_in_reward=False
  algorithm.kl_penalty=low_var_kl

  # branch_adv
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
  data.train_batch_size=64
  data.filter_overlong_prompts=True
  data.truncation=error

  # model
  actor_rollout_ref.model.path=${MODEL_PATH}
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing=True

  # actor
  actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
  actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}
  actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}
  actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}
  actor_rollout_ref.actor.ppo_mini_batch_size=64
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True

  # rollout (vLLM)
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.n=${ROLLOUT_N}
  actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
  actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
  actor_rollout_ref.rollout.enforce_eager=False
  actor_rollout_ref.rollout.enable_chunked_prefill=True
  actor_rollout_ref.rollout.max_num_batched_tokens=32768
  actor_rollout_ref.rollout.enable_dynamic_max_tokens=${ENABLE_DYNAMIC_MAX_TOKENS}
  actor_rollout_ref.rollout.length_log_path=${LENGTH_LOG_PATH}
  actor_rollout_ref.rollout.length_log_interval=1

  # ref
  actor_rollout_ref.ref.fsdp_config.param_offload=True

  # reward
  reward.custom_reward_function.path=${REWARD_FN_PATH}
  reward.custom_reward_function.name=${REWARD_FN_NAME}
  reward.reward_manager.name=${REWARD_MANAGER}
  +reward.custom_reward_function.reward_kwargs.penalty_max_length=${PENALTY_MAX_LENGTH}
  +reward.custom_reward_function.reward_kwargs.overlong_buffer_length=${OVERLONG_BUFFER_LENGTH}
  +reward.custom_reward_function.reward_kwargs.enable_llm_judge=${ENABLE_LLM_JUDGE}

  # trainer
  trainer.experiment_name=${EXPERIMENT_NAME}
  trainer.project_name=${SWANLAB_PROJECT}
  trainer.logger='[swanlab]'
  trainer.n_gpus_per_node=${NUM_GPUS}
  trainer.nnodes=${target_nnodes}
  trainer.save_freq=${SAVE_FREQ}
  trainer.test_freq=${TEST_FREQ}
  trainer.default_local_dir=${SAVE_DIR}
)

if [[ -n "${VALDATA_PATH}" ]]; then
  PY_ARGS+=( "data.val_files=${VALDATA_PATH}" )
fi

echo "[INFO] Effective PY_ARGS:"
printf '  %s\n' "${PY_ARGS[@]}"

# ==================== 7. Launch ====================
python3 -m verl.trainer.main_ppo \
  "${PY_ARGS[@]}"

ray stop --force || true
