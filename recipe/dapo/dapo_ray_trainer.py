# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    _compute_branch_adv_warmup_scale,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import extract_reward
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip


def _write_length_log(
    log_path: str,
    step: int,
    prompt_lengths: list[int],
    response_lengths: list[int],
    pad_len: int,
) -> None:
    """Easy-R1 vllm_rollout_spmd.py:355-368 equivalent. Append one line per fire."""
    if not prompt_lengths or not response_lengths:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
        prompt_mean = sum(prompt_lengths) / len(prompt_lengths)
        response_mean = sum(response_lengths) / len(response_lengths)
        totals = [p + r for p, r in zip(prompt_lengths, response_lengths)]
        total_mean = sum(totals) / len(totals)
        message = (
            f"[len][{step}] prompt(min/mean/max) "
            f"{min(prompt_lengths)}/{prompt_mean:.1f}/{max(prompt_lengths)} | "
            f"response(min/mean/max) {min(response_lengths)}/{response_mean:.1f}/{max(response_lengths)} | "
            f"total(min/mean/max) {min(totals)}/{total_mean:.1f}/{max(totals)} "
            f"(pad_len={pad_len})\n"
        )
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(message)
    except Exception:
        # length logging is best-effort; never crash training because of it.
        pass


def _compute_passrate_metrics(
    uid_list,
    accuracies,
    rollout_n: int,
    verify_types=None,
) -> dict[str, float]:
    """Easy-R1 ray_trainer passrate: per-step distribution of how many of the
    n rollouts got the prompt right. Returns a dict
    {"passrate/0": .., "passrate/10": .., ..., "passrate/100": ..} where the
    bucket key is the percentage of correct rollouts in the group.
    """
    from collections import defaultdict

    if uid_list is None or accuracies is None:
        return {}
    if verify_types is None:
        verify_types = [None] * len(uid_list)

    uid2acc: dict = defaultdict(list)
    for uid, acc, vtype in zip(uid_list, accuracies, verify_types):
        if vtype is not None and str(vtype).lower().strip() == "bbox":
            continue
        uid2acc[uid].append(float(acc))
    if not uid2acc:
        return {}

    bucket_counts = {k: 0 for k in range(rollout_n + 1)}
    for accs in uid2acc.values():
        if not accs:
            continue
        correct = sum(1 for v in accs if v == 1.0)
        correct = max(0, min(correct, rollout_n))
        bucket_counts[correct] += 1

    total = sum(bucket_counts.values())
    if total == 0:
        return {}

    out: dict[str, float] = {}
    ordered_k = [0, rollout_n] + list(range(1, rollout_n))
    for k in ordered_k:
        pct = int(round((k / rollout_n) * 100)) if rollout_n > 0 else 0
        out[f"passrate/{pct}"] = bucket_counts[k] / total
    return out


def _log_group_error_stats(
    log_path: str,
    global_step: int,
    uid_list,
    accuracies,
    answer_preds,
    ground_truths,
    verify_types=None,
) -> None:
    """Easy-R1 ray_trainer._log_group_error_stats verbatim port.

    Each call appends one JSON line describing per-group incorrect-answer
    distribution: mode_collapse detection, entropy, top wrong answer.
    """
    import json
    import math
    from collections import Counter

    if not log_path or uid_list is None or accuracies is None:
        return
    try:
        if hasattr(ground_truths, "tolist"):
            gt_list = ground_truths.tolist()
        else:
            gt_list = list(ground_truths)
        if not answer_preds:
            answer_preds = [""] * len(uid_list)
        if verify_types is None:
            verify_types = [None] * len(uid_list)

        # Easy-R1's ground_truths column is a flat list of strings; verl puts it
        # inside reward_model as a dict {ground_truth: ...}. Normalize either form.
        normalized_gt: list[str] = []
        for g in gt_list:
            if isinstance(g, dict):
                normalized_gt.append(str(g.get("ground_truth", "")))
            else:
                normalized_gt.append("" if g is None else str(g))

        uid2data: dict = {}
        for uid, acc, pred, gt, vtype in zip(
            uid_list, accuracies, answer_preds, normalized_gt, verify_types
        ):
            if vtype is not None and str(vtype).lower().strip() == "bbox":
                continue
            uid = str(uid)
            if uid not in uid2data:
                uid2data[uid] = {"ground_truth": gt, "entries": []}
            uid2data[uid]["entries"].append(
                {"accuracy": float(acc), "answer_pred": str(pred) if pred is not None else ""}
            )

        groups = []
        n_mode_collapse = 0
        n_all_correct = 0
        for uid, data in uid2data.items():
            entries = data["entries"]
            n_total = len(entries)
            incorrect = [e["answer_pred"] for e in entries if e["accuracy"] < 1.0]
            n_wrong = len(incorrect)

            counts = Counter(ans.strip().lower() for ans in incorrect)
            num_unique = len(counts)

            entropy = 0.0
            if n_wrong > 0:
                for cnt in counts.values():
                    p = cnt / n_wrong
                    if p > 0:
                        entropy -= p * math.log(p)

            top_wrong, top_count = counts.most_common(1)[0] if counts else ("", 0)
            mode_collapse = n_wrong > 1 and num_unique == 1
            if mode_collapse:
                n_mode_collapse += 1
            if n_wrong == 0:
                n_all_correct += 1

            groups.append(
                {
                    "uid": uid,
                    "ground_truth": data["ground_truth"],
                    "num_total": n_total,
                    "num_correct": n_total - n_wrong,
                    "num_incorrect": n_wrong,
                    "incorrect_answer_counts": dict(counts.most_common()),
                    "num_unique_incorrect": num_unique,
                    "mode_collapse": mode_collapse,
                    "entropy": round(entropy, 4),
                    "top_wrong_answer": top_wrong,
                    "top_wrong_count": top_count,
                }
            )

        record = {
            "global_step": global_step,
            "num_groups": len(groups),
            "num_groups_all_correct": n_all_correct,
            "num_groups_mode_collapse": n_mode_collapse,
            "groups": groups,
        }
        os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Best-effort; never crash training because of debug logging.
        pass


class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def compute_kl_related_metrics(self, batch: DataProto, metrics: dict, timing_raw: dict):
        batch.batch["response_mask"] = compute_response_mask(batch)

        # recompute old_log_probs
        with marked_timer("old_log_prob", timing_raw, "blue"):
            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
            entropys = old_log_prob.batch["entropys"]
            response_masks = batch.batch["response_mask"]
            actor_config = self.config.actor_rollout_ref.actor
            entropy_agg = agg_loss(
                loss_mat=entropys,
                loss_mask=response_masks,
                loss_agg_mode=actor_config.loss_agg_mode,
                loss_scale_factor=actor_config.loss_scale_factor,
            )
            old_log_prob_metrics = {
                "actor/entropy": entropy_agg.detach().item(),
                "perf/mfu/actor_infer": old_log_prob_mfu,
            }
            metrics.update(old_log_prob_metrics)
            old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

        if self.use_reference_policy:
            # compute reference log_prob
            with marked_timer("ref", timing_raw, "olive"):
                ref_log_prob = self._compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        return batch

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0
        self.max_steps_duration = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        current_epoch = self.global_steps // len(self.train_dataloader)

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                new_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                num_gen_batches += 1
                gen_batch = self._get_gen_batch(new_batch)
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    # ── Easy-R1 vllm_rollout_spmd length log (best-effort) ──
                    # Honors actor_rollout_ref.rollout.length_log_path /
                    # length_log_interval; writes one line per `interval` steps
                    # with prompt/response/total min/mean/max + pad_len.
                    rollout_cfg = self.config.actor_rollout_ref.rollout
                    length_log_path = rollout_cfg.get("length_log_path", None)
                    length_log_interval = int(rollout_cfg.get("length_log_interval", 0) or 0)
                    if length_log_path and (
                        length_log_interval <= 0 or self.global_steps % length_log_interval == 0
                    ):
                        try:
                            attn = gen_batch_output.batch.get("attention_mask")
                            resp = gen_batch_output.batch.get("responses")
                            if attn is not None and resp is not None:
                                resp_len = resp.size(1)
                                prompt_lens = attn[:, :-resp_len].sum(dim=-1).tolist()
                                response_lens = attn[:, -resp_len:].sum(dim=-1).tolist()
                                _write_length_log(
                                    log_path=length_log_path,
                                    step=int(self.global_steps),
                                    prompt_lengths=[int(x) for x in prompt_lens],
                                    response_lengths=[int(x) for x in response_lens],
                                    pad_len=int(attn.size(1)),
                                )
                        except Exception:
                            pass

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            # compute reward model score on new_batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                                rm_scores = self._compute_reward_colocate(new_batch)
                                new_batch = new_batch.union(rm_scores)
                            reward_baseline_tensor, _ = extract_reward(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            new_batch.pop(batch_keys=list(keys_to_pop))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    if self.config.algorithm.use_kl_in_reward:
                        # We need these metrics for apply_kl_penalty if using kl in reward
                        new_batch = self.compute_kl_related_metrics(new_batch, metrics, timing_raw)
                        # otherwise, we will compute those after dynamic sampling

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                            # we first compute reward model score
                            batch_reward = self._compute_reward_colocate(new_batch)
                            new_batch = new_batch.union(batch_reward)

                        # we combine with rule-based rm
                        reward_tensor, reward_extra_infos_dict = extract_reward(new_batch)

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # ── Easy-R1 EDPO debug records: passrate + group_error_stats ──
                        # Both are best-effort, fire whenever `accuracy` is in the reward
                        # extras (mirroring Easy-R1 ray_trainer.py:957-1033, which runs
                        # them regardless of branch_adv_enabled).
                        _acc_arr = reward_extra_infos_dict.get("accuracy") if reward_extra_infos_dict else None
                        _ans_arr = reward_extra_infos_dict.get("answer_pred") if reward_extra_infos_dict else None
                        if _acc_arr is not None and "uid" in new_batch.non_tensor_batch:
                            _uid_list = new_batch.non_tensor_batch["uid"]
                            _verify_types = new_batch.non_tensor_batch.get("verify_type")
                            _rollout_n = int(self.config.actor_rollout_ref.rollout.n)
                            passrate_metrics = _compute_passrate_metrics(
                                uid_list=_uid_list,
                                accuracies=_acc_arr,
                                rollout_n=_rollout_n,
                                verify_types=_verify_types,
                            )
                            metrics.update(passrate_metrics)

                            # group_error_stats.jsonl — Easy-R1 wrote it next to the
                            # checkpoint; mirror that.
                            ges_path = os.path.join(
                                self.config.trainer.get("default_local_dir", "checkpoints"),
                                "group_error_stats.jsonl",
                            )
                            _gt_col = new_batch.non_tensor_batch.get("reward_model")
                            if _gt_col is None:
                                _gt_col = new_batch.non_tensor_batch.get("ground_truth")
                            _log_group_error_stats(
                                log_path=ges_path,
                                global_step=int(self.global_steps),
                                uid_list=_uid_list,
                                accuracies=_acc_arr,
                                answer_preds=_ans_arr if _ans_arr is not None else [""] * len(_uid_list),
                                ground_truths=_gt_col if _gt_col is not None else [""] * len(_uid_list),
                                verify_types=_verify_types,
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)

                        new_batch = new_batch[kept_traj_idxs]
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                self.gen_steps += 1
                                is_last_step = self.global_steps >= self.total_training_steps
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    self.checkpoint_manager.sleep_replicas()

                    # === Updating ===
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    if not self.config.algorithm.use_kl_in_reward:
                        batch = self.compute_kl_related_metrics(batch, metrics, timing_raw)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    # Compute rollout correction weights and off-policy metrics (inherited from RayPPOTrainer)
                    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    if rollout_corr_config is not None and "rollout_log_probs" in batch.batch:
                        batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                        # IS and off-policy metrics already have rollout_corr/ prefix
                        metrics.update(is_metrics)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

                        # ── branch_adv (EDPO) ──────────────────────────────
                        # Mirror of verl.trainer.ppo.ray_trainer.fit() so that DAPO recipe
                        # also honors algorithm.branch_adv_* when the user has it enabled.
                        branch_adv_kwargs = None
                        algo_cfg = self.config.algorithm
                        if algo_cfg.get("branch_adv_enabled", False):
                            acc_key = algo_cfg.get("branch_adv_acc_key", "accuracy")
                            ans_key = algo_cfg.get("branch_adv_answer_key", "answer_pred")
                            fmt_key = algo_cfg.get("branch_adv_format_key", "format")
                            acc_arr = batch.non_tensor_batch.get(acc_key)
                            ans_arr = batch.non_tensor_batch.get(ans_key)
                            fmt_arr = batch.non_tensor_batch.get(fmt_key)
                            if acc_arr is not None and ans_arr is not None:
                                branch_scale = _compute_branch_adv_warmup_scale(
                                    self.global_steps,
                                    int(algo_cfg.get("branch_adv_warmup_start", 0)),
                                    int(algo_cfg.get("branch_adv_warmup_steps", 0)),
                                )
                                if branch_scale > 0.0:
                                    branch_log_path = algo_cfg.get("branch_adv_log_path", None)
                                    full_log_path = None
                                    if branch_log_path:
                                        local_dir = self.config.trainer.get(
                                            "default_local_dir", "checkpoints"
                                        )
                                        full_log_path = os.path.join(local_dir, branch_log_path)
                                    branch_adv_out_metrics: dict = {}
                                    branch_adv_kwargs = {
                                        "acc_list": [float(x) for x in list(acc_arr)],
                                        "answer_list": [str(x) for x in list(ans_arr)],
                                        "format_list": (
                                            [float(x) for x in list(fmt_arr)] if fmt_arr is not None else None
                                        ),
                                        "branch_adv_enabled": True,
                                        "branch_adv_alpha": float(
                                            algo_cfg.get("branch_adv_alpha", 0.2)
                                        )
                                        * branch_scale,
                                        "branch_adv_beta": float(
                                            algo_cfg.get("branch_adv_beta", 0.2)
                                        )
                                        * branch_scale,
                                        "branch_adv_kappa": float(
                                            algo_cfg.get("branch_adv_kappa", 2.0)
                                        ),
                                        "branch_adv_log_path": full_log_path,
                                        "branch_adv_step": int(self.global_steps),
                                        "branch_adv_out_metrics": branch_adv_out_metrics,
                                    }
                                    metrics["branch_adv/scale"] = branch_scale

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                            branch_adv_kwargs=branch_adv_kwargs,
                        )

                        if branch_adv_kwargs is not None:
                            for k, v in branch_adv_kwargs["branch_adv_out_metrics"].items():
                                metrics[f"branch_adv/{k}"] = v

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self._update_actor(batch)

                        # Check if ESI/training plan is close to expiration
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, "green"):
                                self._save_checkpoint()

                        with marked_timer("update_weights", timing_raw, "red"):
                            self.checkpoint_manager.update_weights()
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw.get("step", 0)
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
        # check if last step checkpint exists
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            # save last step checkpoint
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)
