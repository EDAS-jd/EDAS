# Copyright 2025 Individual Contributor: Mert Unsal
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

import inspect
from typing import Any

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase


@register("batch")
class BatchRewardManager(RewardManagerBase):
    """Batch reward manager for the experimental reward loop.

    The legacy ``verl.workers.reward_manager.batch`` manager calls a custom
    reward function with list-valued ``data_sources``, ``solution_strs``,
    ``ground_truths`` and ``extra_infos``. The experimental reward loop uses a
    different async manager API, so this class preserves that legacy contract.
    """

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_fn_key = config.data.get("reward_fn_key", "data_source")
        reward_fn_config = config.reward.get("custom_reward_function") or {}
        self.reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    async def run_single(self, data: DataProto) -> dict:
        outputs = await self.run_batch(data[-1:])
        return outputs[-1]

    async def run_batch(self, data: DataProto) -> list[dict]:
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompt_ids.shape[-1]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        responses_str = []
        for i in range(len(data)):
            valid_len = int(valid_response_lengths[i].item())
            valid_response_ids = response_ids[i][:valid_len]
            response_str = await self.loop.run_in_executor(
                None, lambda ids=valid_response_ids: self.tokenizer.decode(ids, skip_special_tokens=True)
            )
            responses_str.append(response_str)

        data_sources = list(data.non_tensor_batch[self.reward_fn_key])
        reward_models = list(data.non_tensor_batch["reward_model"])
        ground_truths = [item.get("ground_truth", None) if isinstance(item, dict) else None for item in reward_models]

        raw_extras = list(data.non_tensor_batch.get("extra_info", [{} for _ in range(len(data))]))
        rollout_reward_scores = list(data.non_tensor_batch.get("reward_scores", [{} for _ in range(len(data))]))
        extra_infos: list[dict[str, Any]] = []
        for extra, reward_scores in zip(raw_extras, rollout_reward_scores, strict=False):
            if isinstance(extra, dict):
                merged = dict(extra)
            elif hasattr(extra, "tolist") and isinstance(extra.tolist(), dict):
                merged = dict(extra.tolist())
            else:
                merged = {}
            merged["rollout_reward_scores"] = reward_scores
            extra_infos.append(merged)

        if self.is_async_reward_score:
            scores = await self.compute_score(
                data_sources=data_sources,
                solution_strs=responses_str,
                ground_truths=ground_truths,
                extra_infos=extra_infos,
                **self.reward_kwargs,
            )
        else:
            scores = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_sources=data_sources,
                    solution_strs=responses_str,
                    ground_truths=ground_truths,
                    extra_infos=extra_infos,
                    **self.reward_kwargs,
                ),
            )

        outputs = []
        for score in scores:
            reward_extra_info = {}
            if isinstance(score, dict):
                reward = score["score"]
                reward_extra_info.update(score)
            else:
                reward = score
                reward_extra_info["acc"] = score
            outputs.append({"reward_score": reward, "reward_extra_info": reward_extra_info})
        return outputs
