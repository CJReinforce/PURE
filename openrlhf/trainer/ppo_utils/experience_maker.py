import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import ray
import torch
import torch.nn as nn
from tqdm import tqdm

from openrlhf.models.actor import Actor
from openrlhf.models.utils import (
    compute_approx_kl,
    compute_reward,
    masked_mean,
    unpacking_samples,
)
from openrlhf.utils import (
    compute_score,
    right_padding,
    token_log_prob_to_step_log_prob,
    turn_process_reward_logits_to_reward,
)
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn, remote_rm_fn_ray

logger = init_logger(__name__)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advantages: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    kl: Optional[torch.Tensor] = None

    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        self.values = to(self.values, device)
        self.attention_mask = to(self.attention_mask, device)
        self.action_mask = to(self.action_mask, device)
        self.kl = to(self.kl, device)
        self.info = {key: to(value, device) for key, value in self.info.items()}
        return self

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        self.values = pin_memory(self.values)
        self.attention_mask = pin_memory(self.attention_mask)
        self.action_mask = pin_memory(self.action_mask)
        self.kl = pin_memory(self.kl)
        self.info = {key: pin_memory(value) for key, value in self.info.items()}
        return self


@dataclass
class Samples:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
    num_actions: int or (B,), the number of actions (tokens) in the response.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    score_ids: Optional[torch.LongTensor]
    score_mask: Optional[torch.BoolTensor]
    reward_mask: Optional[torch.BoolTensor]
    max_num_steps: Union[int, torch.Tensor]
    num_steps: Optional[torch.Tensor]
    solved: Optional[torch.Tensor]
    have_answers: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    problem_length: torch.Tensor
    response_length: torch.Tensor
    total_length: torch.Tensor


class NaiveExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: str = None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.perf_stats = None
        self.advantage_estimator = strategy.args.advantage_estimator
        self.step_separator = '\n\n'
        self.step_separator_tokens = []
        for i in range(len(self.tokenizer)):
            if self.tokenizer.decode(i).endswith(self.step_separator):
                self.step_separator_tokens.append(i)
        self.step_separator_tokens = torch.LongTensor(
            self.step_separator_tokens, 
        ).to(device=torch.cuda.current_device())
        self.prm_step_separator = '\n'
        self.prm_step_separator_token = self.tokenizer.encode(
            self.prm_step_separator, 
            return_tensors='pt'
        ).squeeze(0).to(device=torch.cuda.current_device())

    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience_list(self, all_data: Union[str, List[str]], **generate_kwargs) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        # generate responses, problem -> sample
        samples_list = self.generate_samples(all_data, **generate_kwargs)
        torch.distributed.barrier()

        # sample -> experience, calculate reward, kl
        experiences = []
        for samples in tqdm(
            samples_list,
            desc="make_experience",
            disable=True,  # not self.strategy.is_rank_0(),
        ):
            experiences.append(self.make_experience(samples).to_device("cpu"))

        # experience -> experience, calculate reward - baseline
        experiences, rewards = self.process_experiences(experiences)

        for experience, reward in zip(experiences, rewards):
            experience = experience.to_device("cuda")
            reward = reward.to(device="cuda")
            # calculate reward - beta * kl
            reward = reward - self.kl_ctl.value * experience.kl

            # calculate advantage and return
            if self.advantage_estimator == "gae":
                raise NotImplementedError
                # experience.advantages, experience.returns = self.get_advantages_and_returns(
                #     experience.values,
                #     reward,
                #     experience.action_mask,
                #     generate_kwargs["gamma"],
                #     generate_kwargs["lambd"],
                # )
            elif self.advantage_estimator in ["reinforce", "rloo"]:
                returns = reward.fliplr().cumsum(dim=1).fliplr()
                # 0 adv (no loss) for step separator tokens
                returns.masked_fill_(experience.info["process_reward_mask"], 0)
                experience.returns = returns
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            if not getattr(self, "packing_samples", False):
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = reward[:, 0]
            experience.info["return"] = return_sums

            # remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]
            del experience.info["max_num_steps"]
            del experience.info["score_mask"]
            del experience.info["num_tokens_per_step"]
            experience.to_device("cpu")
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_data, **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.
        """
        assert not getattr(self, "packing_samples", False)
        args = self.strategy.args
        self.actor.eval()

        all_prompts, all_answers, have_answers = all_data['prompt'], all_data['answer'], all_data['have_answer']
        # sample multiple response
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_answers = sum([[answer] * args.n_samples_per_prompt for answer in all_answers], [])
        have_answers = sum([[have_answer] * args.n_samples_per_prompt for have_answer in have_answers], [])
        samples_list = []
        for i in tqdm(
            range(0, len(all_prompts), args.micro_rollout_batch_size),
            desc=f"rollout, rank {self.strategy.get_rank()}",
            disable=False,  # not self.strategy.is_rank_0(),
        ):
            prompts = all_prompts[i : i + args.micro_rollout_batch_size]
            answers = all_answers[i : i + args.micro_rollout_batch_size]
            have_ans = have_answers[i : i + args.micro_rollout_batch_size]
            inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
            sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)

            num_actions = action_mask.size(1)
            bs, problem_length = inputs["input_ids"].size()
            
            solution_tokens = sequences[:,-num_actions:]
            # find step separator, typically '\n\n'
            row_ids, column_ids = torch.where(
                torch.isin(solution_tokens, self.step_separator_tokens)
            )
            # +1 for the last step with eos rather than step separator
            max_num_steps = max([column_ids[row_ids==i].numel() for i in range(bs)]) + 1
            # start to end index of each step
            score_ids = torch.full((bs, max_num_steps), -1, dtype=torch.long, device=sequences.device)
            # whether the end of step
            reward_mask = torch.zeros_like(solution_tokens, dtype=torch.bool)
            eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(1)
            solved = []
            for j in range(bs):
                num_steps = column_ids[row_ids==j].numel()
                score_ids[j, :num_steps] = column_ids[row_ids==j] + 1
                reward_mask[j, column_ids[row_ids==j]] = True
                score_ids[j, num_steps] = eos_indices[j] + 1
                reward_mask[j, eos_indices[j]] = True

                if answers[j] != 'null':
                    solved.append(compute_score(
                        solution_str=self.tokenizer.decode(
                            solution_tokens[j], 
                            skip_special_tokens=True,
                        ), 
                        ground_truth=answers[j],
                    ))
                else:
                    # later, verifiable reward = solved - 1
                    # for the case of no answer, VR = 0
                    solved.append(1)
            
            score_mask = score_ids != -1
            solved = torch.tensor(solved, device=sequences.device)
            have_ans = torch.tensor(have_ans, device=sequences.device, dtype=torch.bool)

            samples = Samples(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                score_ids=score_ids,
                score_mask=score_mask,
                reward_mask=reward_mask,
                max_num_steps=max_num_steps,
                num_steps=score_mask.float().sum(dim=-1),
                num_actions=num_actions,
                solved=solved,
                have_answers=have_ans,
                packed_seq_lens=None,
                problem_length=problem_length,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
            )
            samples_list.append(samples)
        return samples_list

    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        self.actor.eval()
        self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()
        if self.critic is not None:
            self.critic.eval()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        problem_length = samples.problem_length
        score_ids = samples.score_ids
        score_mask = samples.score_mask
        bs = sequences.size(0)

        # log probs
        token_log_probs = self.actor(sequences, num_actions, attention_mask)

        # init log probs
        base_token_log_probs = self.initial_model(sequences, num_actions, attention_mask)

        # values
        if self.critic is not None:
            value = self.critic(sequences, num_actions, attention_mask)
        else:
            value = None

        # rewards
        if self.remote_rm_url is not None:
            assert NotImplementedError
            # remote RM
            # queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            # r = remote_rm_fn(self.remote_rm_url, queries=queries).to(device=action_log_probs.device)
        else:
            # local RM
            problem_ids = sequences[:, :problem_length]
            problem_attn_mask = attention_mask[:, :problem_length]
            solution_tokens = sequences[:,-num_actions:]
            solution_attn_mask = attention_mask[:,-num_actions:]

            # reconstruct input_ids
            # add prm_step_separator_token for each step for predicting process reward
            input_ids = []
            attn_mask = []
            num_tokens_per_step = []
            # drop '\n\n' at the end of each step, then add '\n'
            for i in range(bs):
                input_ids_ = problem_ids[i]
                attn_mask_ = problem_attn_mask[i]
                num_tokens_per_step_ = []
                # split tokens of each step
                for idx, j in enumerate(score_ids[i][score_mask[i]]):
                    if idx == 0:
                        step_tokens = solution_tokens[i, :j-1]
                        step_attn_mask = solution_attn_mask[i, :j-1]
                        num_tokens_per_step_.append(j-1)
                    else:
                        step_tokens = solution_tokens[i, score_ids[i, idx-1]:j-1]
                        step_attn_mask = solution_attn_mask[i, score_ids[i, idx-1]:j-1]
                        num_tokens_per_step_.append(j - score_ids[i, idx-1] - 1)
                    # add separator token for each step
                    input_ids_ = torch.cat(
                        [input_ids_, step_tokens, self.prm_step_separator_token]
                    )
                    attn_mask_ = torch.cat(
                        [attn_mask_, step_attn_mask, torch.ones(
                            1, device=sequences.device, dtype=attention_mask.dtype
                        )]
                    )
                input_ids.append(input_ids_)
                attn_mask.append(attn_mask_)
                num_tokens_per_step.append(
                    torch.tensor(num_tokens_per_step_, device=sequences.device),
                )
            input_ids = right_padding(input_ids, self.tokenizer.pad_token_id)
            attn_mask = right_padding(attn_mask, 0)
            num_tokens_per_step = right_padding(num_tokens_per_step, 0)

            pred_rew_ids = (score_ids - 1 + problem_length).masked_fill(
                ~score_mask, -1
            )
            
            # process reward for every tokens
            # only non-zero for the step separator tokens
            process_reward_matrix = torch.zeros_like(
                token_log_probs, 
                device=token_log_probs.device, 
                dtype=token_log_probs.dtype,
            )
            if "PRM" in self.strategy.args.reward_mode:
                process_reward = self.reward_model(input_ids, attn_mask)
                # step-level process reward
                process_reward = torch.stack(
                    [process_reward[i, pred_rew_ids[i]] for i in range(bs)]
                )
                # weighted sum of step-level process reward (approx. min)
                process_reward = turn_process_reward_logits_to_reward(
                    process_reward, score_mask, 
                    return_outcome_reward=False, 
                    temperature=0.1,
                    disable_weighted_reward=self.strategy.args.disable_weighted_reward,
                )
                process_reward_matrix[samples.reward_mask] = process_reward[
                    score_mask].to(dtype=token_log_probs.dtype)

        kl = compute_approx_kl(
            token_log_probs,
            base_token_log_probs,
            action_mask=action_mask,
            use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
        )

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "process_reward": process_reward_matrix,
            "process_reward_mask": samples.reward_mask,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
            "max_num_steps": samples.max_num_steps,
            "num_steps": samples.num_steps,
            "solved": samples.solved,
            "have_answers": samples.have_answers,
            "num_tokens_per_step": num_tokens_per_step,
            "score_mask": score_mask,
        }
        # reset model state
        self.actor.train()
        if self.critic is not None:
            self.critic.train()

        return Experience(
            sequences,
            token_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )

    @torch.no_grad()
    def process_experiences(self, experiences: List[Experience]) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.

        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        args = self.strategy.args
        assert args.advantage_estimator == "rloo"

        # micro_rollout_batch_size must be a multiple of n_samples_per_prompt
        # calculate: PR - PR_baseline + coef * (VR - VR_baseline)
        process_rewards = []
        for experience in experiences:
            # process reward
            process_reward = experience.info["process_reward"].clone()
            original_shape = process_reward.shape
            process_reward = process_reward.reshape(
                -1, args.n_samples_per_prompt, process_reward.size(1),
            )
            outcome_reward = process_reward.sum(-1)

            # check if outcome reward mathces the ground truth
            experience.info["reward"] = outcome_reward.flatten()
            # outcome_reward > 0 means prediction is correct
            # outcome_reward <= 0 means prediction is wrong
            prediction = experience.info["reward"].clone().sign()
            prediction[prediction == -1] = 0

            have_answers = experience.info["have_answers"]
            solved = experience.info["solved"]
            match = torch.full_like(solved, -1.0, dtype=torch.float)
            match[have_answers] = (
                prediction[have_answers] == solved[have_answers]
            ).float()
            experience.info["match"] = match

            reward_mask = experience.info["process_reward_mask"]
            # average reward per step as baseline
            if args.reward_baseline == "step":
                num_steps = experience.info["num_steps"]
                num_steps = num_steps.reshape(-1, args.n_samples_per_prompt)
                reward_mask = reward_mask.reshape(
                    -1, args.n_samples_per_prompt, reward_mask.size(1),
                )

                average_process_reward = outcome_reward / num_steps
                process_reward_baseline = (
                    average_process_reward.sum(-1, keepdim=True) - average_process_reward
                ) / (args.n_samples_per_prompt - 1)

                process_reward = (
                    process_reward - process_reward_baseline.unsqueeze(-1)
                ).masked_fill(~reward_mask, 0)
                process_reward = process_reward.reshape(original_shape)
            # (avg. reward per token) * (num of tokens of corresponding step) as baseline
            else:
                response_length = experience.info["response_length"]
                response_length = response_length.reshape(-1, args.n_samples_per_prompt)
                average_reward_per_token = (
                    outcome_reward.sum(-1, keepdim=True) - outcome_reward
                ) / (
                    response_length.sum(-1, keepdim=True) - response_length
                )
                
                process_reward = process_reward.reshape(original_shape)
                average_reward_per_token = average_reward_per_token.flatten()
                num_tokens_per_step = experience.info["num_tokens_per_step"]
                score_mask = experience.info["score_mask"]
                bs = process_reward.size(0)

                for i in range(bs):
                    process_reward[
                        i, reward_mask[i]
                    ] -= average_reward_per_token[i] * num_tokens_per_step[
                        i, score_mask[i]
                    ]

            if "VR" in args.reward_mode:
                # verifier reward
                verifier_reward = experience.info["solved"] - 1  # in {-1, 0}
                verifier_reward = verifier_reward.reshape(-1, args.n_samples_per_prompt)
                verifier_reward_baseline = (
                    verifier_reward.sum(-1, keepdim=True) - verifier_reward
                ) / (args.n_samples_per_prompt - 1)
                verifier_reward = verifier_reward - verifier_reward_baseline
                verifier_reward = verifier_reward.flatten()

                # add verifier reward to the last step
                action_mask = experience.action_mask
                eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(1)
                process_reward[
                    torch.arange(process_reward.size(0)), 
                    eos_indices
                ] += verifier_reward * args.verifiable_reward_coef
            
            process_rewards.append(process_reward)
        return experiences, process_rewards


class RemoteExperienceMaker(NaiveExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, packing_samples=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.packing_samples = packing_samples
        self.prm_step_separator_token = self.prm_step_separator_token.to("cpu")

    @torch.no_grad()
    def make_experience_list(self, all_data: Union[str, List[str]], **generate_kwargs) -> List[Experience]:
        if self.strategy.args.perf:
            self.perf_stats = {
                "generate_time": 0,
                "actor_value_rm_time": 0,
                "wait_time": 0,
            }
        experiences = super().make_experience_list(all_data, **generate_kwargs)
        if self.critic is not None:
            for experience in experiences:
                # send experience to critic
                experience_cpu = deepcopy(experience)
                experience_cpu.to_device("cpu")
                self._ref = self.critic.append.remote(experience_cpu)
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_data, **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        if self.vllm_engines is None:
            return super().generate_samples(all_prompts, **generate_kwargs)

        return self._generate_vllm(all_data, **generate_kwargs)

    @torch.no_grad()
    def make_experience(self, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        self.actor.eval()
        device = torch.cuda.current_device()
        assert not isinstance(self.reward_model, List)

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        packed_seq_lens = samples.packed_seq_lens
        problem_length = samples.problem_length
        score_ids = samples.score_ids
        score_mask = samples.score_mask
        bs = sequences.size(0)

        start = time.time()
        sequences_cpu, attention_mask_cpu = (
            sequences.to("cpu"),
            attention_mask.to("cpu"),
        )

        # init log probs
        base_action_log_probs_ref = self.initial_model.forward.remote(
            sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
        )

        # values
        if self.critic is not None:
            value_ref = self.critic.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
            )
            # avoid CUDA OOM when colocate models
            if self.strategy.args.colocate_critic_reward:
                ray.get([value_ref])
                ray.get([self.critic.empty_cache.remote()])
        else:
            value_ref = ray.put(None)

        if getattr(self.strategy.args, 'colocate_actor_ref', True):
            ray.get([base_action_log_probs_ref])
            ray.get([self.initial_model.empty_cache.remote()])

        # support remote RM API with ray
        if not self.remote_rm_url:
            problem_ids = sequences_cpu[:, :problem_length]
            problem_attn_mask = attention_mask_cpu[:, :problem_length]
            solution_tokens = sequences_cpu[:,-num_actions:]
            solution_attn_mask = attention_mask_cpu[:,-num_actions:]

            # reconstruct input_ids
            # add prm_step_separator_token for each step for predicting process reward
            input_ids = []
            attn_mask = []
            num_tokens_per_step = []
            # drop '\n\n' at the end of each step, then add '\n'
            for i in range(bs):
                input_ids_ = problem_ids[i]
                attn_mask_ = problem_attn_mask[i]
                num_tokens_per_step_ = []
                # split tokens of each step
                for idx, j in enumerate(score_ids[i][score_mask[i]]):
                    if idx == 0:
                        step_tokens = solution_tokens[i, :j-1]
                        step_attn_mask = solution_attn_mask[i, :j-1]
                        num_tokens_per_step_.append(j-1)
                    else:
                        step_tokens = solution_tokens[i, score_ids[i, idx-1]:j-1]
                        step_attn_mask = solution_attn_mask[i, score_ids[i, idx-1]:j-1]
                        num_tokens_per_step_.append(j - score_ids[i, idx-1] - 1)
                    # add separator token for each step
                    input_ids_ = torch.cat(
                        [input_ids_, step_tokens, self.prm_step_separator_token]
                    )
                    attn_mask_ = torch.cat(
                        [attn_mask_, step_attn_mask, torch.ones(
                            1, device=attn_mask_.device, dtype=attention_mask.dtype
                        )]
                    )
                input_ids.append(input_ids_)
                attn_mask.append(attn_mask_)
                num_tokens_per_step.append(
                    torch.tensor(num_tokens_per_step_, device=sequences.device),
                )
            input_ids = right_padding(input_ids, self.tokenizer.pad_token_id)
            attn_mask = right_padding(attn_mask, 0)
            num_tokens_per_step = right_padding(num_tokens_per_step, 0)

            pred_rew_ids = (score_ids - 1 + problem_length).masked_fill(
                ~score_mask, -1
            )
            
            if "PRM" in self.strategy.args.reward_mode:
                r_ref = self.reward_model.forward.remote(
                    input_ids, attn_mask, 
                    packed_seq_lens=packed_seq_lens, 
                    no_grad=True,
                )
            else:
                r_ref = ray.put(None)

            if getattr(self.strategy.args, 'colocate_critic_reward', True):
                ray.get([r_ref])
                ray.get([self.reward_model.empty_cache.remote()])
        else:
            raise NotImplementedError
            # remote RM
            if not self.packing_samples:
                queries = self.tokenizer.batch_decode(sequences_cpu, skip_special_tokens=False)
            else:
                sequences_list = []
                offset = 0
                tokens_list = sequences_cpu.tolist()[0]
                for length in packed_seq_lens:
                    sequences_list.append(tokens_list[offset : offset + length])
                    offset += length
                queries = self.tokenizer.batch_decode(sequences_list, skip_special_tokens=False)

            for rm in self.remote_rm_url:
                r = remote_rm_fn_ray.remote(rm, queries=queries)
                r_refs.append(r)

        # log probs
        action_log_probs = self.actor(
            sequences, num_actions, attention_mask, 
            packed_seq_lens=packed_seq_lens,
        )
        actor_value_rm_time = time.time() - start

        # wait initial/critic/reward model done
        start = time.time()
        ref_values = ray.get([base_action_log_probs_ref, value_ref, r_ref])
        wait_time = time.time() - start

        base_action_log_probs, value, process_reward = ref_values[0], ref_values[1], ref_values[2]
        base_action_log_probs = base_action_log_probs.to(device)
        if value is not None:
            value = value.to(device)

        # process reward for every tokens
        # only non-zero for the step separator tokens
        process_reward_matrix = torch.zeros_like(
            action_log_probs, 
            device=action_log_probs.device, 
            dtype=action_log_probs.dtype,
        )
        if "PRM" in self.strategy.args.reward_mode:
            process_reward = process_reward.to(device)
            # step-level process reward
            process_reward = torch.stack(
                [process_reward[i, pred_rew_ids[i]] for i in range(bs)]
            )
            # weighted sum of step-level process reward (approx. min)
            process_reward = turn_process_reward_logits_to_reward(
                process_reward, score_mask, 
                return_outcome_reward=False, 
                temperature=0.1,
                disable_weighted_reward=self.strategy.args.disable_weighted_reward,
            )
            process_reward_matrix[samples.reward_mask] = process_reward[
                score_mask].to(dtype=action_log_probs.dtype)

        # avoid CUDA OOM when colocate models
        if getattr(self.strategy.args, 'colocate_critic_reward', True) and not self.remote_rm_url:
            ray.get([self.reward_model.empty_cache.remote()])

        if getattr(self.strategy.args, 'colocate_actor_ref', True):
            torch.cuda.empty_cache()

        kl = compute_approx_kl(
            action_log_probs,
            base_action_log_probs,
            action_mask=action_mask,
            use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
        )

        if not self.packing_samples:
            kl_mean = masked_mean(kl, action_mask, dim=-1)
        else:
            # convert tensor into list of tensors so that it's easier to manipulate
            # within dataset.
            sequences = unpacking_samples(sequences, packed_seq_lens)
            attention_mask = None
            action_log_probs = unpacking_samples(action_log_probs, num_actions)
            if value is not None:
                value = unpacking_samples(value, num_actions)

            kl = unpacking_samples(kl, num_actions)
            kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)

        info = {
            "kl": kl_mean,
            "process_reward": process_reward_matrix,
            "process_reward_mask": samples.reward_mask,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
            "max_num_steps": samples.max_num_steps,
            "num_steps": samples.num_steps,
            "solved": samples.solved,
            "have_answers": samples.have_answers,
            "num_tokens_per_step": num_tokens_per_step,
            "score_mask": score_mask,
        }

        if self.strategy.args.perf:
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time

        experience = Experience(
            sequences,
            action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )

        self.actor.train()  # reset model state
        return experience

    def _generate_vllm(self, all_data, **kwargs) -> List[Samples]:
        from vllm import SamplingParams

        # round-robin load balance
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )

        all_prompts, all_answers, have_answers = all_data['prompt'], all_data['answer'], all_data['have_answer']
        # Expand prompt list based on the number of samples per prompt
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_answers = sum([[answer] * args.n_samples_per_prompt for answer in all_answers], [])
        have_answers = sum([[have_answer] * args.n_samples_per_prompt for have_answer in have_answers], [])
        all_prompt_token_ids = self.tokenize_fn(
            all_prompts, self.prompt_max_len, padding=False,
        )["input_ids"]

        # Distribute requests to engines and collect responses to outputs
        all_output_refs = []
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
            if prompt_token_ids:
                all_output_refs.append(
                    llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
                )

        # Retrieve and combine results from all outputs
        all_outputs = sum(ray.get(all_output_refs), [])

        samples_list = []
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            outputs = all_outputs[i : i + args.micro_rollout_batch_size]
            answers = all_answers[i : i + args.micro_rollout_batch_size]
            have_ans = have_answers[i : i + args.micro_rollout_batch_size]

            if not self.packing_samples:
                # NOTE: concat all outputs to following format:
                #
                # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
                # | token token token token token | token token [EOS] [PAD] |
                # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
                # |<---------- prompt ----------->|<-------- answer ------->|
                max_input_len, max_output_len = 0, 0
                for output in outputs:
                    max_input_len = max(max_input_len, len(output.prompt_token_ids))
                    max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                for output in outputs:
                    # left padding input
                    input_len = len(output.prompt_token_ids)
                    input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

                    # right padding output
                    output_len = len(output.outputs[0].token_ids)
                    output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)

                    # concat input and output
                    sequences.append(input_ids + output_ids)

                sequences = torch.tensor(sequences)
                sequences, attention_mask, action_mask = self.actor.process_sequences(
                    sequences, max_input_len, eos_token_id, pad_token_id
                )
                sequences = sequences.to("cuda")
                attention_mask = attention_mask.to("cuda")
                action_mask = action_mask.to("cuda")

                num_actions = action_mask.size(1)
                bs = sequences.size(0)

                solution_tokens = sequences[:, -num_actions:]
                # find step separator, typically '\n\n'
                row_ids, column_ids = torch.where(
                    torch.isin(solution_tokens, self.step_separator_tokens)
                )
                # +1 for the last step with eos rather than step separator
                max_num_steps = max([column_ids[row_ids==i].numel() for i in range(bs)]) + 1
                # start to end index of each step
                score_ids = torch.full(
                    (bs, max_num_steps), -1, 
                    dtype=torch.long, 
                    device=sequences.device,
                )
                # whether the end of step
                reward_mask = torch.zeros_like(solution_tokens, dtype=torch.bool)
                eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(1)
                solved = []
                for j in range(bs):
                    num_steps = column_ids[row_ids==j].numel()
                    score_ids[j, :num_steps] = column_ids[row_ids==j] + 1
                    reward_mask[j, column_ids[row_ids==j]] = True
                    score_ids[j, num_steps] = eos_indices[j] + 1
                    reward_mask[j, eos_indices[j]] = True

                    if answers[j] != 'null':
                        solved.append(compute_score(
                            solution_str=self.tokenizer.decode(
                                solution_tokens[j], 
                                skip_special_tokens=True,
                            ), 
                            ground_truth=answers[j],
                        ))
                    else:
                        # later, verifiable reward = solved - 1
                        # for the case of no answer, VR = 0
                        solved.append(1)
                
                score_mask = score_ids != -1
                solved = torch.tensor(solved, device=sequences.device)
                have_ans = torch.tensor(have_ans, device=sequences.device, dtype=torch.bool)

                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        score_ids=score_ids,
                        score_mask=score_mask,
                        reward_mask=reward_mask,
                        max_num_steps=max_num_steps,
                        num_steps=score_mask.float().sum(dim=-1),
                        num_actions=num_actions,
                        solved=solved,
                        have_answers=have_ans,
                        packed_seq_lens=None,
                        problem_length=max_input_len,
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                    )
                )
            else:
                raise NotImplementedError
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                num_actions = []
                for i, output in enumerate(outputs):
                    input_len = len(output.prompt_token_ids)
                    output_len = len(output.outputs[0].token_ids)
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))
                    attention_mask.extend([i + 1] * (input_len + output_len))

                    # current_action_mask = [0] * (input_len - 1) + [1] * output_len + [0]
                    # num_actions.append(max(1, sum(current_action_mask)))
                    num_actions.append(max(1, output_len))

                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None
                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                    )
                )
        return samples_list

    def flush(self):
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None
