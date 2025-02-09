import os
import os.path
import shutil
from abc import ABC
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from openrlhf.models import Actor, GPTLMLoss, PolicyLoss
from openrlhf.utils.distributed_sampler import DistributedSampler

from .ppo_utils import (
    AdaptiveKLController,
    Experience,
    FixedKLController,
    NaiveExperienceMaker,
    NaiveReplayBuffer,
)


class PURETrainer(ABC):
    def __init__(
        self,
        strategy,
        actor: Actor,
        reward_model: nn.Module,
        initial_actor: Actor,
        last_actor: Actor,
        initial_reward_model: nn.Module,
        ema_model: Actor,
        actor_optim: Optimizer,
        actor_scheduler,
        reward_model_optim: Optimizer,
        reward_model_scheduler,
        ema_beta: float = 0.992,
        init_kl_coef: float = 0.001,
        kl_target: float = None,
        kl_horizon: int = 10000,
        ptx_coef: float = 0,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        micro_rollout_batch_size: int = 8,
        gradient_checkpointing: bool = False,
        max_epochs: int = 1,
        max_norm: float = 1.0,
        tokenizer: Optional[Callable[[Any], dict]] = None,
        prompt_max_len: int = 128,
        dataloader_pin_memory: bool = True,
        save_hf_ckpt: bool = False,
        disable_ds_ckpt: bool = False,
        **generate_kwargs,
    ) -> None:

        super().__init__()
        self.strategy = strategy
        self.args = strategy.args
        self.save_hf_ckpt = save_hf_ckpt
        self.disable_ds_ckpt = disable_ds_ckpt
        self.micro_rollout_batch_size = micro_rollout_batch_size
        self.max_epochs = max_epochs
        self.tokenizer = tokenizer
        self.generate_kwargs = generate_kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.max_norm = max_norm
        self.ptx_coef = ptx_coef
        self.micro_train_batch_size = micro_train_batch_size
        self.kl_target = kl_target
        self.prompt_max_len = prompt_max_len
        self.ema_beta = ema_beta
        self.gradient_checkpointing = gradient_checkpointing

        self.actor = actor
        self.reward_model = reward_model
        self.initial_actor = initial_actor
        self.last_actor = last_actor
        self.initial_reward_model = initial_reward_model
        self.ema_model = ema_model
        self.actor_optim = actor_optim
        self.actor_scheduler = actor_scheduler
        self.reward_model_optim = reward_model_optim
        self.reward_model_scheduler = reward_model_scheduler

        self.actor_loss_fn = PolicyLoss(eps_clip)
        self.ptx_loss_fn = GPTLMLoss()

        self.freezing_actor_steps = getattr(self.args, "freezing_actor_steps", -1)

        # Mixtral 8x7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(init_kl_coef, kl_target, kl_horizon)
        else:
            self.kl_ctl = FixedKLController(init_kl_coef)

        # maybe it's only for LLM training stage,
        # and perhaps needs other ExperienceMaker & ReplayBuffer for PRM training stage
        self.experience_maker = NaiveExperienceMaker(
            actor,
            None,
            reward_model,
            initial_actor,
            tokenizer,
            prompt_max_len,
            self.kl_ctl,
            strategy,
        )
        packing_samples = getattr(self.args, "packing_samples", False)
        self.replay_buffer = NaiveReplayBuffer(
            micro_train_batch_size, buffer_limit, buffer_cpu_offload, packing_samples
        )

        # save codes and scripts
        if self.strategy.is_rank_0():
            code_path = os.path.join(self.args.save_path, "src")
            script_path = os.path.join(self.args.save_path, "scripts")
            shutil.copytree(src="./openrlhf", dst=code_path)
            shutil.copytree(src="./examples/scripts", dst=script_path)
            print("Save codes and scripts successfully")

        # wandb/tensorboard setting
        self._wandb = None
        self._tensorboard = None
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/epoch")
            wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)

        # Initialize TensorBoard writer if wandb is not available
        if self.strategy.args.use_tensorboard and self._wandb is None and self.strategy.is_rank_0():
            from torch.utils.tensorboard import SummaryWriter

            os.makedirs(self.strategy.args.use_tensorboard, exist_ok=True)
            log_dir = os.path.join(self.strategy.args.use_tensorboard, strategy.args.wandb_run_name)
            self._tensorboard = SummaryWriter(log_dir=log_dir)

    def fit(
        self,
        args,
        prompts_dataloader,
        pretrain_dataloader,
        consumed_samples=0,
        num_update_steps_per_episodes=1,
    ) -> None:
        num_rollouts_per_episodes = (
            num_update_steps_per_episodes
            * args.train_batch_size
            // args.max_epochs
            // args.rollout_batch_size
            // args.n_samples_per_prompt
        )

        # get eval and save steps
        if args.eval_steps == -1:
            args.eval_steps = num_rollouts_per_episodes  # Evaluate once per episode
        if args.save_steps == -1:
            args.save_steps = float("inf")  # do not save ckpt

        self.prompts_dataloader = prompts_dataloader
        self.pretrain_dataloader = pretrain_dataloader

        # Restore step and start_epoch
        steps = consumed_samples // args.rollout_batch_size + 1
        start_episode = consumed_samples // args.rollout_batch_size // num_rollouts_per_episodes
        consumed_samples = consumed_samples % (num_rollouts_per_episodes * args.rollout_batch_size)

        for episode in range(start_episode, args.num_episodes):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(
                    episode, consumed_samples=0 if episode > start_episode else consumed_samples
                )
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=not self.strategy.is_rank_0(),
            )

            for rand_data in self.prompts_dataloader:
                for i, experience in enumerate(
                    self.experience_maker.make_experience_list(
                        rand_data, **self.generate_kwargs)
                ):
                    if i == 0:
                        output = self.tokenizer.decode(
                            experience.sequences[0], 
                            skip_special_tokens=True,
                        )
                        print_info = {
                            "rollout": output.split("\n\n"),
                            "outcome_reward": round(experience.info["reward"][0].item(), 2),
                            "process_reward": experience.info["process_reward"][
                                0, experience.info["process_reward_mask"][0]],
                            "level": rand_data['level'][0],
                        }
                        if rand_data['have_answer'][0].item():
                            print_info["answer"] = rand_data['answer'][0]
                            print_info["match"] = bool(experience.info["match"][0].item())
                            print_info["solved"] = bool(experience.info["solved"][0].item())
                        else:
                            print_info["ref_answer"] = rand_data['ref_answer'][0]
                        self.strategy.print(print_info)
                    del experience.info["process_reward"]
                    del experience.info["process_reward_mask"]
                    self.replay_buffer.append(experience)

                torch.cuda.empty_cache()
                if not self.args.disable_advantage_normalization:
                    self.replay_buffer.normalize("advantages", self.strategy)
                status = self.ppo_train(steps)
                # calculate solved rate
                status.update(self.calculate_solved_rate())
                self.replay_buffer.clear()
                torch.cuda.empty_cache()

                if "kl" in status:
                    self.kl_ctl.update(status["kl"], args.rollout_batch_size * args.n_samples_per_prompt)
                pbar.set_postfix(status)

                # logs/checkpoints
                client_states = {"consumed_samples": steps * args.rollout_batch_size}
                self.save_logs_and_checkpoints(args, steps, pbar, status, client_states)

                pbar.update()
                steps = steps + 1

        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.close()

    def ppo_train(self, global_steps=0):
        # replay buffer may be empty at first, we should rebuild at each training
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for experience in pbar:
                experience.to_device(device)
                status = self.training_step(experience, global_steps)

                # for DP
                # weighted mean for kl
                if "kl" in status:
                    status["kl"] *= status["response_length"]
                    status = self.strategy.all_reduce(status)
                    status["kl"] /= status["response_length"]

                short_status = {}

                if "policy_loss" in status:
                    short_status = {
                        "pg": status["policy_loss"],
                        "rm": status["reward"],
                        "ret": status["return"],
                        "glen": status["response_length"],
                        "slen": status["num_steps"],
                        "tlen": status["total_length"],
                        "kl": status["kl"],
                        "act_lr": status["actor_lr"],
                    }

                if "critic_loss" in status:
                    short_status["cri"] = status["critic_loss"]
                    short_status["vals"] = status["values"]
                    short_status["cri_lr"] = status["critic_lr"]

                if "ptx_loss" in status:
                    short_status["ptx"] = status["ptx_loss"]

                status_list.append(status)
                pbar.set_postfix(short_status)

        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        return status_mean

    def training_step(self, experience: Experience, global_steps) -> Dict[str, float]:
        status = {}
        if global_steps > self.freezing_actor_steps:
            status = self.training_step_actor(experience)
        return status

    def training_step_actor(self, experience: Experience) -> Dict[str, float]:
        self.actor.train()

        # TODO: this is a bad indicator to say that data is packed...
        if isinstance(experience.sequences, list):
            sequences = torch.cat(experience.sequences, dim=0).unsqueeze(0)
            old_action_log_probs = torch.cat(experience.action_log_probs, dim=0).unsqueeze(0)
            advantages = torch.cat(experience.advantages, dim=0).unsqueeze(0)
            num_actions = [v.numel() for v in experience.advantages]
            packed_seq_lens = [s.numel() for s in experience.sequences]
            attention_mask = torch.cat(
                [torch.full_like(s, i + 1) for i, s in enumerate(experience.sequences)], dim=0
            ).unsqueeze(0)
        else:
            sequences = experience.sequences
            old_token_log_probs = experience.action_log_probs
            advantages = experience.advantages
            action_mask = experience.action_mask
            num_tokens = action_mask.size(1)
            packed_seq_lens = None
            attention_mask = experience.attention_mask

        # actor loss
        token_log_probs, output = self.actor(
            sequences,
            num_tokens,
            attention_mask=attention_mask,
            return_output=True,
            packed_seq_lens=packed_seq_lens,
        )
        
        # loss function
        actor_loss, loss_info = self.actor_loss_fn(
            token_log_probs,
            old_token_log_probs,
            advantages,
            action_mask=action_mask,
        )
        # mixtral
        if self.aux_loss:
            aux_loss = output.aux_loss
        else:
            aux_loss = 0
        loss = actor_loss + aux_loss * self.args.aux_loss_coef
        self.strategy.backward(loss, self.actor, self.actor_optim)

        # ptx loss
        if self.pretrain_dataloader is not None:
            data = next(self.pretrain_dataloader)
            inputs = data[1].squeeze(1).to(torch.cuda.current_device())
            attention_mask = data[2].squeeze(1).to(torch.cuda.current_device())
            label = torch.where(
                attention_mask.bool(),
                inputs,
                self.ptx_loss_fn.IGNORE_INDEX,
            )

            output = self.actor(inputs, attention_mask=attention_mask, return_output=True)
            ptx_log_probs = output["logits"]

            # loss function
            ptx_loss = self.ptx_loss_fn(ptx_log_probs, label)
            # mixtral
            if self.aux_loss:
                aux_loss = output.aux_loss
            else:
                aux_loss = 0
            loss = ptx_loss + aux_loss * self.args.aux_loss_coef
            self.strategy.backward(self.ptx_coef * loss, self.actor, self.actor_optim)

        self.strategy.optimizer_step(self.actor_optim, self.actor, self.actor_scheduler, name="actor")
        if self.ema_model:
            self.strategy.moving_average(self.actor, self.ema_model, self.ema_beta, "cpu")

        # status
        status = {
            "policy_loss": actor_loss.item(), 
            "actor_lr": self.actor_scheduler.get_last_lr()[0],
        }
        if self.pretrain_dataloader is not None:
            status["ptx_loss"] = ptx_loss.item()
        for k, v in experience.info.items():
            if k == "kl":
                status[k] = (
                    (v * experience.info["response_length"]).sum() / experience.info["response_length"].sum()
                ).item()
            elif k in ["solved", "have_answers", "match"]:
                pass
            else:
                status[k] = v.mean().item()
        for k, v in loss_info.items():
            status[k] = v.mean().item()
        return status

    def calculate_solved_rate(self):
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=len(self.replay_buffer),
            shuffle=False,
            drop_last=False,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        all_data = next(iter(dataloader)).info
        device = torch.cuda.current_device()
        solved, have_answers, match = all_data["solved"].to(device), all_data["have_answers"].to(device), all_data["match"].to(device)

        num_solved = solved[have_answers].sum().float()
        torch.distributed.all_reduce(num_solved)
        num_samples = have_answers.sum().float()
        torch.distributed.all_reduce(num_samples)
        num_match = match[have_answers].sum().float()
        torch.distributed.all_reduce(num_match)
        total_samples = torch.tensor(solved.numel(), device=device, dtype=torch.float)
        torch.distributed.all_reduce(total_samples)

        info = {
            "solved": num_solved.item() / num_samples.item() if num_samples.item() > 0 else 0,
            "match": num_match.item() / num_samples.item() if num_samples.item() > 0 else 0,
            "have_answers": num_samples.item() / total_samples.item(),
        }
        return info

    def save_logs_and_checkpoints(self, args, global_step, step_bar, logs_dict={}, client_states={}):
        if global_step % args.logging_steps == 0:
            # wandb
            if self._wandb is not None and self.strategy.is_rank_0():
                logs = {
                    "train/%s" % k: v
                    for k, v in {
                        **logs_dict,
                        "global_step": global_step,
                    }.items()
                }
                if self.experience_maker.perf_stats is not None:
                    logs.update({f"perf/experience_maker/{k}": v for k, v in self.experience_maker.perf_stats.items()})
                self._wandb.log(logs)
            # TensorBoard
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)
                if self.experience_maker.perf_stats is not None:
                    for k, v in self.experience_maker.perf_stats.items():
                        self._tensorboard.add_scalar(f"perf/experience_maker/{k}", v, global_step)

        # TODO: Add evaluation mechanism for PPO
        if global_step % args.eval_steps == 0:
            # self.evaluate(self.eval_dataloader, global_step)
            pass
        # save ckpt
        # TODO: save best model on dev, use loss/perplexity/others on whole dev dataset as metric
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            self._save_checkpoint(args, tag, client_states)

    def _save_checkpoint(self, args, tag, client_states):
        if not self.disable_ds_ckpt:
            self.strategy.save_ckpt(
                self.actor.model,
                os.path.join(args.ckpt_path, "_actor"),
                tag,
                args.max_ckpt_num,
                args.max_ckpt_mem,
                client_states,
            )

        if self.save_hf_ckpt:
            save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
            self.strategy.save_model(self.actor, self.tokenizer, save_path)
