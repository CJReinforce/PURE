import os

import torch
from datasets import interleave_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer


def get_tokenizer(pretrain, model, padding_side="left", strategy=None, use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
    tokenizer.padding_side = padding_side
    # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
    # https://github.com/facebookresearch/llama-recipes/pull/196
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def get_strategy(args):
    from openrlhf.utils.deepspeed import DeepspeedStrategy

    strategy = DeepspeedStrategy(
        seed=getattr(args, "seed", 42),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args, "micro_train_batch_size", 1),
        train_batch_size=getattr(args, "train_batch_size", 128),
        zero_stage=args.zero_stage,
        bf16=getattr(args, "bf16", True),
        args=args,
    )
    return strategy


def blending_datasets(
    datasets,
    probabilities,
    strategy=None,
    seed=42,
    max_count=5000000,
    return_eval=True,
    stopping_strategy="first_exhausted",
    train_split="train",
    eval_split="test",
):
    datasets = datasets.split(",")
    probabilities = list(map(float, probabilities.split(",")))
    assert len(probabilities) == len(datasets)

    train_data_list = []
    eval_data_list = []
    for i, dataset in enumerate(datasets):
        dataset = dataset.strip()
        strategy.print(f"dataset: {dataset}")

        data_dir = dataset.split("@")[1].strip() if "@" in dataset else None
        dataset = dataset.split("@")[0].strip()
        dataset_basename = os.path.basename(dataset)

        ext = os.path.splitext(dataset)[-1]
        # local python script
        if ext == ".py" or (
            os.path.isdir(dataset) and os.path.exists(os.path.join(dataset, f"{dataset_basename}.py"))
        ):
            data = load_dataset(dataset, trust_remote_code=True)
            strategy.print(f"loaded {dataset} with python script")
        # local text file
        elif ext in [".json", ".jsonl", ".csv"]:
            ext = ext.lower().strip(".")
            if ext == "jsonl":
                ext = "json"
            data = load_dataset(ext, data_files=dataset)
            strategy.print(f"loaded {dataset} with data_files={dataset}")
        # local dataset saved with `datasets.Dataset.save_to_disk`
        elif os.path.isdir(dataset):
            data = load_from_disk(dataset)
            strategy.print(f"loaded {dataset} from disk")
        # remote/local folder or common file
        else:
            data = load_dataset(dataset, data_dir=data_dir)
            strategy.print(f"loaded {dataset} from files")

        if train_split and train_split in data:
            train_data = data[train_split].select(range(min(max_count, len(data[train_split]))))
        else:
            train_data = data.select(range(min(max_count, len(data))))
        train_data_list.append(train_data)

        if return_eval:
            if eval_split and eval_split in data:
                eval_data = data[eval_split].select(range(min(max_count, len(data[eval_split]))))
            # train will contains eval? TODO
            else:
                eval_data = train_data.select(range(min(max_count, int(len(train_data) * 0.03))))
            eval_data_list.append(eval_data)

    # merge datasets
    if strategy.is_rank_0():
        print(train_data_list)

    train_dataset = interleave_datasets(
        train_data_list,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
    if return_eval:
        eval_dataset = interleave_datasets(
            eval_data_list,
            probabilities=probabilities,
            seed=seed,
            stopping_strategy=stopping_strategy,
        )
        return train_dataset, eval_dataset
    else:
        return train_dataset


def convert_token_to_id(token, tokenizer):
    if isinstance(token, str):
        token = tokenizer.encode(token, add_special_tokens=False)
        assert len(token) == 1
        return token[0]
    else:
        raise ValueError("token should be int or str")


def right_padding(input_ids, pad_token_id):
    max_len = max([i.size(-1) for i in input_ids])
    for i, input_idx in enumerate(input_ids):
        real_idx = input_idx # if input_idx.numel() == 1 else input_idx.squeeze()
        if real_idx.ndim == 1:
            real_idx = real_idx.unsqueeze(0)
        assert real_idx.ndim == 2

        input_ids[i] = torch.cat([
            real_idx, 
            torch.tensor(
                [pad_token_id] * (max_len - real_idx.size(-1)),
                dtype=real_idx.dtype,
                device=real_idx.device
            ).unsqueeze(0).repeat(real_idx.size(0), 1)
        ], dim=-1)
    input_ids = torch.cat(input_ids, dim=0)
    return input_ids


def turn_process_reward_logits_to_reward(
    logits, reward_mask, 
    return_outcome_reward=False, 
    temperature=0.1,
    disable_weighted_reward=False,
):
    assert logits.ndim == 3 and \
        logits.size(-1) == 2 and \
            reward_mask.ndim == 2 and \
                reward_mask.shape == logits.shape[:2]
    softmax_logits = logits.softmax(dim=-1)
    process_reward = softmax_logits[..., 1] - softmax_logits[..., 0]
    process_reward = process_reward.masked_fill(~reward_mask, 0)
    
    if not disable_weighted_reward:
        weight = torch.softmax(
            -process_reward.masked_fill(
                ~reward_mask, float("inf")
            ) / temperature, 
            dim=-1,
        )
        process_reward = weight * process_reward

    if return_outcome_reward:
        outcome_reward = process_reward.sum(dim=-1)
        return process_reward, outcome_reward
    return process_reward


def token_log_prob_to_step_log_prob(token_log_probs, score_ids, score_mask):
    bs = token_log_probs.size(0)
    step_log_probs = []
    for i in range(bs):
        step_log_probs_per_sample = []
        for idx, j in enumerate(score_ids[i][score_mask[i]]):
            if idx == 0:
                step_log_prob = token_log_probs[i, :j]
            else:
                step_log_prob = token_log_probs[i, score_ids[i, idx-1]:j]
            step_log_probs_per_sample.append(step_log_prob.sum(dim=0, keepdim=True))
        step_log_probs.append(torch.cat(step_log_probs_per_sample, dim=0))
    return right_padding(step_log_probs, 0)