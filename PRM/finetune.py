import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import transformers
from datasets import load_dataset
from torch import distributed as dist
from trl import PRMConfig, PRMTrainer


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2-0.5B")


@dataclass
class DataArguments:
    train_data_path: str = field(default="trl-lib/math_shepherd")
    eval_data_path: str = field(default=None)
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(PRMConfig):
    cache_dir: Optional[str] = field(default=None)
    max_length: int = field(default=128000)
    max_completion_length: int = field(default=8000)
    fix_llm: bool = field(default=False)


def safe_save_model_for_hf_trainer(
        trainer: transformers.Trainer, 
        output_dir: str,
    ):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()

    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


def make_supervised_data_module(data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    assert data_args.train_data_path is not None
    
    train_dataset = load_dataset(data_args.train_data_path, split="train")
    eval_dataset = load_dataset(data_args.train_data_path, split="test")

    return dict(
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset, 
    )


def train():
    os.environ["WANDB_PROJECT"]="PRM_Math_Shepherd"
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    
    (
        model_args,
        data_args,
        training_args,
        _
    ) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    # Load model and tokenizer
    model = transformers.AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        use_cache = False,
    )

    # freeze llm except last layer if needed
    if training_args.fix_llm:
        model.model.requires_grad_(False)
                
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        use_fast=False,
        trust_remote_code=True,
    )

    # Load data
    data_module = make_supervised_data_module(data_args=data_args)

    # Start trainner
    trainer = PRMTrainer(
        model=model, 
        processing_class=tokenizer, 
        args=training_args, 
        **data_module
    )

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(
        trainer=trainer, 
        output_dir=training_args.output_dir
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    train()