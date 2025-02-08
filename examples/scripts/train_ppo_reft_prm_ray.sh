set -x 

export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
timestep=$(date "+%m%d%H%M%S")

python -m openrlhf.cli.train_reft_prm_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 4 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 4 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 1 \
   --temperature 0.5 \
   --advantage_estimator rloo \
   --reward_baseline token \
   --reward_mode PRMVR \
   --verifiable_reward_coef 1.0 \
   --n_samples_per_prompt 4 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 64 \
   --max_epochs 1 \
   --micro_train_batch_size 2 \
   --actor_learning_rate 3e-7 \
   --init_kl_coef 0.001 \
   --pretrain /mnt/petrelfs/chengjie/Qwen2.5-Math-7B \
   --reward_pretrain /mnt/petrelfs/chengjie/ceph5/qwen25-math-7b-PRM800k-bs128-lr1e-6-epoch-1-stage2 \
   --save_path /mnt/petrelfs/chengjie/ceph5/openrlhf/ray_debug_$timestep \
   --ckpt_path /mnt/petrelfs/chengjie/ceph5/openrlhf/ray_debug_$timestep/checkpoints \
   --save_steps 10 \
   --max_ckpt_num 15 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 128 \
   --num_episodes 2000 \
   --prompt_max_len 1024 \
   --generate_max_len 2048 \
   --zero_stage 2 \
   --bf16 \
   --prompt_data data/math_level3to5_data_processed_with_qwen_prompt.json \
   --input_key question \
   --max_samples 1000000 \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --disable_ds_ckpt \
   --save_hf_ckpt \
   --use_wandb True

# --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}' [Install deps]
# --ref_reward_offload [Offload to CPU]
# --remote_rm_url http://localhost:5000/get_reward

# --vllm_sync_backend nccl (Only for multi-nodes with vLLM 0.6.4+ or vLLM 0.4.2)