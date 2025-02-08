set -x

DEVICES=$CUDA_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
MASTER_PORT=36001
timestep=$(date "+%m%d%H%M%S")


read -r -d '' training_commands <<EOF
openrlhf.cli.train_ppo \
   --pretrain /mnt/petrelfs/chengjie/Qwen2.5-Math-7B \
   --temperature 0.5 \
   --reward_pretrain /mnt/petrelfs/chengjie/ceph5/qwen25-math-7b-PRM800k-bs128-lr1e-6-epoch-1-stage2 \
   --save_path /mnt/petrelfs/chengjie/ceph5/openrlhf/debug_$timestep \
   --ckpt_path /mnt/petrelfs/chengjie/ceph5/openrlhf/debug_$timestep/checkpoints \
   --advantage_estimator rloo \
   --reward_baseline step \
   --reward_mode VR \
   --n_samples_per_prompt 4 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 32 \
   --save_steps 10 \
   --max_ckpt_num 10 \
   --logging_steps 1 \
   --eval_steps -1 \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --max_epochs 2 \
   --num_episodes 20 \
   --prompt_max_len 1024 \
   --generate_max_len 2048 \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --init_kl_coef 0.001 \
   --prompt_data data/math_level3to5_data_processed_with_qwen_prompt.json \
   --input_key question \
   --max_samples 1000000 \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --use_wandb True \
   --disable_ds_ckpt \
   --save_hf_ckpt
EOF

    # set input_template in the code
    # --apply_chat_template
    # --normalize_reward
    # --packing_samples
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --remote_rm_url http://localhost:5000/get_reward

if [[ ${1} != "slurm" ]]; then
    deepspeed --master_port $MASTER_PORT --include localhost:$DEVICES --module $training_commands
fi