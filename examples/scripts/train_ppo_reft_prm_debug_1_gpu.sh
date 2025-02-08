set -x

DEVICES=$CUDA_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES


GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=36001


DISTRIBUTED_ARGS="
    --num_gpus $GPUS_PER_NODE \
    --num_nodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

timestep=$(date "+%m%d%H%M%S")


read -r -d '' training_commands <<EOF
openrlhf.cli.train_ppo \
   --pretrain /mnt/petrelfs/chengjie/Qwen2.5-Math-7B \
   --temperature 0.5 \
   --reward_pretrain /mnt/petrelfs/chengjie/ceph5/qwen25-math-7b-PRM800k-bs128-lr1e-6-epoch-1-stage2 \
   --save_path /mnt/petrelfs/chengjie/ceph5/openrlhf/debug_$timestep \
   --ckpt_path /mnt/petrelfs/chengjie/ceph5/openrlhf/debug_$timestep/checkpoints \
   --advantage_estimator rloo \
   --reward_baseline token \
   --n_samples_per_prompt 2 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 2 \
   --save_steps 50 \
   --logging_steps 1 \
   --eval_steps -1 \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --max_epochs 2 \
   --prompt_max_len 1024 \
   --generate_max_len 2048 \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --init_kl_coef 0.01 \
   --prompt_data data/8k_math_72k_numina.jsonl \
   --input_key question \
   --max_samples 1000000 \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing
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