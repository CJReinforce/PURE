set -ex

# export CUDA_VISIBLE_DEVICES=0
MODEL_NAME_OR_PATH=<local path to the model>

if [[ "${MODEL_NAME_OR_PATH,,}" =~ "deepseek" ]]; then
    PROMPT_TYPE="deepseek-distill-cot-ft"
    temperature=0.6
    top_p=0.95
    max_gen_len=32768
else
    PROMPT_TYPE="qwen25-math-cot-ft"
    temperature=0.0
    top_p=1.0
    max_gen_len=3000
fi

SPLIT="test"
NUM_TEST_SAMPLE=-1
OUTPUT_DIR="./output"
DATA_NAME="math500,minerva_math,olympiadbench,aime24,amc23"

TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --max_tokens_per_call $max_gen_len \
    --temperature $temperature \
    --top_p $top_p \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 42 \
    --n_sampling 1 \
    --start 0 \
    --end -1 \
    --use_vllm

echo "Model CKPT: "$MODEL_NAME_OR_PATH