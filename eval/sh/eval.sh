set -ex

PROMPT_TYPE="qwen25-math-cot-ft"
MODEL_NAME_OR_PATH=<local path to the model>
OUTPUT_DIR="./output"

SPLIT="test"
NUM_TEST_SAMPLE=-1

DATA_NAME="math500,minerva_math,olympiadbench,aime24,amc23"
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --max_tokens_per_call 3000 \
    --seed 42 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm
