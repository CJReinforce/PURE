import json

from datasets import load_dataset
from tqdm import tqdm

ds = load_dataset("PRIME-RL/Eurus-2-RL-Data", split="train")

new_ds = []
for data in tqdm(ds):
    if data['ability'] == 'math' and \
        'synthetic' not in data['data_source'] and \
            data['data_source'] in ['numina_cn_k12', 'numina_olympiads'] and \
            len(data['reward_model']['ground_truth']) < 20:
        
        question = data['prompt'][-1]['content'].rstrip(
            '\n\nPresent the answer in LaTex format: \\boxed{Your answer}'
        )
        if '___' not in question and \
            '\xa0' not in question and \
                '=' not in question:
            answer = data['reward_model']['ground_truth']
            new_data = {
                'question': question, 
                'ref_answer': answer, 
                'level': data['data_source'],
                'answer': 'null',
            }
            new_ds.append(new_data)


with open('math_level3to5_data_processed_with_qwen_prompt.json', 'r') as f:
    ds = json.load(f)


for data in tqdm(ds):
    new_data = {
        'question': data['question'],
        'answer': data['answer'],
        'level': str(data['level']),
        'ref_answer': 'null',
    }
    new_ds.append(new_data)

# save new_ds
with open('8k_math_72k_numina.jsonl', 'w', encoding='utf-8') as f:
    for item in new_ds:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')