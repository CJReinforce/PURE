import json

import numpy as np

# load json file: math_level3to5_data_processed_with_qwen_prompt.json
with open("data/math_level3to5_data_processed_with_qwen_prompt.json", "r") as f:
    ds = json.load(f)

new_ds = []
k = 0
for data in ds:
    question = data["question"]
    level = str(data["level"])
    
    if np.random.rand() < 0.9:
        answer = "null"
        ref_ans = data["answer"]
    else:
        answer = data["answer"]
        ref_ans = "null"
        k += 1
    new_ds.append({"question": question, "answer": answer, "ref_answer": ref_ans, "level": level})

# save new_ds as json file
with open(f"data/8k_math_{k}_answers.json", "w") as f:
    json.dump(new_ds, f, indent=2)