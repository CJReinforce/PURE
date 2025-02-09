### Requirements
You can install the required packages with the following command:
```bash
cd latex2sympy
pip install -e .
cd ..
pip install -r requirements.txt 
pip install vllm==0.5.1 --no-build-isolation
pip install transformers==4.42.3
```

### Evaluation
You can evaluate the fine-tuned model with the following command:
```bash
bash sh/eval.sh
```

## Acknowledgement
The codebase is adapted from [math-evaluation-harness](https://github.com/ZubinGou/math-evaluation-harness).
