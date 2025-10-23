<div align="center">

# Stop Summation: Min-Form Credit Assignment Is All Process Reward Model Needs for Reasoning

[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2504.15275)  [![Github](https://img.shields.io/badge/PURE-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/CJReinforce/PURE)  [![GitHub Stars](https://img.shields.io/github/stars/CJReinforce/PURE?style=for-the-badge&logo=github&logoColor=white&label=Stars&color=000000)](https://github.com/CJReinforce/PURE)  [![Hugging Face Collection](https://img.shields.io/badge/PURE_Collection-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/jinachris/pure-67a85510dc24acd26bb8109f)

</div>

## 🎉 News

- **[2025/10/23]** 🔥🔥Our paper is accepted by NeurIPS 2025.🔥🔥
- **[2025/04/22]** Released our Paper on arXiv. See [here](https://arxiv.org/pdf/2504.15275)
- **[2025/03/24]** We re-implement our algorithm based on verl. ✨✨ Key features: **(1)** add ~50 additional metrics to comprehensively monitor the training process and stability, **(2)** add [a custom wandb workerspace](https://github.com/CJReinforce/PURE/tree/verl/verl/utils/logger/wandb_ws.py) to monitor ~20 important metrics, **(3)** add curriculum learning.✨✨
- **[2025/02/22]** We release the [notion blog](https://tungsten-ink-510.notion.site/Stop-Gamma-Decay-Min-Form-Credit-Assignment-Is-All-Process-Reward-Model-Needs-for-Reasoning-19fcb6ed0184804eb07fd310b38af155?pvs=4), which details our algorithm, the difference between gamma-decay and min-form credit assignment, examples of reward hacking, and so on.
- **[2025/02/09]** We release the training, evaluation code, [wandb](https://api.wandb.ai/links/cjreinforce/xvwk7pe9), and [checkpoints](https://huggingface.co/collections/jinachris/pure-67a85510dc24acd26bb8109f). Paper's on it's way!

## 📖 Introduction

This month, we saw a huge boost in LLM reasoning power from the verifiable reward (VR)-based Reinforcement learning fine-tuning (ReFT). Previous work has encountered challenges and made unsuccessful attempts in exploring PRM, so we wonder: How far can PRM actually take us? How does it stack up against VR-based methods in reasoning performance, training costs?

To answer these questions, we present **PURE** (**P**rocess-s**U**pervised **R**einforcement l**E**arning). Employing Qwen2.5-Math-7B as the base model, we train a PRM using PRM800K dataset, and then fine-tune another Qwen2.5-Math-7B model using only 8K MATH prompts, process rewards from the PRM, and optional verifiable rewards. **Our framework supports multiple reward types: only process reward (PURE-PRM), only verifiable reward (PURE-VR) which is the Deepseek-R1-zero's setting, or a mix of both (PURE-PRM+VR)**, as shown in the following table.

📊 The final model achieves pass@1 accuracy of 82.6% on MATH500, 82.5% on AMC, and 53.3% on average across 5 benchmarks, beating Qwen2.5-math-7B-instruct, PRIME, and SimpleRL with just either <1/50th RL data or 1/5th of the compute resources. 

***All results are in pass@1 accuracy***

|                            | AIME 2024 | MATH 500 | AMC      | Minerva Math | OlympiadBench | Avg.     |
| -------------------------- | --------- | -------- | -------- | ------------ | ------------- | -------- |
| Qwen2.5-Math-7B-Base       | 13.3      | 71.8     | 47.5     | 29.8         | 35.1          | 39.5     |
| Qwen-2.5-Math-7B-Instruct  | 16.7      | 83.2     | 52.5     | 37.5         | 41.3          | 46.2     |
| Eurus-2-7B-PRIME           | 26.7      | 79.2     | 57.8     | **38.6**     | 42.1          | 48.9     |
| Qwen2.5-7B-SimpleRL-Zero   | **33.3**  | 77.2     | 62.5     | 33.5         | 37.6          | 48.8     |
| **Qwen2.5-7B-PURE-PRM+VR***    | 20.0      | **82.6** | **82.5** | 37.1         | 44.1          | **53.3** |
| **Qwen2.5-7B-PURE-PRM**        | 16.7      | 81.8     | 60.0     | 38.2         | 44.7          | 49.3     |
| **Qwen2.5-7B-PURE-VR**         | 23.3      | 79.4     | 60.0     | 36.8         | 41.8          | 48.3     |

*The SOTA model was trained using 8K MATH problems, of which only ~800 gave ground-truth final answers that could be used to calculate VRs.

> Note: [Eurus-2-7B-PRIME](https://github.com/PRIME-RL/PRIME), and [SimpleRL-Zero](https://github.com/hkust-nlp/simpleRL-reason) are also based on Qwen-2.5-Math-7B.

## 🔧 Quick Start

We implement our algorithm on two frameworks, OpenRLHF and verl, in 2 different branches respectively. If you are new to our project, we recommend using verl version.

### Installation

#### 1. OpenRLHF version

Please follow [OpenRLHF's guidance](https://github.com/OpenRLHF/OpenRLHF/tree/main?tab=readme-ov-file#installation) to configure required environments. Then run `pip install -r requirements.txt`.

#### 2. verl version

Please refer to the [official installation guidance](https://verl.readthedocs.io/en/latest/start/install.html#install-from-custom-environment) of verl.

### Training of PRM

We train the PRM in 2 stages using [TRL](https://github.com/huggingface/trl) and a [preprocessed PRM800K dataset](https://huggingface.co/datasets/HuggingFaceH4/prm800k-trl-dedup). In the first stage, we freeze the LLM and only train the last score layer (MLP) with 1e-4 learning rate rate for 3 epochs. In the second stage, we unfreeze the LLM and fine-tune all parameters with 1e-6 learning rate for 1 epoch. The resultant PRM is released through [HuggingFace](https://huggingface.co/jinachris/Qwen2.5-Math-7B-PRM800K).

```bash
cd PRM
# stage 1
bash train_stage_1.sh
# stage 2
bash train_stage_2.sh
```

### Training of LLM

#### 1. OpenRLHF version

Switch to the [openrlhf branch](https://github.com/CJReinforce/PURE/tree/openrlhf). Run the following command. The parameter `reward_mode` in the [script](https://github.com/CJReinforce/PURE/blob/openrlhf/examples/scripts/train_pure.sh) controls the reward type and can be set to `PRM`, `VR`, and `PRMVR`.

```bash
bash examples/scripts/train_pure.sh
```

It uses Ray+vLLM for rollout acceleration, with the first 4 GPUs allocated for the actor, initial actor (reference model), and PRM. The remaining GPUs are used for the vLLM engines. This setup works with 5 to 8 GPUs—just adjust the number of vLLM engines in the script accordingly.

#### 2. verl version

Switch to the [verl branch](https://github.com/CJReinforce/PURE/tree/verl). Set the reward type in the [config file](verl/trainer/config/ppo_trainer.yaml):

1. `PURE-VR` uses `reward_model.enable=False reward_model.reward_manager=prime`
2. `PURE-PRM` uses `reward_model.enable=True reward_model.reward_manager=blank`
3. `PURE-PRM+VR` uses `reward_model.enable=True reward_model.reward_manager=prime`.

Then start training:

```bash
python -m verl.trainer.main_ppo
```

The hybrid engine of verl allows for higher gpu utilization compared to the openrlhf version.

### Evaluation

We use [Qwen Math's codebase](https://github.com/QwenLM/Qwen2.5-Math/tree/main/evaluation) for evaluation (i.e., pass@1 accuracy). For fairness considerations, we completely prohibited solving problems by calling code, following SimpleRL. Please follow the `/eval` instructions for evaluation.

## 📝 TODO:

- [x] re-implementation on [verl](https://github.com/volcengine/verl)
- [x] paper with more discussions and evaluations
- [ ] attempts to mitigate reward hacking for PRM (Online PURE)

## 🎈 Citation

If you find our code useful, we would appreciate it if you could cite our work:

```bibtex
@article{cheng2025stop,
  title={Stop Summation: Min-Form Credit Assignment Is All Process Reward Model Needs for Reasoning},
  author={Cheng, Jie and Qiao, Ruixi and Li, Lijun and Guo, Chao and Wang, Junle and Xiong, Gang and Lv, Yisheng and Wang, Fei-Yue},
  journal={arXiv preprint arXiv:2504.15275},
  year={2025}
}
```

## 🌻 Acknowledgement

We implement our RL algorithm based on [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) and [verl](https://github.com/volcengine/verl). We thank the developers of OpenRLHF and the author of SimpleRL for discussion! In addition, we also refer to [TRL](https://github.com/huggingface/trl), [PRIME](https://github.com/PRIME-RL/PRIME)'s code and hyperparameter values to varying degrees. Thank them for their wonderful work!
