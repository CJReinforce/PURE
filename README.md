<div align="center">

# Stop Gamma Decay: Min-Form Credit Assignment Is All Process Reward Model Needs for Reasoning

[![Notion](https://img.shields.io/badge/Notion-%23000000.svg?style=for-the-badge&logo=notion&logoColor=white)](https://tungsten-ink-510.notion.site/Stop-Gamma-Decay-Min-Form-Credit-Assignment-Is-All-Process-Reward-Model-Needs-for-Reasoning-19fcb6ed0184804eb07fd310b38af155?pvs=4)  [![Github](https://img.shields.io/badge/PURE-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/CJReinforce/PURE)  [![Wandb](https://img.shields.io/badge/Wandb_Report-fcd022?style=for-the-badge&logo=weightsandbiases&logoColor=000)](https://api.wandb.ai/links/cjreinforce/xvwk7pe9)  [![Wandb](https://img.shields.io/badge/Wandb_Log-fcd022?style=for-the-badge&logo=weightsandbiases&logoColor=000)](https://wandb.ai/cjreinforce/openrlhf_train_ppo/workspace?nw=nwuserchrisjina)  [![Hugging Face Collection](https://img.shields.io/badge/PURE_Collection-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/jinachris/pure-67a85510dc24acd26bb8109f)

</div>

**TL;DR:** Process Reward Model (PRM) trained on PRM800K dataset can fine-tune LLM to achieve SOTA-level mathematical reasoning capabilities for **< $150 (8 A100 GPUs * 16 hours)**.

## 🎉 News:

- [2025/03/24] We re-implement our algorithm based on verl. See [verl branch](https://github.com/volcengine/verl). 
- [2025/02/22] 🔥🔥We release the [notion blog](https://tungsten-ink-510.notion.site/Stop-Gamma-Decay-Min-Form-Credit-Assignment-Is-All-Process-Reward-Model-Needs-for-Reasoning-19fcb6ed0184804eb07fd310b38af155?pvs=4), which details our algorithm, the difference between gamma-decay and min-form credit assignment, examples of reward hacking, and so on🔥🔥
- [2025/02/09] We release the training, evaluation code, [wandb logs](https://wandb.ai/cjreinforce/openrlhf_train_ppo/workspace?nw=nwuserchrisjina), and [checkpoints](https://huggingface.co/collections/jinachris/pure-67a85510dc24acd26bb8109f). Paper's on it's way!

## 📖 Introduction

This month, we saw a huge boost in LLM reasoning power from the verifiable reward (VR)-based Reinforcement learning fine-tuning (ReFT), like [DeepSeek R1](https://github.com/deepseek-ai/DeepSeek-R1), [SimpleRL-Zero](https://github.com/hkust-nlp/simpleRL-reason), and [TinyZero](https://github.com/Jiayi-Pan/TinyZero). Previous work has encountered challenges and made unsuccessful attempts in exploring PRM, so we wonder: How far can PRM actually take us? How does it stack up against VR-based methods in reasoning performance, training costs?

To answer these questions, we present **PURE** (**P**rocess-s**U**pervised **R**einforcement l**E**arning). Using Qwen2.5-Math-7B as the base model, we train a PRM on 369k data from the PRM800K dataset, and then fine-tune another Qwen2.5-Math-7B model using only 8K MATH prompts, process rewards from the PRM, and optional verifiable rewards. For the RL algorithm, we use the PPO loss with an RLOO advantage estimator. We improve credit assignment by using a weighted sum of the process rewards, $\sum_t \text{softmax}(-\text{PR}_t/T)\cdot\text{PR}_t$ which approximates ${\min}_t \text{PR}_t$ when $T\rightarrow 0$, instead of the usual gamma decay sum $\sum_t \gamma^t \cdot \text{PR}_t$ to calculate return. **Our framework supports multiple reward types: only process reward (PURE-PRM), only verifiable reward (PURE-VR) which is the Deepseek-R1-zero's setting, or a mix of both (PURE-PRM+VR)**, as shown in the following table.

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

We implement our algorithm on two frameworks, OpenRLHF and verl, respectively. The `main` branch is based on openrlhf and the `verl` branch is based on verl.

### Installation

#### OpenRLHF version

Please follow [OpenRLHF's guidance](https://github.com/OpenRLHF/OpenRLHF/tree/main?tab=readme-ov-file#installation) to configure required environments. Then run `pip install -r requirements.txt`.

#### verl version

Please refer to the [official installation guidance](https://verl.readthedocs.io/en/latest/start/install.html#install-from-custom-environment) of verl.

### Training of PRM

We train the PRM in 2 stages using [TRL](https://github.com/huggingface/trl) and a [preprocessed PRM800K dataset](https://huggingface.co/datasets/HuggingFaceH4/prm800k-trl-dedup). In the first stage, we freeze the LLM and only train the last score layer (MLP) with 1e-4 learning rate rate for 3 epochs. In the second stage, we unfreeze the LLM and fine-tune all parameters with 1e-6 learning rate for 1 epoch. The resultant PRM is released through [HuggingFace](https://huggingface.co/jinachris/Qwen2.5-Math-7B-PRM800K).

```bash
# on main branch
cd PRM
# stage 1
bash train_stage_1.sh
# stage 2
bash train_stage_2.sh
```

### Training of LLM

#### OpenRLHF version

To start training, run the following command. It uses Ray+vLLM for rollout acceleration, with the first 4 GPUs allocated for the actor, initial actor (reference model), and PRM. The remaining GPUs are used for the vLLM engines. This setup works with 5 to 8 GPUs—just adjust the number of vLLM engines in the script accordingly.

```bash
bash examples/scripts/train_pure.sh
```

#### verl version

Switch to the [verl branch](https://github.com/volcengine/verl). Modify the `actor_rollout_ref.model.path`, `trainer.default_local_dir` in the [config file](verl/trainer/config/ppo_trainer.yaml). Then start training:

```bash
python -m verl.trainer.main_ppo
```

The single controller of verl allows for higher gpu utilization compared to the openrlhf version.

### Evaluation of Math Reasoning

We use [Qwen Math's codebase](https://github.com/QwenLM/Qwen2.5-Math/tree/main/evaluation) for evaluation (i.e., pass@1 accuracy). For fairness considerations, we completely prohibited solving problems by calling code, following SimpleRL. Please follow the `/eval` instructions for evaluation.

## :bulb: Discussions

See our [notion blog](https://tungsten-ink-510.notion.site/Stop-Gamma-Decay-Min-Form-Credit-Assignment-Is-All-Process-Reward-Model-Needs-for-Reasoning-19fcb6ed0184804eb07fd310b38af155?pvs=4) for more discussions.

### Reward Hacking

Reward hacking often occurs when relying solely on process rewards from the PRM, typically marked by sudden, extreme changes in metrics like reward, KL divergence, and loss. Meanwhile, the model starts only generating irrelevant outputs like "thanks" or "happy birthday" without any other string related to the questions. Since PRM is trained and inferences causally, such outputs can yield positive process scores, even though they are meaningless for math reasoning. You can see examples of this in the [wandb log](https://wandb.ai/cjreinforce/openrlhf_train_ppo/workspace?nw=nwuserchrisjina) with run name starting with "PRM_". 

In our experiments, reward hacking usually happened within the first 200 steps. However, before this occurs, the model performs well. For example, the Qwen2.5-7B-PURE-PRM model shown in the table above is saved at step 100, before hacking began.

Another factor that can trigger reward hacking is the baseline choice in RLOO. One intuitive way is to use the average process reward per step from other answers in the group as the baseline. However, this setting favors answers with fewer steps (refer to this [issue](https://github.com/CJReinforce/PURE/issues/2) for details). Since we split steps using a specific character (i.e., "\n\n"), we find the model sometimes avoids this character, producing answers with fewer steps but excessively long tokens per step. The PRM struggles to assign accurate process rewards to such lengthy steps. To address this,  we change the baseline to the average reward per token from other answers, multiplied by the number of tokens in the current step. This improvement penalizes steps with more tokens more heavily and removes the bias toward fewer steps.

### Aha Moment

Unfortunately, we did not observe the aha moment, self-reflection, or long CoT for schemes using PRM. We suppose that even if an answer like "\<response A\> Wait, wait. \<response B\>" is generated in the rollout, the PRM will assign negative process rewards to response A and positive process rewards to response B. The PPO algorithm then probably decrease the sampling probability of response A, and increase the sampling probability of response B, resulting in the final model that just outputs response B and thus no aha moment.

## 📝 TODO:

- [x] re-implementation on [verl](https://github.com/volcengine/verl) (see `verl` [branch](https://github.com/CJReinforce/PURE/tree/verl))
- [ ] paper with more discussions and evaluations
- [ ] attempts to mitigate reward hacking for PRM (Online PURE)

## 🎈 Citation

If you find our code useful, we would appreciate it if you could cite our work:

```bibtex
@misc{cheng2025pure,
  title={Stop Gamma Decay: Min-Form Credit Assignment Is All Process Reward Model Needs for Reasoning},
  author={Jie Cheng and Lijun Li and Gang Xiong and Jing Shao and Yisheng Lv and Fei-Yue Wang},
  year={2025},
  howpublished={\url{https://tungsten-ink-510.notion.site/Stop-Gamma-Decay-Min-Form-Credit-Assignment-Is-All-Process-Reward-Model-Needs-for-Reasoning-19fcb6ed0184804eb07fd310b38af155?pvs=4}},
  note={Notion Blog}
  year={2025}
}
```

## 🌻 Acknowledgement

We implement our RL algorithm based on [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) and [verl](https://github.com/volcengine/verl). We thank the developers of OpenRLHF and the author of SimpleRL for discussion! In addition, we also refer to [TRL](https://github.com/huggingface/trl), [PRIME](https://github.com/PRIME-RL/PRIME)'s code and hyperparameter values to varying degrees. Thank them for their wonderful work!
