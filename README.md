<div align="center">

# PURE: PRM is still Effective and Compute-Efficient for LLM Math Reasoning

[![Github](https://img.shields.io/badge/PURE-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/CJReinforce/PURE)  [![Wandb](https://img.shields.io/badge/Wandb_Log-fcd022?style=for-the-badge&logo=weightsandbiases&logoColor=000)](https://api.wandb.ai/links/cjreinforce/xvwk7pe9)  [![Hugging Face Collection](https://img.shields.io/badge/PURE_Collection-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/jinachris/pure-67a85510dc24acd26bb8109f)

</div>

**TL;DR:** Starting from Qwen2.5-Math-7B, Process Reward Model (PRM) trained on PRM800K dataset can fine-tune LLM to achieve superior mathematical reasoning capabilities that are comparable to previous baselines, using **only 8 A100 GPUs within 1 day**.

## 🎉 News:

- [2025/02/09] We release the training, evaluation code, [wandb log](https://api.wandb.ai/links/cjreinforce/xvwk7pe9), and [checkpoints](https://huggingface.co/collections/jinachris/pure-67a85510dc24acd26bb8109f). Paper's on it's way!

## 📖 Introduction

This month, we saw a huge boost in LLM reasoning power from the verifiable reward (VR)-based Reinforcement learning fine-tuning (ReFT), like [DeepSeek R1](https://github.com/deepseek-ai/DeepSeek-R1), [SimpleRL-Zero](https://github.com/hkust-nlp/simpleRL-reason), and [TinyZero](https://github.com/Jiayi-Pan/TinyZero). Previous work has encountered challenges and made unsuccessful attempts in exploring PRM, so we wonder: How far can PRM actually take us? How does it stack up against VR-based methods in reasoning performance, training costs?

To answer these questions, we present **PURE** (**P**rocess-s**U**pervised **R**einforcement l**E**arning). Using Qwen2.5-Math-7B as the base model, we train a PRM on 369k data from the PRM800K dataset, and then fine-tune another Qwen2.5-Math-7B model using only 8K MATH prompts, process rewards from the PRM, and optional verifiable rewards. For the RL algorithm, we use the PPO loss with an RLOO advantage estimator. We improve credit assignment by using a weighted sum of the process rewards, $\sum_t \text{softmax}(-\text{PR}_t/T)\cdot\text{PR}_t$ which approximates ${\min}_t \text{PR}_t$ when $T\rightarrow 0$, instead of the usual gamma decay sum $\sum_t \gamma^t \cdot \text{PR}_t$ to calculate return.

The final model achieves 81.4% on MATH500, 72.5% on AMC, and 50.9% on average across 5 benchmarks, beating Qwen2.5-math-7B-instruct and SimpleRL-Zero with just 1/5th of the compute resources. Our framework supports multiple reward types: process reward (PURE-PRM), verifiable reward (PURE-VR), or a mix of both (PURE-PRM+VR), as shown in the following table.

***All results are in pass@1 accuracy***


|                            | AIME 2024 | MATH 500 | AMC      | Minerva Math | OlympiadBench | Avg.     |
| -------------------------- | --------- | -------- | -------- | ------------ | ------------- | -------- |
| Qwen2.5-Math-7B-Base       | 16.7      | 52.4     | 52.5     | 12.9         | 16.4          | 30.2     |
| Qwen-2.5-Math-7B-Instruct  | 13.3      | 79.8     | 50.6     | 34.6         | 40.7          | 43.8     |
| rStar-Math-7B              | 26.7      | 78.4     | 47.5     | -            | **47.1**      | -        |
| Eurus-2-7B-PRIME           | 26.7      | 79.2     | 57.8     | **38.6**     | 42.1          | 48.9     |
| Qwen2.5-7B-SimpleRL-Zero   | **33.3**  | 77.2     | 62.5     | 33.5         | 37.6          | 48.8     |
| Qwen2.5-7B-PURE-PRM+VR     | 16.7      | **81.4** | **72.5** | **38.6**     | 45.5          | **50.9** |
| Qwen2.5-7B-PURE-PRM        | 16.7      | 81.0     | 60.0     | 37.5         | 43.4          | 47.7     |
| Qwen2.5-7B-PURE-VR         | 16.7      | 77.0     | 60.0     | 37.5         | 40.1          | 46.3     |

> Note: [rStar-Math-7B](https://arxiv.org/abs/2501.04519), [Eurus-2-7B-PRIME](https://github.com/PRIME-RL/PRIME), and [SimpleRL-Zero](https://github.com/hkust-nlp/simpleRL-reason) are also based on Qwen-2.5-Math-7B.

***Data and GPUs comparison of different approaches***


|                | Qwen2.5-Math-7B-Instruct        | rStar-Math-7B                  | Eurus-2-7B-PRIME         | Qwen2.5-7B-SimpleRL-Zero | Qwen2.5-7B-PURE         |
| -------------- | ------------------------------- | ------------------------------ | ------------------------ | ------------------------ | ----------------------- |
| **Base Model** | Qwen2.5-Math-7B                 | Qwen2.5-Math-7B                | Qwen2.5-Math-7B          | Qwen2.5-Math-7B          | Qwen2.5-Math-7B         |
| **SFT Data**   | 2.5M (open-source and in-house) | ~7.3M (MATH, NuminaMath, etc.) | 230K                     | 0                        | 0                       |
| **RM Data**    | 618K (in-house)                 | ~7k (in-house)                 | 0                        | 0                        | 369k (open-source)      |
| **RM**         | Qwen2.5-Math-RM (72B)           | None                           | Eurus-2-7B-SFT           | None                     | [Qwen2.5-Math-7B-PRM800K](https://huggingface.co/jinachris/Qwen2.5-Math-7B-PRM800K) |
| **RL Data**    | 66K queries × 32 samples        | ~3.647M × 16                   | 150K queries × 4 samples | 8K queries × 8 samples   | 8K queries × 4 samples  |
| **GPUs**       | -                               | 80 H100 at most                | 8 A100                   | 40 A100                  | 8 A100                  |

## 🔧 Quick Start

### Installation

Our code is implemented based on OpenRLHF. Please follow [OpenRLHF's guidance](https://github.com/OpenRLHF/OpenRLHF/tree/main?tab=readme-ov-file#installation) to configure required environments. Then run `pip install -r requirements.txt`

### Training of PRM

We train the PRM in 2 stages using [TRL](https://github.com/huggingface/trl) and a [preprocessed PRM800K dataset](https://huggingface.co/datasets/HuggingFaceH4/prm800k-trl-dedup). In the first stage, we freeze the LLM and only train the last score layer (MLP) with 1e-4 learning rate rate for 3 epochs. In the second stage, we unfreeze the LLM and fine-tune all parameters with 1e-6 learning rate for 1 epoch. The resultant PRM is released through [HuggingFace](https://huggingface.co/jinachris/Qwen2.5-Math-7B-PRM800K).

```bash
cd PRM
# stage 1
bash train_stage_1.sh
# stage 2
bash train_stage_2.sh
```

We evaluate our PRM using BoN method, [ProcessBench](https://arxiv.org/abs/2412.06559), and [PRMBench](https://arxiv.org/abs/2501.03124). 

- For BoN, we use the data from [RLHFlow](https://github.com/RLHFlow/RLHF-Reward-Modeling/tree/main/math-rm). With $N=1024$ answers generated by Deepseek-7B for each question, we use our PRM to calculate process rewards and then aggregate them into an outcome reward for each answer. The answer with the highest outcome reward is selected as the final answer. Our PRM achieves pass@1024 scores of 91.6% on GSM8K and 62.6% on MATH, compared to RLHFlow's best scores of 93.0% and 58.1%, respectively.

| Generator Model         | Method                    | GSM8K | MATH |
| ------------- | ------------- | ------------- | -------- |
| Deepseek-7B | Pass@1 | 83.9 | 42.4 |
| Deepseek-7B | Majority Voting@1024 | 89.7 | 57.4  |
| Deepseek-7B | Deepseek-PRM@1024 | **93.0** | 58.1 |
| Deepseek-7B | Mistral-PRM@1024 (OOD) | 91.9 | 56.9 |
| Deepseek-7B | Our-PRM@1024 (OOD) | 91.6 | **62.6** |

- On ProcessBench, which tests the PRM's ability to identify the first process error, our PRM scores an average F1 of 57.5, outperforming the best PRM's F1 score of 56.5 in ProcessBench. 

| Process Reward Model    | GSM8K    | MATH     | OlympiadBench | OmniMATH | Average  |
| ----------------------- | -------- | -------- | ------------- | -------- | -------- |
| Math-Shepherd-PRM-7B    | 47.9     | 29.5     | 24.8          | 23.8     | 31.5     |
| RLHFlow-PRM-Mistral-8B  | 50.4     | 33.4     | 13.8          | 15.8     | 28.4     |
| RLHFlow-PRM-Deepseek-8B | 38.8     | 33.8     | 16.9          | 16.9     | 26.6     |
| Skywork-PRM-7B          | **70.8** | 53.6     | 22.9          | 21.0     | 42.1     |
| Qwen2.5-Math-7B-PRM800K | 68.2     | 62.6     | **50.7**      | 44.3     | 56.5     |
| Our PRM-7B              | 69.0     | **66.5** | 48.4          | **45.9** | **57.5** |

- On PRMBench, which is designed to assess the fine-grained error detection capabilities of PRMs, our PRM gets an overall score of 65.3, ranking 🥉third among open source PRMs. You can find our PRM named `Pure-PRM-7B`	on the [official leaderboard](https://prmbench.github.io/).

These results confirm that our PRM is SOTA and suitable for fine-tuning LLMs.

### Training of LLM

To start training, run the following command. It uses Ray+vLLM for rollout acceleration, with the first 4 GPUs allocated for the actor, initial actor (reference model), and PRM. The remaining GPUs are used for the vLLM engines. This setup works with 5 to 8 GPUs—just adjust the number of vLLM engines in the script accordingly.

```bash
bash examples/scripts/train_pure.sh
```

### Evaluation of Math Reasoning

We used [Qwen Math's codebase](https://github.com/QwenLM/Qwen2.5-Math/tree/main/evaluation) for evaluation (i.e., pass@1 accuracy). For fairness considerations, we completely prohibited solving problems by calling code, following SimpleRL. Please follow the `/eval` instructions for evaluation.

## :bulb: Discussions

### Reward Hacking

Reward hacking often occurs when relying solely on process rewards from the PRM, typically marked by sudden, extreme changes in metrics like reward, KL divergence, and loss. Meanwhile, the model starts only generating irrelevant outputs like "thanks" or "happy birthday" without any other string related to the questions. Since PRM is trained and inferences causally, such outputs can yield positive process scores, even though they are meaningless for math reasoning. You can see examples of this in the [wandb log](https://api.wandb.ai/links/cjreinforce/xvwk7pe9) with run name starting with "PRM_". 

In our experiments, reward hacking usually happened within the first 200 steps. However, before this occurs, the model performs well. For example, the Qwen2.5-7B-PURE-PRM model shown in the table above is saved at step 100, before hacking began.

Another factor that can trigger reward hacking is the baseline choice in RLOO. One intuitive way is to use the average process reward per step from other answers in the group as the baseline. However, this setting favors answers with fewer steps. Since we split steps using a specific character (i.e., "\n\n"), we find the model sometimes avoids this character, producing answers with fewer steps but excessively long tokens per step. The PRM struggles to assign accurate process rewards to such lengthy steps. To address this,  we change the baseline to the average reward per token from other answers, multiplied by the number of tokens in the current step. This improvement penalizes steps with more tokens more heavily and removes the bias toward fewer steps.

### Aha Moment

Unfortunately, we did not observe the aha moment, self-reflection, or long CoT for schemes using PRM. We suppose that even if an answer like "\<response A\> Wait, wait. \<response B\>" is generated in the rollout, the PRM will assign negative process rewards to response A and positive process rewards to response B. The PPO algorithm then probably decrease the sampling probability of response A, and increase the sampling probability of response B, resulting in the final model that just outputs response B and thus no aha moment.

## 📝 TODO:

- [ ] paper with more discussions and evaluations
- [ ] attempts to mitigate reward hacking for PRM (Online PURE).

## 🎈 Citation

If you find our code useful, we would appreciate it if you could cite our work:

```bibtex
@misc{cheng2025pure,
  title={PURE: PRM is still Effective and Compute-Efficient for LLM Math Reasoning},
  author={Jie Cheng, Lijun Li, Gang Xiong, Jing Shao, Yisheng Lv},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished={\url{https://github.com/CJReinforce/PURE}},
  year={2025}
}
```

## 🌻 Acknowledgement

We implement our RL algorithm based on [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF). We thank the developers of OpenRLHF and the author of SimpleRL for discussion! In addition, we also refer to [TRL](https://github.com/huggingface/trl), [PRIME](https://github.com/PRIME-RL/PRIME)'s code and hyperparameter values to varying degrees. Thank them for their wonderful work!

<details>
<summary> Random Thoughts</summary>

I’m very happy about DeepSeek's great success and extremely grateful for their selflessly open-sourced models. Their success has also brought a tremendous level of attention and expectations to VR-based ReFT methods. However, I still want to look back and see where exactly the PRM path can take us. As Tim Berners-Lee once said, "We need diversity of thought in the world to face new challenges." Perhaps, when the conversation becomes longer or the number of steps increases, VR/ORM-based ReFT methods may underperform due to sparse rewards or credit assignment issues?

Since late '23, I've wanted to use LLMs to solve long-range decision-making and reasoning tasks. I tried to play chess using Qwen and Llama, but it was too difficult for their abilities at that time (and still is, even now), and I saw no hope of success with ReFT using such base policies. In mid-'24, I aimed to use an open-sourced model to solve GitHub issues, which is what SWE-Bench evaluated. I was able to obtain a good base policy through prompt engineering, but the conversations were too long, coupled with very sparse rewards, so I still couldn’t get ReFT to work. Now, with sufficient datasets and sufficiently capable base models for mathematical reasoning tasks, I’m finally able to implement the idea I had two years ago. I’m grateful to the open-source community for giving me the opportunity to achieve this goal!

This project took me about a week, and there are still many imperfections. In the future, in the paper, we plan to include more experiments to comprehensively discuss the differences between the PRM and VR approaches. I hope you can understand and sympathize with the current limitations.

</details>
