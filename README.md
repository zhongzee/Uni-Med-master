# Uni-Med
The official codes for **Detecting Any instruction-to-answer interaction relationship:Universal Instruction-to-Answer Navigator for Med-VQA**


- [Uni-Med](#uni-med)
  - [Usage](#usage)
    - [1. Create Environment](#1-create-environment)
    - [2. Prepare Dataset](#2-prepare-dataset)
    - [3. Train](#3-train)
  - [Acknowledgement](#acknowledgement)


## Usage

### 1. Create Environment 

Please refer to https://github.com/chaoyi-wu/PMC-LLaMA

### 2. Prepare IAI-MED-Dataset 
#### 2.1 Generate Visual Explanation
1  Please refer to ./Uni/prompt/visual_explanation_prompt.py
#### 2.2 Label the real intent
2  Please refer to ./Uni/prompt/core_explain_and_instruct_mark.py

### 3. Train

Please refer to src/Uni-Med/train_downstream.sh

## Acknowledgement

CLIP -- https://github.com/openai/CLIP

PMC-CLIP -- https://github.com/WeixiongLin/PMC-CLIP

PMC-LLaMA -- [https://github.com/zphang/minimal-llama](https://github.com/chaoyi-wu/PMC-LLaMA)

LLaMA: Open and Efficient Foundation Language Models -- https://arxiv.org/abs/2302.13971

PMC-VQA: https://huggingface.co/datasets/xmcmic/PMC-VQA/

PASTA: https://github.com/QingruZhang/PASTA


We thank the authors for their open-sourced code and encourage users to cite their works when applicable.