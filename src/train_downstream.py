import argparse
import os
import json
import math
import tqdm.auto as tqdm
from typing import Optional
import transformers
from Dataset.Slake_Dataset import Slake_Dataset
from Dataset.VQA_RAD_Dataset import VQA_RAD_Dataset
from models.QA_model import QA_model
from transformers import Trainer,TrainerCallback
from dataclasses import dataclass, field
import time
import os
from torch.utils.data import DataLoader  
import torch
import wandb
# /root/autodl-tmp/PMC_LLAMA_7B

# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)
# /root/autodl-tmp/Uni_Results/QA_no_pretrain_no_aug_test_epoch100_choise/VQA_RAD/checkpoint-6128
# /root/autodl-tmp/MedVInT-TD/VQA_lora_PMC_LLaMA_PMCCLIP/choice/checkpoint-4000
@dataclass
class ModelArguments:
    model_path: Optional[str] = field(default="/root/autodl-tmp/LLama-7B-hf-orin")
    ckp: Optional[str] = field(default="/root/autodl-tmp/Uni_Results/QA_no_pretrain_no_aug_test_epoch100_choise/VQA_RAD/checkpoint-6128")#如果是在PMC-VQA上就可以不加载 下游任务微调就要


    ## Q_former ##
    N: Optional[int] = field(default=12)
    H: Optional[int] = field(default=8)
    img_token_num: Optional[int] = field(default=32) 
    ## Basic Setting ##
    voc_size: Optional[int] = field(default=32000)
    hidden_dim: Optional[int] = field(default=4096) # 13B:32001 5120 7B 32000 4096
    checkpointing: Optional[bool] = field(default=True)#
    ## Image Encoder ##
    Vision_module: Optional[str] = field(default='PMC-CLIP')
    visual_model_path: Optional[str] = field(default='/root/autodl-tmp/pmc_clip/checkpoint.pt')
    #visual_model_config: Optional[str] = field(default='./img_checkpoint/RN50_fusion4.json')
    ## Peft ##
    is_lora: Optional[bool] = field(default=True)
    peft_mode: Optional[str] = field(default="lora")
    lora_rank: Optional[int] = field(default=8)
# autodl-tmp/VQA_RAD/trainset_llava_COT.csv
# autodl-tmp/VQA_RAD/trainset.csv
@dataclass
class DataArguments:
    Train_csv_path: str = field(default='/root/autodl-tmp/VQA_RAD/trainset_llava_COT.csv', metadata={"help": "Path to the training data."})
    Eval_csv_path: str = field(default='/root/autodl-tmp/VQA_RAD/testset.csv', metadata={"help": "Path to the training data."})
    tokenizer_path: str = field(default='/root/autodl-tmp/LLama-7B-hf-orin', metadata={"help": "Path to the training data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    
@dataclass
class CustomTrainingArguments(transformers.TrainingArguments):
    n_gpu: int = field(default=1, metadata={"help": "Number of GPUs."})

class ThroughputCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = time.time()
        self.num_examples = 0

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.train_start_time
        throughput = self.num_examples / total_time
        print(f"Overall training throughput: {throughput:.2f} examples/second over {args.n_gpu} GPU(s)")

    def on_step_end(self, args, state, control, **kwargs):
        # 每个step结束时更新处理的example数量
        # args.per_device_train_batch_size 是每个设备的batch大小
        # args.gradient_accumulation_steps 是梯度累积的步骤
        # args.n_gpu 是使用的GPU数量
        self.num_examples += (args.per_device_train_batch_size * args.gradient_accumulation_steps * args.n_gpu)


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    print("Setup Data")
    if 'VQA_RAD' in data_args.Train_csv_path:
        training_args.run_name = training_args.run_name + '_VQA_RAD'
        training_args.output_dir = training_args.output_dir + '/VQA_RAD'
        Train_dataset = VQA_RAD_Dataset(data_args.Train_csv_path, data_args.tokenizer_path, text_type = 'choice')
        Eval_dataset = VQA_RAD_Dataset(data_args.Eval_csv_path, data_args.tokenizer_path, text_type = 'choice')
    if 'Slake1.0' in data_args.Train_csv_path:
        training_args.run_name = training_args.run_name + '_Slake'
        training_args.output_dir = training_args.output_dir + '/Slake'
        Train_dataset = Slake_Dataset(data_args.Train_csv_path, data_args.tokenizer_path, text_type = 'choice')
        Eval_dataset = Slake_Dataset(data_args.Eval_csv_path, data_args.tokenizer_path, text_type = 'choice')

    print("Setup Model")
    ckp = model_args.ckp + '/pytorch_model.bin'
    print(ckp)
    model = QA_model(model_args)
    # print("Loading Pre-train Model")
    # model.load_state_dict(torch.load(ckp, map_location='cpu'))
    print('Start training')
    start_time = time.time()
    # trainer = Trainer(model=model,
    #                   train_dataset = Train_dataset,
    #                   eval_dataset = Eval_dataset,
    #                   args=training_args,
    #                   )

    # 请确保指定n_gpu
    # training_args.n_gpu = 1

    # 创建Trainer对象
    trainer = Trainer(
        model=model,
        train_dataset=Train_dataset,
        eval_dataset=Eval_dataset,
        args=training_args,
        callbacks=[ThroughputCallback()]  # 添加ThroughputCallback
    )
    # 计时结束

    trainer.train()
    end_time = time.time()

    # 计算吞吐量
    total_time = end_time - start_time  # 总时间
    total_batches = len(Train_dataset)  # 总批次数
    throughput = total_batches / total_time  # 吞吐量（批次/秒）

    # total_time = end_time - start_time  # 总时间
    # total_batches = 10 # 总批次数
    # throughput = total_batches / total_time  # 吞吐量（批次/秒）

    print(f"Training Throughput: {throughput} batches/second")

    trainer.save_state()
    
if __name__ == "__main__":
    main()
"""
--Train_csv_path "/root/autodl-tmp/VQA_RAD/Task_COT_New_related_final.csv" --Eval_csv_path "/root/autodl-tmp/VQA_RAD/testset.csv" --output_dir ./Results/VQA_RAD_pretrain_no_aug --run_name VQA_RAD_pretrain_no_aug --num_train_epochs 100 --per_device_train_batch_size 2 --per_device_eval_batch_size 8 --gradient_accumulation_steps 1 --evaluation_strategy "epoch" --save_strategy "epoch" --load_best_model_at_end True --save_total_limit 2 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --deepspeed ./ds_config/ds_config_zero2.json --checkpointing false --bf16 True --tf32 True


"""