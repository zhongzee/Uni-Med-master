import argparse
import os
import json

import ipdb
import math
import tqdm.auto as tqdm
from typing import Optional
import transformers
# from Dataset.VQA_RAD_Dataset import VQA_RAD_Dataset
from Dataset.VQA_RAD_Dataset_cut import VQA_RAD_Dataset_cut
from models.QA_model import QA_model
from transformers import Trainer
from dataclasses import dataclass, field
import os
from torch.utils.data import DataLoader
import torch
import numpy as np
import difflib
import csv


# /root/autodl-tmp/MedVInT-TD/VQA_lora_PMC_LLaMA_PMCCLIP/choice/checkpoint-4000
# Uni-Med/src/Uni/Results/QA_no_pretrain_no_aug_test1/VQA_RAD/checkpoint-766
# /root/autodl-tmp/Uni_Results/QA_no_pretrain_no_aug_test_epoch100_choise/VQA_RAD/checkpoint-6128
@dataclass
class ModelArguments:
    model_path: Optional[str] = field(default="/root/autodl-tmp/LLama-7B-hf-orin")
    ckp: Optional[str] = field(
        default="/root/autodl-tmp/Uni_Results/QA_no_pretrain_no_aug_test_epoch100_choise/VQA_RAD/checkpoint-6128")  #
    checkpointing: Optional[bool] = field(default=False)
    ## Q_former ##
    N: Optional[int] = field(default=12)
    H: Optional[int] = field(default=8)
    img_token_num: Optional[int] = field(default=32)

    ## Basic Setting ##
    voc_size: Optional[int] = field(default=32000)
    hidden_dim: Optional[int] = field(default=4096)

    ## Image Encoder ##
    Vision_module: Optional[str] = field(default='PMC-CLIP')
    visual_model_path: Optional[str] = field(default='/root/autodl-tmp/pmc_clip/checkpoint.pt')

    ## Peft ##
    is_lora: Optional[bool] = field(default=True)
    peft_mode: Optional[str] = field(default="lora")
    lora_rank: Optional[int] = field(default=8)


@dataclass
class DataArguments:
    img_dir: str = field(default='/root/autodl-tmp/VQA_RAD/cut-mix2/ABD/ABN/', metadata={"help": "Path to the training data."})
    Test_csv_path: str = field(default='/root/autodl-tmp/VQA_RAD/cut-mix2/ABD/ABN/ABN_data.csv',#  # /root/autodl-tmp/VQA_RAD/test_close.csv
                               metadata={"help": "Path to the training data."})
    tokenizer_path: str = field(default='/root/autodl-tmp/LLama-7B-hf-orin',
                                metadata={"help": "Path to the training data."})
    trier: int = field(default=0)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(default="./Results")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")


def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()


def find_most_similar_index(str_list, target_str):
    """
    Given a list of strings and a target string, returns the index of the most similar string in the list.
    """
    # Initialize variables to keep track of the most similar string and its index
    most_similar_str = None
    most_similar_index = None
    highest_similarity = 0

    # Iterate through each string in the list
    for i, str in enumerate(str_list):
        # Calculate the similarity between the current string and the target string
        similarity = str_similarity(str, target_str)

        # If the current string is more similar than the previous most similar string, update the variables
        if similarity > highest_similarity:
            most_similar_str = str
            most_similar_index = i
            highest_similarity = similarity

    # Return the index of the most similar string
    return most_similar_index


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print("Setup Data")
    row_count = 0
    # if os.path.exists('result_final'+str(data_args.trier)+'.csv'):

    #     with open('result_final'+str(data_args.trier)+'.csv', 'r') as file:
    #         reader = csv.reader(file)
    #         row_count = sum(1 for row in reader)-1
    Test_dataset_close = VQA_RAD_Dataset_cut(data_args.Test_csv_path, data_args.tokenizer_path, mode='Test',
                                         text_type='blank', start=row_count)

    # batch size should be 1
    Test_dataloader_close = DataLoader(
        Test_dataset_close,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        sampler=None,
        shuffle=False,
        collate_fn=None,
        drop_last=False,
    )
    Test_dataset_open = VQA_RAD_Dataset_cut(data_args.Test_csv_path.replace('close.csv', 'open.csv'),
                                        data_args.tokenizer_path, mode='Test', text_type='blank', start=row_count)

    # # batch size should be 1
    # Test_dataloader_open = DataLoader(
    #     Test_dataset_open,
    #     batch_size=1,
    #     num_workers=1,
    #     pin_memory=True,
    #     sampler=None,
    #     shuffle=False,
    #     collate_fn=None,
    #     drop_last=False,
    # )

    model = QA_model(model_args).to('cuda')
    # closed_save_path = 'result_final_greedy_VQA_RAD_close' + model_args.ckp.split('/')[-3] + '_' + \
    #                    model_args.ckp.split('/')[-2] + '.csv'

    closed_save_path = 'group_ABN_VQA_RAD_close' + model_args.ckp.split('/')[-3] + '_' + \
                       model_args.ckp.split('/')[-2] + '.csv'

    import tqdm

    # 初始化tokenizer和模型
    tokenizer = Test_dataset_close.tokenizer
    model = model.to('cuda')

    # 准备CSV文件以写入结果
    with open(closed_save_path, mode='w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Figure_path', 'Question', 'Pred', 'Label'])

        # 外层循环：遍历每个问答对
        for base_idx in tqdm.tqdm(range(len(Test_dataset_close))):
            base_sample = Test_dataset_close[base_idx] #

            # 准备基础问答对的输入数据
            base_input_ids = tokenizer(base_sample['input_ids'], return_tensors="pt").to('cuda')
            base_input_ids['input_ids'][0][0] = 1
            base_images = base_sample['images'].unsqueeze(0).to('cuda')  # 添加batch维度

            # 内层循环：遍历除了基础问答对之外的所有问答对
            for mix_idx in range(len(Test_dataset_close)):
                if mix_idx == base_idx:  # 跳过基础问答对自身
                    continue
                mix_sample = Test_dataset_close[mix_idx]

                # 准备混合问答对的输入数据
                mix_input_ids = tokenizer(mix_sample['input_ids'], return_tensors="pt").to('cuda')
                mix_input_ids['input_ids'][0][0] = 1
                mix_images = mix_sample['images'].unsqueeze(0).to('cuda')  # 添加batch维度

                # 执行visual_cutmix操作
                with torch.no_grad():
                    # 这里假设visual_cutmix方法可以处理两组图像和文本
                    generation_ids = model.visual_cutmix([base_input_ids['input_ids'], mix_input_ids['input_ids']],
                                                         [base_images, mix_images], [base_sample, mix_sample],
                                                         tokenizer)

                # 处理生成的结果并写入CSV文件
                # ...

    # with open(closed_save_path, mode='w') as outfile:
    #         writer = csv.writer(outfile)
    #         writer.writerow(['Figure_path','Question','Pred','Label'])
    #         tokenizer = Test_dataset_close.tokenizer
    #         for sample in tqdm.tqdm(Test_dataloader_close):
    #             # ipdb.set_trace()
    #             input_ids = Test_dataset_close.tokenizer(sample['input_ids'],return_tensors="pt").to('cuda')
    #             input_ids['input_ids'][0][0]=1
    #             # images = sample['images'].to('cuda')
    #             model = model.to('cuda')
    #             images = sample['images'].to('cuda')
    #             with torch.no_grad():
    #                 # generation_ids = model.generate_long_sentence(input_ids['input_ids'],images)#
    #                 generation_ids = model.visual_cutmix(input_ids['input_ids'], images, sample,tokenizer)
    #
    #                 # generation_ids = model.generate_long_sentence(input_ids['input_ids'], images)
    #             generated_texts = Test_dataset_close.tokenizer.batch_decode(generation_ids, skip_special_tokens=True)
    #             for i in range(len(generated_texts)):
    #                 label = sample['labels'][i]
    #                 img_path = sample['img_path'][i]
    #                 pred = generated_texts[i]
    #                 writer.writerow([img_path,sample['input_ids'][i],pred,label])
    #                 cc = cc + 1

if __name__ == "__main__":
    #os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    main()