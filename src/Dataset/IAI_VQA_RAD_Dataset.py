import torch
from torch.utils.data import Dataset
from torchvision import transforms
import json
import PIL
import numpy as np
import torch.nn.functional as F
import transformers
import pandas as pd
import random
import copy
from .randaugment import RandomAugment    
from PIL import Image
import tqdm
import ipdb
import re

# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)

class IAI_VQA_RAD_Dataset(Dataset):
    def __init__(self , csv_path, tokenizer_path, img_dir='/root/autodl-tmp/VQA_RAD/', img_tokens = 32, seq_length = 512,voc_size = 32000, mode = 'Train',start = 0,text_type = 'choice'):
        self.img_root = img_dir
        self.data = pd.read_csv(csv_path).iloc[start:]
        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token_id=0
        self.tokenizer.eos_token_id=1
        self.mode = mode
        self.img_padding = [-100 for i in range(img_tokens)]
        self.attn_padding = [1 for i in range(img_tokens)]
        self.H = 512
        self.W = 512
        self.C = 3
        self.text_type = text_type
        
        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.transform = transforms.Compose([                        
                transforms.RandomResizedCrop((self.H,self.W),scale=(0.2, 1.0), interpolation=Image.BICUBIC),
                #transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',]),     
                transforms.ToTensor(),
                normalize,
            ]) 
        if self.mode == 'Test':
            self.transform = transforms.Compose([                        
                    transforms.Resize((self.H,self.W), interpolation=Image.BICUBIC),
                    transforms.ToTensor(),
                    normalize,
                ])
            
        self.mode = mode
        self.seq_length = seq_length
        self.voc_size = voc_size
        
    def __len__(self):
        return len(self.data)
    
    def random_answer(self, Question,Answer):
        ipdb.set_trace()
        Answer = str(Answer)
        pre_text = 'Question: '+ Question
        # pre_text = 'Question: ' + Question + 'The Answer is:' # 原本
        # final_o = 'Question: '+ Question +'The Answer is:' + Answer （原本）
        # final_o = 'Question: ' + Question + "COT based Explanation is " + 'The Final Answer is:'  +Answer
        final_o = 'Question: ' + Question + Answer
        return pre_text,final_o

    def parse_and_restore_text(self, question, answer):
        # ipdb.set_trace()
        # 正则表达式用于匹配和提取被标记的文本
        task_related_pattern = r"<task_related>(.*?)<\/task_related>"
        task_related_visual_pattern = r"<task_related_visual>(.*?)<\/task_related_visual>"

        # 提取被标记的文本
        task_related_text = re.findall(task_related_pattern, question)
        task_related_visual_text = re.findall(task_related_visual_pattern, answer)

        # 还原文本（去除标记）
        restored_question = re.sub(task_related_pattern, r"\1", question)
        restored_answer = re.sub(task_related_visual_pattern, r"\1", answer)

        return restored_question, restored_answer, task_related_text, task_related_visual_text

    def tokenize_and_mark(self, question, answer, task_related, task_related_visual):
        # This method will tokenize the question and answer and mark the task-related sets
        # First, tokenize the question and answer as usual
        encoded_inputs = self.tokenizer(question, answer, padding='max_length', truncation=True,
                                        max_length=self.seq_length)
        input_ids = encoded_inputs['input_ids']
        attention_mask = encoded_inputs['attention_mask']

        # Now, mark the task-related and task-related-visual tokens
        task_related_ids = self.tokenizer(' '.join(task_related), add_special_tokens=False)['input_ids']
        task_related_visual_ids = self.tokenizer(' '.join(task_related_visual), add_special_tokens=False)['input_ids']

        return input_ids, attention_mask, task_related_ids, task_related_visual_ids
        # You can now integrate these marked ids into your token-level CutMix strategy

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        Question = sample['question']
        Answer = sample['answer']

        # 解析并恢复文本
        # restored_question, restored_answer, task_related_text, task_related_visual_text = self.parse_and_restore_text(
        #     Question, Answer)

        # 读取图像
        img_path = self.img_root + sample['image_name']
        img = PIL.Image.open(img_path).convert('RGB')
        image = self.transform(img)

        # 分词并标记
        # input_ids, attention_mask, task_related_ids, task_related_visual_ids = self.tokenize_and_mark(
        #     restored_question, restored_answer, task_related_text, task_related_visual_text)
        restored_question, restored_answer, task_related_text, task_related_visual_text = self.parse_and_restore_text(
            Question, Answer)

        # 根据模式处理数据
        if self.mode == 'Train':
            # 随机答案处理（示例实现，可能需要根据具体需求修改）
            pre_text, final_o = self.random_answer(restored_question, restored_answer)
            # pre_text, final_o = self.random_answer(Question, Answer)
            final_o_encoded = self.tokenizer(final_o)
            input_ids = final_o_encoded['input_ids']
            input_ids.append(self.tokenizer.eos_token_id)
            input_ids = np.array(input_ids)
            if len(input_ids) < self.seq_length:
                input_ids = np.pad(input_ids, (0, self.seq_length - len(input_ids)), 'constant', constant_values=0)
            else:
                input_ids = input_ids[:self.seq_length]

            label = copy.deepcopy(input_ids)
            label[label == 0] = -100
            if pre_text:
                pre_text_encoded = self.tokenizer(pre_text)
                label[:len(pre_text_encoded['input_ids'])] = -100

            label = np.array(self.img_padding + label.tolist())
            item = {
                'input_ids': input_ids,
                'images': image,
                'labels': label,
                'original_question': Question,
                'original_answer': Answer,
                'restored_question': restored_question,
                'restored_answer': restored_answer,
                'task_related_text': task_related_text,
                'task_related_visual_text': task_related_visual_text
            }

        elif self.mode == 'Test':
            item = {
                'input_ids': 'Question: ' + Question + 'The Answer is:',
                'orin_input_ids': 'Question: ' + restored_question + 'The Answer is:',
                'img_path': sample['image_name'],
                'images': image,
                'labels': Answer,
                'original_question': Question,
                'original_answer': Answer,
                'restored_question': restored_question,
                'restored_answer': restored_answer,
                'task_related_text': task_related_text,
                'task_related_visual_text': task_related_visual_text
            }

        return item

