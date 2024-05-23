import json
import csv


def process_json_to_csv(json_file, csv_file):
    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 打开CSV文件并准备写入
    with open(csv_file, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        # 写入标题行
        writer.writerow(
            ['id', 'image_name', 'image_organ', 'answer', 'answer_type', 'question_type', 'question', 'phrase_type'])

        # 遍历每个条目并写入CSV
        for entry in data:
            qid = entry['id']
            image_name = entry['image']
            image_organ = entry['image_organ']
            answer = entry['conversations'][1]['value']  # gpt的回答
            answer_type = entry['answer_type']
            question_type = entry['question_type']
            question = entry['conversations'][0]['value']  # human的问题
            phrase_type = entry['phrase_type']

            # 写入行
            writer.writerow([qid, image_name, image_organ,answer, answer_type, question_type, question, phrase_type])


# 使用示例
process_json_to_csv('/root/autodl-tmp/VQA_RAD/Task_COT_related_final.json', '/root/autodl-tmp/VQA_RAD/Task_COT_related_final.csv')
