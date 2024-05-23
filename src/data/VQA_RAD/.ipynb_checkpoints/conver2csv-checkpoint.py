import json
import csv

# 假设 JSON 数据存储在 'data.json' 文件中
json_file = 'trainset_llava_COT.json'

# 输出 CSV 文件的名称
csv_file = 'train_cot.csv'

# 读取 JSON 数据
with open(json_file, 'r') as file:
    data = json.load(file)

# 写入 CSV 文件
with open(csv_file, 'w', newline='', encoding='utf-8') as file:
    csv_writer = csv.writer(file)
    
    # 写入标题行
    headers = data[0].keys()
    csv_writer.writerow(headers)
    
    # 写入数据行
    for item in data:
        csv_writer.writerow(item.values())
