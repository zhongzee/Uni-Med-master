import requests
import json
import base64
import os
from tqdm import tqdm

# OpenAI API Key
api_key = "sk-LiKYZht2ejHRYYYv344987469d6c47BcB8F6C5D89c92Ee65"

# Path to your JSON file with dialogue and images
json_file_path = "/mnt/afs/liwenhao/wuzhongze/LLaVA-Med/data/eval/PMC-VQA/train.json"

# Paths to your template files
template_path = "/mnt/afs/liwenhao/wuzhongze/LLaVA-Med/data/eval/PMC-VQA/Uni_COT_Generate.py"

# Function to load a template file
def load_template(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Load the templates
template = load_template(template_path)

# Load the JSON file to get the dialogues and images
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Process each item in the data
for item in tqdm(data, desc="Processing Images"):
    image_path = os.path.join(os.path.dirname(json_file_path), item['image'])

    # Read and encode the image
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    # Extract existing dialogues
    existing_dialogues = item['conversations']


    # Create the payload for the API request
    # Rest of your code...
  
    
# Create the payload for the API request
# Below is an example of how you might structure the payload to request a conversation
# based on the visual content of the image and using COT (Chain of Thought) reasoning.
    payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
            "role": "system",
            "content": ("You are an AI assistant specialized in biomedical topics. "
                        "Given a question, multiple choice answers, and a selected correct answer, "
                        "your task is to provide an explanation directly following the chosen answer, "
                        "using the Chain of Thought method to explain why this answer is correct based on the visual content of the medical image. "
                        "You should also explain why the other options are not correct（If has） . Please avoid repeating the description of the question or the answer,"
                        "and go straight to the explanation, keeping the response concise.")
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Based on the uploaded medical image and the answer provided in the existing dialogues, "
                        "please generate a response using Chain of Thought reasoning. Your response should focus strictly on the visual content of the image. "
                        "Start with the selected answer and then provide reasoning for this choice, as well as explanations for why the other options are not correct. "
                        "Ensure all parts of your response are directly related to the visual aspects of the uploaded image."
                        f"Existing question pairs include and the question and all answer options and the final answer: {existing_dialogues}.Following the template: {template}."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

    # Make the API request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    response = requests.post("https://40.chatgptsb.net/v1/chat/completions", headers=headers, json=payload)

    if response.ok:
        response_json = response.json()
        # 提取GPT的响应
        new_conversation_value = response_json['choices'][0]['message']['content']

        # 从现有对话中提取正确答案
        correct_answer = existing_dialogues[1]['value'].split('.')[0]  # 获取答案部分，假设以点号分隔

        # 检查新对话值是否以正确答案开始，如果是，就去除重复的答案部分
        if new_conversation_value.startswith(correct_answer):
            # 去除重复的答案部分，仅保留链式推理的解释
            explanation_part = new_conversation_value[len(correct_answer):].lstrip(".")
            # 现在，explanation_part包含了不重复的推理部分
        else:
            # 如果新的回复不是以正确答案开始的，我们可以直接使用它
            explanation_part = new_conversation_value

        # 更新现有对话中的GPT回复，将新的解释部分添加到之前的答案之后
        existing_dialogues[1]['value'] += explanation_part

        # Save the updated dialogues to a JSON file
        updated_json_file_path = "/mnt/afs/liwenhao/wuzhongze/LLaVA-Med/data/eval/PMC-VQA/train_COT_test_Uni_updated.json"

        with open(updated_json_file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)
    else:
        print(f"Error processing item {item['id']}: {response.text}")
# Rest of your code to save the updated data...

# # Save the updated data to a new JSON file
# updated_json_file_path = "/Users/wuzhongze/Documents/workspace/MLLM/LLaVA-Med-main/data/downstream_data/PMC-VQA/train_COT_test_Uni_updated.json"
# with open(updated_json_file_path, 'w', encoding='utf-8') as file:
#     json.dump(data, file, indent=4)