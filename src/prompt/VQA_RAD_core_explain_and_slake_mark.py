import requests
import json
import base64
import os
from tqdm import tqdm
import ipdb


# List of generic Chain of Thought prompts for various questions and images

def save_processed_ids(data, file_path):
    """
    Saves the IDs of processed items to a JSON file.

    :param data: List of dictionaries, each containing an 'id' key.
    :param file_path: Path to the JSON file where IDs will be saved.
    """
    processed_ids = [item['id'] for item in data]

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(processed_ids, file, indent=4)


# Example usage
# Assuming 'data' contains the processed items and you have the file path
# updated_json_file_path = "/path/to/processed_ids.json"
# save_processed_ids(data, updated_json_file_path)

import re


def extract_explanation_part(new_conversation_value, correct_answer):
    """
    Extracts the explanation part from the GPT's response.

    :param new_conversation_value: The complete response from GPT.
    :param correct_answer: The correct answer that needs to be checked in the response.
    :return: The explanation part of the response.
    """
    # Split the response into sentences considering possible multiple line breaks
    sentences = re.split(r'\n+', new_conversation_value.strip())
    first_sentence = sentences[0]

    # Check if the first sentence contains the correct answer
    if correct_answer.lower() in first_sentence.lower():
        # Keep all content after the first sentence
        explanation_part = '\n'.join(sentences[1:])
    else:
        # Keep the entire response if the first sentence does not contain the correct answer
        explanation_part = new_conversation_value

    return explanation_part


def extract_question_and_answer_parts(new_conversation_value):
    # Split the response into lines
    lines = new_conversation_value.strip().split('\n')

    # Initialize variables
    question_part = None
    answer_part = None

    # Extract question and answer parts
    for line in lines:
        if line.startswith("- Question:"):
            question_part = line[len("- Question:"):].strip()
        elif line.startswith("- Answer:"):
            answer_part = line[len("- Answer:"):].strip()
        # Continuation of the code to extract question and answer parts
        if question_part is not None and answer_part is not None:
            break

    return question_part, answer_part


# Example usage:
# Assuming new_conversation_value is the response from GPT and correct_answer is known
# explanation_part = extract_explanation_part(new_conversation_value, correct_answer)
# Now explanation_part contains the explanation without the redundant answer sentence


# OpenAI API Key
api_key = ""

# Path to your JSON file with dialogue and images
json_file_path = "./trainset_llava_COT.json"

# Paths to your template files
template_path = "./task_cot_related_template.py"


# Function to load a template file
def load_template(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


# Load the templates
template = load_template(template_path)

# Load the JSON file to get the dialogues and images
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

start_index = next((i for i, item in enumerate(data) if item['id'] == "3345"), None)

# # 检查是否找到了该ID
if start_index is not None:
    # 过滤掉该ID之前的所有项目
    data = data[start_index:]
else:
    # 如果没有找到指定ID，可能需要处理异常或退出
    print("Specified ID not found in data.")

# Process each item in the data
for item in tqdm(data, desc="Processing Images"):
    image_path = os.path.join(os.path.dirname(json_file_path), item['image'])

    # Read and encode the image
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    # Extract existing dialogues
    existing_dialogues = item['conversations']
    #     print("existing_dialogues=",existing_dialogues)
    # Extract existing dialogues
    question = existing_dialogues[0]["value"]
    answer = existing_dialogues[1]["value"]
    question_type = item['question_type']
    image_organ = item['image_organ']

    # Extract the latest question and its direct answer from the conversations
    latest_question = existing_dialogues[-2]['value'] if len(existing_dialogues) > 1 else "No question provided."
    latest_answer = existing_dialogues[-1]['value'] if len(existing_dialogues) > 1 else "No answer provided."

    #     print("existing_question=",latest_question)
    #     print("existing_answer=",latest_answer)

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "system",
                "content": ("""
                            You are an AI assistant specialized in biomedical topics. 
                            Given the latest question about a medical image and its direct answer,your task is to use <task_related></task_related> tags to enclose the direct question to choose the most important emplement to reflect the user’s intentions.
                            And and <task_related_visual></task_related_visual> tags to wrap the description of the visual content that supports the answer.
                            The content should be related to the critical visual elements from the image and be as precise as possible. 
                            Each tag can be used repeatedly and the response should be complete, including the tags.
                            The each tag can be used multiple times or once. Mark the question as many times as necessary. When marking answers, make sure that the marked content includes complete visual descriptions.
                            All question packages must be fine-grained using at least two tags！
                            Pay attention to the marking of negative words！
                            Here is an example to guide you:
                            When labeling questions, label verbs and nouns rather than predicates. Don't mark combinations like "are" and "what is"！！！
                            When marking answers, try to mark the visual content that exists in the visual content of the image and supports the answer, and the content is semantically complete and can be mapped to the corresponding individual in the image through text, and don't use more than 2 for tagging.
                            Here is an example to guide you:
                            1.Identify and label the key medical terms in the question that are directly related to the diagnosis or condition being inquired about, and focus your answer on these terms.

                            2.Identify and label the specific anatomical or pathological terms present in the question.

                            3.Identify and label  the medical terminology within the question that relates to visible abnormalities or conditions in the image.

                            It is not possible to wrap up all the content of the question, and important elements must be selected in a fine-grained manner.

                            Tageed content should have different focus points for different question types, such as:

                            -For MODALITY type questions, answers should focus on confirming what type of medical imaging the image is (e.g., X-ray, MRI, CT scan, etc.).

                            -For ORGAN type questions, answers should focus on the organs and their abnormalities visible in the image.

                            -For PRES type questions, the focus needs to be on confirming or ruling out the presence of specific pathological phenomena.

                            -For POS type questions, the focus is on describing the exact location where the anomaly was found.
                            """
                            )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"The current question and answer of uploaded image is {question} and {answer}, The question type is {question_type} and the organ type is {image_organ}.you can refer to the reply methods in the template:{template}"
                            "Based on the uploaded medical image and the provided question and answer, "
                            "enclose the import elements in question to reflect the user’s intentions with <task_related></task_related> tags. "
                            "Then, use <task_related_visual></task_related_visual> tags to highlight the specific visual evidence from the image that supports your answer,"
                            "The content of the <Task_related_visual><\Task_related_visual> tag needs to be as detailed and precise as possible, corresponding to the target visual content of the answer or question description."
                            "Please return the entire response, including your tags, and ensure no modifications are made to the original question and answer."
                            "Here are some guidelines: "
                            "When labeling questions, label verbs and nouns rather than predicates. Don't mark combinations like 'are' and 'what is' "
                            "When marking answers, try to mark the visual content that exists in the visual content of the image and supports the answer, and the content is semantically complete and can be mapped to the corresponding individual in the image through text, and don't use more than 2 for tagging."
                            "To avoid wrapping invalid and redundant information, <task_related><\task_related> must wrap a direct answer to the question, which cannot exceed 10 words."
                            "<Task_related_visual><\Task_related_visual> wrap the content cannot exceed 20 words. but please also return the entire response!"
                            "The contents of the package need to be as accurate as possible. Each identifier pari can only be used once."
                            "Again, you are just marking, do not make any modifications to the content of the original answer, or generate new content, and ensure that all content is returned based on the markup."
                            "All question packages must be fine-grained using at least two tags!"
                            f"You must follow the mark template{template}"
                            )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
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
        new_conversation_value = response_json['choices'][0]['message']['content']
        # Remove the ```json and ``` markers
        new_conversation_value = new_conversation_value.replace('```json', '').replace('```', '').strip()

        # Now parse the string into a Python dictionary
        try:
            new_conversation_data = json.loads(new_conversation_value)
            #             print("new_conversation_data:", new_conversation_data)

            #           # Check if new_conversation_data is a list or dictionary and extract the question and answer parts
            if isinstance(new_conversation_data, list) and len(new_conversation_data) > 0:
                existing_dialogues[0]['value'] = new_conversation_data[0]["conversations"][0]['value']
                existing_dialogues[1]['value'] = new_conversation_data[0]["conversations"][1]['value']
            elif isinstance(new_conversation_data, dict):
                existing_dialogues[0]['value'] = new_conversation_data["conversations"][0]['value']
                existing_dialogues[1]['value'] = new_conversation_data["conversations"][1]['value']

        except Exception as e:
            print("JSON decoding error:", e)

        # 保存更新后的对话到JSON文件
        updated_json_file_path = "./train_COT_task_related_remain170.json"
        with open(updated_json_file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)
    #         # After processing your data
    #         saved_json_file_path = "/Users/wuzhongze/Documents/workspace/MLLM/LLaVA-Med-main/data/downstream_data/VQA-RAD/train_COT_saved.json"
    #         save_processed_ids(data, updated_json_file_path)
    else:
        print(f"Error processing item {item['id']}: {response.text}")