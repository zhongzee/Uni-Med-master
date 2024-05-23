import json

# Load the JSON file
file_path = '/root/autodl-tmp/VQA_RAD/trainset_llava_COT.json'

with open(file_path, 'r') as file:
    data = json.load(file)

# Function to reformat the "value" in GPT responses
def reformat_value(value):
    # Find the index of the first period (.) to split the answer from the explanation
    period_index = value.find(".")
    if period_index != -1:
        # Split the answer and the explanation
        answer = value[:period_index + 1].strip()
        explanation = value[period_index + 1:].strip()
        # Concatenate the explanation with "The answer is " and the answer
#         new_value = f"{explanation} The answer is {answer}"
        new_value = f"{explanation} {answer}"
        return new_value
    else:
        # Return the original value if no period found
        return value

# Iterate through the data and modify the "value" in GPT responses
for item in data:
    conversations = item.get("conversations", [])
    for conversation in conversations:
        if conversation.get("from") == "gpt":
            original_value = conversation.get("value", "")
            new_value = reformat_value(original_value)
            conversation["value"] = new_value

# Save the modified data to a new file
modified_file_path = '/root/autodl-tmp/VQA_RAD/trainset_llava_COT_New.json'
with open(modified_file_path, 'w') as file:
    json.dump(data, file, indent=4)
