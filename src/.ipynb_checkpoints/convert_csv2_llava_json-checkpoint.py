import json
import csv

# Let's assume the JSON content is stored in a file named 'data.json'
input_file = 'data.json'
output_file = 'output.csv'

# Read the JSON data from the file
with open(input_file, 'r') as f:
    data = json.load(f)

# Open a CSV file for writing
with open(output_file, 'w', newline='') as csvfile:
    fieldnames = ['qid', 'image_name', 'image_organ', 'answer', 'answer_type', 'question_type', 'question', 'phrase_type']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header
    writer.writeheader()

    # Loop through each entry in the JSON data
    for entry in data:
        # Extract the conversation values
        question = entry['conversations'][0]['value']
        answer = entry['conversations'][1]['value']

        # Write to the CSV
        writer.writerow({
            'qid': entry['id'],
            'image_name': entry['image'],
            'image_organ': entry['image_organ'],
            'answer': answer,
            'answer_type': entry['answer_type'],
            'question_type': entry['question_type'],
            'question': question,
            'phrase_type': entry['phrase_type']
        })
