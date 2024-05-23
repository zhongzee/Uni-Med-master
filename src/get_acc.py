import pandas as pd
import collections
from nltk.translate.bleu_score import sentence_bleu
import difflib


# Define the function to normalize the word, if needed
# def normalize_word(word):
#     return word.lower().strip()

def normalize_word(word):
    # Check for NaN (not a number) values and convert them to empty strings
    if pd.isna(word):
        return ""
    # Convert any non-string value to its string representation and normalize
    word_str = str(word) if not isinstance(word, str) else word
    return word_str.lower().strip()

# Define the function to calculate exact match
def calculate_exactmatch(pred_value, gt_value):
    return int(pred_value == gt_value)


# Define the function to calculate F1 score
def calculate_f1score(pred_value, gt_value):
    pred_tokens = pred_value.split()
    gt_tokens = gt_value.split()
    common_tokens = set(pred_tokens) & set(gt_tokens)
    if len(common_tokens) == 0:
        return 0, 0, 0
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(gt_tokens)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score, precision, recall


# Define the function to calculate string similarity
def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()


# Assume 'open_df' and 'closed_df' are the DataFrames for open and closed questions
# Replace with actual paths to the CSV files
open_df = pd.read_csv("./result_final_greedy_VQA_RAD_openQA_no_pretrain_no_aug_test_epoch100_choise_VQA_RAD_fix.csv")
closed_df = pd.read_csv("./result_final_greedy_VQA_RAD_closeQA_no_pretrain_no_aug_test_epoch100_choise_VQA_RAD_fix.csv")


# Define evaluation function
def evaluate(gt, pred):
    exact_scores = []
    f1_scores = []
    bleu_scores = []
    open_hit_scores = []
    closed_scores = []

    for gt_value, pred_value in zip(gt, pred):
        gt_value = normalize_word(gt_value)
        pred_value = normalize_word(pred_value)

        # Calculate exact match
        exact_match = calculate_exactmatch(pred_value, gt_value)
        exact_scores.append(exact_match)

        # Calculate F1 score
        f1, precision, recall = calculate_f1score(pred_value, gt_value)
        f1_scores.append(f1)

        # Calculate BLEU score
        bleu = sentence_bleu([gt_value.split()], pred_value.split())
        bleu_scores.append(bleu)

        # Determine if it's an open or closed question
        if 'yes' in gt_value or 'no' in gt_value:  # Assuming closed questions are Yes/No
            hit = 1.0 if gt_value in pred_value else 0.0
            closed_scores.append(hit)
        else:
            hit = 1.0 if str_similarity(pred_value, gt_value) > 0.8 else 0.0
            open_hit_scores.append(hit)

    # Calculate overall accuracy (not averaging open and closed separately)
    # overall_accuracy = sum(exact_scores) / len(exact_scores)

    return {
        "exact_scores": exact_scores,
        "f1_scores": f1_scores,
        "bleu_scores": bleu_scores,
        "open_hit_scores": open_hit_scores,
        "closed_scores": closed_scores,
        # "overall_accuracy": overall_accuracy
    }


# Evaluate open and closed questions separately
open_results = evaluate(open_df['Label'].tolist(), open_df['Pred'].tolist())
closed_results = evaluate(closed_df['Label'].tolist(), closed_df['Pred'].tolist())

# Combine open and closed scores for overall evaluation
combined_exact_scores = open_results['exact_scores'] + closed_results['exact_scores']
combined_f1_scores = open_results['f1_scores'] + closed_results['f1_scores']
combined_bleu_scores = open_results['bleu_scores'] + closed_results['bleu_scores']
combined_hit_scores = open_results['open_hit_scores'] + closed_results['closed_scores']

# Calculate overall accuracy for combined open and closed questions
overall_accuracy_combined = sum(combined_exact_scores) / len(combined_exact_scores)

# Output the results
print("Open Question Evaluation:")
print(open_results)
print("Closed Question Evaluation:")
print(closed_results)
print("Overall Accuracy (Combined Open and Closed Questions):")
print(overall_accuracy_combined)

