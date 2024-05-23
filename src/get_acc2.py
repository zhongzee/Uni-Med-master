import pandas as pd
import difflib
def normalize_word(word):
    if pd.isna(word):
        return ""
    word_str = str(word) if not isinstance(word, str) else word
    return word_str.lower().strip()

def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()

def calculate_accuracy_closed(df):
    correct_predictions = df[df['Label'].str.lower() == df['Pred'].str.lower()]
    accuracy = len(correct_predictions) / len(df) if len(df) > 0 else 0
    return accuracy

def calculate_accuracy_open(df):
    correct_count = 0
    for _, row in df.iterrows():
        gt_value = normalize_word(row['Label'])
        pred_value = normalize_word(row['Pred'])
        if gt_value in pred_value or str_similarity(pred_value, gt_value) > 0.5:
            correct_count += 1
    accuracy = correct_count / len(df) if len(df) > 0 else 0
    return accuracy


# Example usage:
# Assuming 'open_df' and 'closed_df' are pandas DataFrames with 'Label' and 'Pred' columns
# open_df = pd.read_csv("path_to_open_questions.csv")
# closed_df = pd.read_csv("path_to_closed_questions.csv")

open_df = pd.read_csv("./result_final_greedy_VQA_RAD_openQA_no_pretrain_no_aug_test_epoch100_choise_VQA_RAD_fix.csv")
closed_df = pd.read_csv("./result_final_greedy_VQA_RAD_closedQA_no_pretrain_no_aug_test_epoch100_choise_VQA_RAD_fix.csv")
accuracy_closed = calculate_accuracy_closed(closed_df)
accuracy_open = calculate_accuracy_open(open_df)
print(f"Closed Questions Accuracy: {accuracy_closed}")
print(f"Open Questions Accuracy: {accuracy_open}")
