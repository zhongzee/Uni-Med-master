import pandas as pd

# Define the path to the CSV file (replace this with the actual path of your CSV file)
csv_path = '/root/Uni-Med/src/Uni/result_final_greedy_VQA_RAD_closeQA_no_pretrain_no_aug_test_epoch100_choise_VQA_RAD.csv'

# Read the CSV file
df = pd.read_csv(csv_path)

# Define a function to keep only the first word of a string
def keep_first_word(string):
    return string.split()[0] if isinstance(string, str) and string else ""

# Apply the function to the 'Pred' column
df['Pred'] = df['Pred'].apply(keep_first_word)

# Define the path where you want to save the modified CSV file
output_csv_path = '/root/Uni-Med/src/Uni/result_final_greedy_VQA_RAD_closeQA_no_pretrain_no_aug_test_epoch100_choise_VQA_RAD_fix.csv'

# Save the modified DataFrame to a new CSV file
df.to_csv(output_csv_path, index=False)
