import pandas as pd

path_edit = "/home/vision/workspace/sonnt373/dev/NLP_Research/sentence_classification/dataset/thaydan/topic-50-new-v2/incorrect_edit.csv"

dict_data_edit = {}
# Read the CSV file
df = pd.read_csv(path_edit)  # Replace 'your_file.csv' with your actual file path

for index, row in df.iterrows():
    # Check if 'edit' is not NaN
    if pd.notna(row['edit']):
        # Access each row's values using the column names
        text_value = row['text']
        predicted_value = row['predicted']
        actual_value = row['actual']
        edit_value = row['edit']

        # Print the row's data
        print(f"Row {index}:")
        print(f"Text: {text_value}")
        print(f"Predicted: {predicted_value}")
        print(f"Actual: {actual_value}")
        print(f"Edit: {edit_value}")
        print("-" * 40)
        if edit_value != "Khác":
            dict_data_edit[text_value] = edit_value

print("dict_data_edit:: ", dict_data_edit)



path_org = "/home/vision/workspace/sonnt373/dev/NLP_Research/sentence_classification/dataset/thaydan/topic-50-new-v2/train_data.csv"
path_save = "/home/vision/workspace/sonnt373/dev/NLP_Research/sentence_classification/dataset/thaydan/topic-50-new-v2/train_data_fix_1709.csv"
df_org = pd.read_csv(path_org)
dict_data_org = {}
for index, row in df_org.iterrows():
    text_value = row['text']
    intent_value = row['intent']

    if text_value in dict_data_edit.keys():
        dict_data_org[text_value] = dict_data_edit[text_value]
    else:
        dict_data_org[text_value] = intent_value


# Convert the dictionary into a DataFrame
df_output = pd.DataFrame(list(dict_data_org.items()), columns=['text', 'intent'])

# Save the DataFrame as a CSV file
df_output.to_csv(path_save, index=False)  # Replace 'output_file.csv' with your desired file name

print("CSV file has been saved.")