import pandas as pd
import json

def convert_excel_json(path):
    # Load the Excel file into a DataFrame
    df = pd.read_excel(path)

    # Replace NaN values in the "intent" column with "khác"
    df['intent'] = df['intent'].fillna("khác")
    
    # Replace specific intent values
    df['intent'] = df['intent'].replace('phản ánh các vấn đề khác', 'khác')
    df['intent'] = df['intent'].replace('Khách hàng thắc mắc khiếu nại về việc gia hạn gói cước', 'Thông tin gia hạn gói cước đang sử dụng')
    
    # Convert the DataFrame into a list of JSON objects
    list_json = df[['text', 'intent']].to_dict(orient='records')

    # Save as JSON file
    name_file = path.split(".")[0].split("/")[-1] + ".json"
    json_path = "/home/vision/workspace/sonnt373/dev/NLP_Research/sentence_classification/dataset/thaydan/topic-50-new-v2/" + name_file
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(list_json, f, ensure_ascii=False)
    
    # Save the DataFrame as a CSV file
    csv_path = json_path.replace('.json', '.csv')
    df[['text', 'intent']].to_csv(csv_path, index=False, encoding='utf-8-sig')

# Paths to your Excel files
path1 = "/home/vision/workspace/sonnt373/dev/NLP_Research/sentence_classification/dataset/thaydan/topic-50-new-v2/train_data.xlsx"
path2 = "/home/vision/workspace/sonnt373/dev/NLP_Research/sentence_classification/dataset/thaydan/topic-50-new-v2/test_data.xlsx"
path3 = "/home/vision/workspace/sonnt373/dev/NLP_Research/sentence_classification/dataset/thaydan/topic-50-new-v2/test_2.xlsx"

# Convert and save as JSON and CSV
convert_excel_json(path1)
convert_excel_json(path2)
convert_excel_json(path3)
