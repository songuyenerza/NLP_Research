import numpy as np
# import torch
# from torch import nn, optim
# from sklearn.metrics import f1_score, accuracy_score
# import os
# from json
from pyvi import ViTokenizer
import matplotlib.pyplot as plt
import re


def preprocess_label(df, dict_label):
    # Tạo một danh sách chứa kết quả đã mã hóa cho tất cả các hàng
    all_encoded_labels = []

    # Duyệt qua từng hàng trong cột 'A' của pandas DataFrame
    for value in df['intent']:

        # print("check",value)
        # Tạo một danh sách nhãn mã hóa với giá trị ban đầu là 0
        encoded_labels = [0] * len(dict_label)
        if(isinstance(value, float) is False):
            # Tách giá trị từ hàng hiện tại
            labels = value.split('.')[0].split('+')
            # Duyệt qua từng nhãn và cập nhật giá trị tương ứng trong danh sách nhãn đã tạo
            for label in labels:
                if(label == "khác"):
                    continue
                label = label.strip()
                # print("label", label)
                # print("label", label=="Ý định hướng dẫn chuyển nhánh")
                encoded_labels[dict_label[label]] = 1

        # Thêm danh sách nhãn đã mã hóa vào danh sách kết quả cho tất cả các hàng
        all_encoded_labels.append(encoded_labels)
    labels = all_encoded_labels
    texts = df['text'].values.astype(str)
    return texts, labels


def pre_process_text(list_text):
    """
    Tiền xử lý danh sách văn bản bằng cách thay thế các dấu câu bằng khoảng trắng.

    và biến đổi chữ cái đầu tiên của mỗi văn bản thành chữ hoa.

    Args:
        list_text (list of str): Danh sách các chuỗi văn bản cần tiền xử lý.

    Returns:
        list of str: Danh sách văn bản đã được tiền xử lý.
    """
    try:
        for i, value in enumerate(list_text):
            # list_text[i] = list_text[i][0].lower() + list_text[i][1:]
            list_text[i] = list_text[i][0].upper() + list_text[i][1:]
            list_text[i] = list_text[i].replace(",", " ")
            list_text[i] = list_text[i].replace(".", " ")
            list_text[i] = list_text[i].replace("?", " ")
            # list_text[i] = list_text[i].replace("/", " ")
    except:
        pass
    return list_text


def rdrsegmenter_data(texts):
    for i, value in enumerate(texts):
        texts[i] = ViTokenizer.tokenize(value)
        # print(texts[i])
    return texts


def get_label(output_model_sub, dict_label):
    """
    Chuyển đổi đầu ra của mô hình thành nhãn tương ứng bằng cách sử dụng từ điển nhãn cho trước.

    Args:
        output_model_sub (array-like): Đầu ra của mô hình, dạng mảng hoặc danh sách.
        dict_label (dict): Từ điển ánh xạ từ chỉ số sang nhãn.

    Returns:
        list of str: Danh sách nhãn tương ứng.
    """
    # print(output_model_sub)
    label_correspond = []
    for value in np.where(output_model_sub)[0]:
        # print(value)
        label_correspond.append(list(dict_label.keys())[value])
    return label_correspond

def save_label_distribution_plot(data, title, filename):
    # Calculate label distribution
    label_distribution = data['intent'].value_counts(normalize=True) * 100
    
    # Plotting the label distribution
    plt.figure(figsize=(6, 4))
    label_distribution.plot(kind='bar', color='skyblue' if 'train' in filename else 'salmon')
    plt.title(title)
    plt.xlabel('Label')
    plt.ylabel('Percentage')
    
    # Save and close the plot
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def sanitize_filename(class_name):
    """
    Sanitize the class name to be used as a filename.
    Replace spaces with underscores and remove special characters like slashes.
    """
    # Replace spaces with underscores and remove special characters except alphanumeric and underscores
    sanitized_name = re.sub(r'[^\w\s]', '', class_name).replace(" ", "_")
    return sanitized_name