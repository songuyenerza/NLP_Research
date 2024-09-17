import torch
# from torch import nn, optim
from torch.utils.data import Dataset
# from transformers import PhoBertTokenizer, AutoModel
# from transformers import AutoModel, AutoTokenizer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import f1_score, accuracy_score


class MultiLabelDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length, labels=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        if(self.labels is not None):
            label = torch.tensor(self.labels[idx], dtype=torch.float)
        else:
            label = [0]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }