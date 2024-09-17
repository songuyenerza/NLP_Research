import json
import os
import logging  # Import logging module
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from torch import nn, optim
from intent_customer.function import get_label, rdrsegmenter_data, pre_process_text, preprocess_label, save_label_distribution_plot
from intent_customer.preprocess_data import MultiLabelDataset
from intent_customer.model import PhoBertMultiLabelClassifierCustomer
from sklearn.metrics import f1_score, accuracy_score


OUTPUT = "./output1709"
os.makedirs(OUTPUT, exist_ok=True)

# Setup logging
logging.basicConfig(
    filename=os.path.join(OUTPUT, "training_log.log"),  # Log file path
    filemode='a',  # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    level=logging.INFO  # Log level
)
class TrainIntentAnalysis:
    """Lớp này hỗ trợ việc huấn luyện và đánh giá mô hình dự đoán ý định từ văn bản tiếng Việt."""

    def __init__(self, path_data_train):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        self.path_data_train = path_data_train
        self.path_save_model = os.path.join(OUTPUT, "weights")
        self.label_mapping = None
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        self.scheduler = None
        # init model
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
        self.device = device

    def __load_data(self):
        try:
            if '.json' in self.path_data_train:
                with open(self.path_data_train) as f:
                    intent = json.load(f)
                df = pd.DataFrame(intent)
            elif '.csv' in self.path_data_train:
                df = pd.read_csv(self.path_data_train)

            df_clean = df.loc[df['intent'] != 'khác']
            intent_labels = df_clean['intent'].values.reshape(-1, 1)
            unique_labels = np.unique(intent_labels)
            label_mapping = {value: index for index, value in enumerate(unique_labels)}

            logging.info(f"Data loaded successfully with {len(df_clean)} records.")
            return df, label_mapping
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def train_epoch(self, data_loader):
        self.model = self.model.train()
        total_loss = 0

        for d in tqdm(data_loader):
            input_ids = d["input_ids"].to(self.device)
            attention_mask = d["attention_mask"].to(self.device)
            labels = d["labels"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = self.loss_fn(outputs, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()

        logging.info(f"Training loss for this epoch: {total_loss / len(data_loader)}")

    def eval_model(self, data_loader):
        self.model = self.model.eval()

        predictions = []
        ground_truths = []

        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(self.device)
                attention_mask = d["attention_mask"].to(self.device)
                labels = d["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                predictions.extend((outputs > 0.6).type(torch.int).tolist())
                ground_truths.extend(labels.tolist())

        f1 = f1_score(ground_truths, predictions, average='macro')
        accuracy = accuracy_score(ground_truths, predictions)
        logging.info(f"Evaluation - F1 score: {f1:.4f}, Accuracy: {accuracy:.4f}")
        return f1, accuracy

    def train_model(self, num_epochs, batch_size, max_length, path_weights, mode="train"):
        try:
            df_train, self.label_mapping = self.__load_data()

            self.model = PhoBertMultiLabelClassifierCustomer(len(self.label_mapping)).to(self.device)
            self.loss_fn = nn.BCELoss()
            self.optimizer = optim.AdamW(self.model.parameters(), lr=5e-5)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=4, gamma=0.1)

            df_train, df_val = train_test_split(df_train, test_size=0.1, stratify=df_train['intent'], random_state=42)

            if mode == "train":
                save_label_distribution_plot(df_train, 'Label Distribution in Training Set', os.path.join(OUTPUT, 'train_label_distribution.png'))
                save_label_distribution_plot(df_val, 'Label Distribution in Validation Set', os.path.join(OUTPUT, 'val_label_distribution.png'))

            val_texts, val_labels = preprocess_label(df_val, self.label_mapping)
            train_texts, train_labels = preprocess_label(df_train, self.label_mapping)

            val_texts = pre_process_text(val_texts)
            train_texts = pre_process_text(train_texts)

            val_texts = rdrsegmenter_data(val_texts)
            train_texts = rdrsegmenter_data(train_texts)

            val_dataset = MultiLabelDataset(val_texts, self.tokenizer, max_length=max_length, labels=val_labels)
            train_dataset = MultiLabelDataset(train_texts, self.tokenizer, max_length=max_length, labels=train_labels)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            best_acc = 0

            if mode == "train":
                if not os.path.exists(self.path_save_model):
                    os.makedirs(self.path_save_model)
                    logging.info(f"Directory save_model created at {self.path_save_model}")

                for epoch in range(num_epochs):
                    logging.info(f"Epoch {epoch + 1}/{num_epochs} started.")
                    self.train_epoch(train_loader)

                    f1, accuracy = self.eval_model(val_loader)
                    self.scheduler.step()

                    if accuracy > best_acc:
                        best_acc = round(accuracy, 2)
                        torch.save(self.model.state_dict(), os.path.join(self.path_save_model, f"best_model_acc_{best_acc}.pth"))
                        logging.info(f"New best model saved with accuracy {best_acc}")

                    logging.info(f"Epoch {epoch + 1} completed. Validation accuracy: {accuracy:.4f}, F1: {f1:.4f}")

                # Save label mapping
                with open(os.path.join(self.path_save_model, "label_mapping.json"), "w") as f:
                    json.dump(self.label_mapping, f)

                logging.info("Model training completed.")
                return accuracy

            else:
                self.model.load_state_dict(torch.load(path_weights))
                f1, accuracy = self.eval_model(val_loader)
                logging.info(f"Model evaluation completed. Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        except Exception as e:
            logging.error(f"Error in train_model: {str(e)}")
            raise


# Example usage
path_train = "/home/vision/workspace/sonnt373/dev/NLP_Research/sentence_classification/dataset/thaydan/topic-50-new-v2/train_data_fix_1709.csv"
mode = "train"  # or "train"
path_weights = ""  # Not needed when mode='train'

model = TrainIntentAnalysis(path_train)
model.train_model(num_epochs=10, batch_size=16, max_length=256, path_weights=path_weights, mode=mode)
