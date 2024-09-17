import torch
import time
import pandas as pd
import logging
from transformers import AutoTokenizer, AutoModel, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(
    filename="training_log.log",
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Load data from CSV
df = pd.read_csv('./dataset/sentiment-analysis-dataset/fpt.csv', header=None, names=['label', 'text'])

# Split into train and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Save label distribution plots for train and val datasets
def save_label_distribution_plot(data, title, filename):
    label_distribution = data['label'].value_counts(normalize=True) * 100
    plt.figure(figsize=(6, 4))
    label_distribution.plot(kind='bar', color='skyblue' if 'train' in filename else 'salmon')
    plt.title(title)
    plt.xlabel('Label')
    plt.ylabel('Percentage')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

save_label_distribution_plot(train_df, 'Label Distribution in Training Set', 'train_label_distribution.png')
save_label_distribution_plot(val_df, 'Label Distribution in Validation Set', 'val_label_distribution.png')

# Load PhoBERT tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")
base_model = AutoModel.from_pretrained("vinai/phobert-base-v2")
logging.info(f"Loaded model with hidden size: {base_model.config.hidden_size}")

# Customize the model for sequence classification
class PhoBERTForSequenceClassification(torch.nn.Module):
    def __init__(self, base_model, num_labels):
        super(PhoBERTForSequenceClassification, self).__init__()
        self.base_model = base_model
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(base_model.config.hidden_size, base_model.config.hidden_size),
            # torch.nn.Dropout(0.1),
            torch.nn.Linear(base_model.config.hidden_size, num_labels)
        )
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_token_state = last_hidden_state[:, 0]
        logits = self.classifier(cls_token_state)
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

# Convert data into Dataset format
class MyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Prepare DataLoader
train_dataset = MyDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer, max_len=128)
val_dataset = MyDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer, max_len=128)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# Settings
num_classes = len(df['label'].unique())
model = PhoBERTForSequenceClassification(base_model, num_classes)
optimizer = AdamW(model.parameters(), lr=2e-4)
num_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
logging.info(f"Using device: {device}")

# Training loop with logging and progress bar
for epoch in range(num_epochs):
    logging.info(f"Starting epoch {epoch+1}/{num_epochs}")
    model.train()
    start_time = time.time()

    # Initialize the tqdm progress bar for training
    train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)", leave=False)

    for batch in train_progress:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        loss = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()

        # Update the progress bar with the current loss value
        train_progress.set_postfix(loss=loss.item())

        # Log the loss for every batch (or every n batches if desired)
        logging.info(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Evaluation loop
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Evaluating)", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask=attention_mask)
            predicted_labels = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()

    accuracy = correct / total
    epoch_time = time.time() - start_time
    logging.info(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds. Accuracy: {accuracy:.4f}")

    # Save the model
    model_path = f"./weights/model_epoch_{epoch+1}.pt"
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved at {model_path}")
