import torch
import time
from transformers import AutoTokenizer, AutoModel, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
class_paths = ["full_names.txt", "address.txt", "sdt.txt"]

# read data
texts = []
labels = []

for i, class_path in enumerate(class_paths):
    with open(class_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines]
        texts.extend(lines)
        labels.extend([i] * len(lines))

print("-------------------------------------------------------")
print(f"==========> len data = {len(texts)}, {len(labels)}")
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Load PhoBERT tokenizer and base model
tokenizer = AutoTokenizer.from_pretrained("Fsoft-AIC/ViDeBERTa-xsmall")
base_model = AutoModel.from_pretrained("Fsoft-AIC/ViDeBERTa-xsmall")
print("hidden_size = ", base_model.config.hidden_size)
# Customize the model for sequence classification
class PhoBERTForSequenceClassification(torch.nn.Module):
    def __init__(self, base_model, num_labels):
        super(PhoBERTForSequenceClassification, self).__init__()
        self.base_model = base_model
        
        # self.classifier = torch.nn.Linear(base_model.config.hidden_size, num_labels)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(base_model.config.hidden_size, base_model.config.hidden_size),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(base_model.config.hidden_size, num_labels)
        )
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the last hidden state
        last_hidden_state = outputs.last_hidden_state

        # Usually, we take the hidden state of the first token ([CLS] token) for classification tasks
        cls_token_state = last_hidden_state[:, 0]

        logits = self.classifier(cls_token_state)

        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


# Number of classes
num_classes = 3

# Create the PhoBERT-based classification model
model = PhoBERTForSequenceClassification(base_model, num_classes)

# Convert data 
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

#  DataLoader
train_dataset = MyDataset(train_texts, train_labels, tokenizer, max_len=128)
test_dataset = MyDataset(test_texts, test_labels, tokenizer, max_len=128)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Setting
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 1

# Train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)
model.to(device)

for epoch in range(num_epochs):
    print("-------------------------------------------------------")
    print("epochs = ", epoch)
    model.train()
    count = 0
    start_time = time.time()  # Start time for the epoch

    for batch in tqdm(train_loader):
        count += 1
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        loss = model(input_ids, attention_mask=attention_mask, labels=labels)  # Update this line
        if count % 50 ==0:
            print("loss = ", loss)
        loss.backward()
        optimizer.step()

    # eval
    print("-------------------------------------------------------")
    print("eval")
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask=attention_mask)  # Update this line
            predicted_labels = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (predicted_labels == labels).sum().item()

        accuracy = correct / total
        end_time = time.time()  # End time for the epoch
        epoch_time = end_time - start_time  # Elapsed time for the epoch

    print(f"Epoch {epoch + 1}, eval: Accuracy: {accuracy}------------- loss = {loss}")
    # Save the model
    model_path = f"./weights/model_epoch_{epoch + 1}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

    