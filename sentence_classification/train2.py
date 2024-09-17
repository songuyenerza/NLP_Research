import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Step 1: Load the CSV file without headers
# Assuming the format is: label, text
df = pd.read_csv('./dataset/sentiment-analysis-dataset/fpt.csv', header=None, names=['label', 'text'])

# Step 2: Split the dataset into 80% training and 20% validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

print("train_df:: ", train_df.head())
print("val_df:: ", val_df.head())

# Step 3: Load the pre-trained PhoBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

# Step 4: Tokenize the datasets
class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.texts = data['text'].values
        self.labels = data['label'].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize the text using PhoBERT tokenizer
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Step 5: Create training and validation datasets
train_dataset = SentimentDataset(train_df, tokenizer, max_len=128)
val_dataset = SentimentDataset(val_df, tokenizer, max_len=128)

# Step 6: Create DataLoader for training and validation
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Step 7: Load pre-trained PhoBERT model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained('vinai/phobert-base', num_labels=3)  # Adjust `num_labels` to match your dataset

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(">>>>>    device :::: ", device)

# Step 8: Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)

# Step 9: Training the model with tqdm progress bar
def train_model(model, train_loader, val_loader, optimizer, device, epochs=3):
    print(">>>>>>>> Start Training.....")
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_train_accuracy = 0
        
        # Create progress bar for training loop
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Training)", leave=False)
        
        for batch in train_progress:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update total loss
            total_train_loss += loss.item()

            # Calculate accuracy
            preds = torch.argmax(logits, dim=1)
            total_train_accuracy += (preds == labels).sum().item()

            # Update progress bar description with loss and accuracy
            train_progress.set_postfix({
                'loss': f'{loss.item():.3f}', 
                'accuracy': f'{total_train_accuracy / len(train_loader.dataset):.3f}'
            })

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_accuracy = total_train_accuracy / len(train_loader.dataset)

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Training loss: {avg_train_loss:.3f}")
        print(f"Training accuracy: {avg_train_accuracy:.3f}")

        # Step 10: Validate the model after each epoch
        model.eval()
        total_val_accuracy = 0
        total_val_loss = 0

        # Create progress bar for validation loop
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Validation)", leave=False)
        
        with torch.no_grad():
            for batch in val_progress:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits

                total_val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                total_val_accuracy += (preds == labels).sum().item()

                # Update validation progress bar description with loss and accuracy
                val_progress.set_postfix({
                    'loss': f'{loss.item():.3f}', 
                    'accuracy': f'{total_val_accuracy / len(val_loader.dataset):.3f}'
                })

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_accuracy = total_val_accuracy / len(val_loader.dataset)

        print(f"\nValidation loss: {avg_val_loss:.3f}")
        print(f"Validation accuracy: {avg_val_accuracy:.3f}")
        model.train()

# Step 11: Train the model
train_model(model, train_loader, val_loader, optimizer, device, epochs=3)

# Step 12: Save the model
model.save_pretrained('phobert-sentiment-classification-model')
tokenizer.save_pretrained('phobert-sentiment-classification-tokenizer')
