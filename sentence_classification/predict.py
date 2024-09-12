
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import time
import numpy as np
# Create the PhoBERT-based classification model
# Load PhoBERT tokenizer and base model

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

num_classes = 3
tokenizer = AutoTokenizer.from_pretrained("Fsoft-AIC/ViDeBERTa-xsmall")
base_model = AutoModel.from_pretrained("Fsoft-AIC/ViDeBERTa-xsmall")
model = PhoBERTForSequenceClassification(base_model, num_classes)
print('[model:]', model)
class_full = ["customer_names", "address", "sdt"]
# Load the trained model
model_path = "./weights/model_epoch_1.pt"
model.load_state_dict(torch.load(model_path))
model.eval()

# Inference
input_text = " HN-20-1O-TX01 MT-05 Nệl dung hàng ( rống SL sản phẩm:"
encoded_input = tokenizer.encode_plus(
    input_text,
    truncation=True,
    padding='max_length',
    max_length=128,
    return_tensors='pt'
)
input_ids = encoded_input['input_ids']
attention_mask = encoded_input['attention_mask']
while(5):
    t0 = time.time()
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)
        probabilities = F.softmax(logits, dim=1)
        predicted_labels = torch.argmax(logits, dim=1)

    print("time per text = ", time.time() - t0, '(s)')
    print("--------------------------------------------------------------")
    print("input text: ", input_text)
    print("Predicted label:", class_full[predicted_labels.item()], '------> ', "Probabilities:", np.array(probabilities)[0][predicted_labels.item()])
