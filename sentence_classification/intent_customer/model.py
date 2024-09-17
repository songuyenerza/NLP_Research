from torch import nn
from transformers import AutoModel

class PhoBertMultiLabelClassifierCustomer(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
#         self.phobert = DistilBertModel.from_pretrained('distilbert-base-multilingual-cased')
        # self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.phobert.config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        output = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output.last_hidden_state
        pooler = hidden_state[:, 0]
        # drop = self.dropout(pooler)
        drop = pooler
        linear = self.classifier(drop)
        return self.sigmoid(linear)