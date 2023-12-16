from transformers import BertForSequenceClassification
import torch.nn as nn


class BertModel(nn.Module):
    def __init__(self, num_categories):
        super().__init__()
        self.bert_model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=num_categories)
        self.num_categories = num_categories

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        return outputs.logits
