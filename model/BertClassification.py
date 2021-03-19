from transformers import BertModel, BertPreTrainedModel
import torch.nn as nn

class BertClassificationModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        #print(labels)
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        #取cls [768,1]
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        if labels == None:
            return logits
        else:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels.float())
            return loss, logits
