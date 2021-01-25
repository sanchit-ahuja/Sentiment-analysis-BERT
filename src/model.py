import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import tensorboard
import torch
import torch.nn as nn
import transformers
from sklearn import metrics
from transformers import AdamW, get_linear_schedule_with_warmup

import config


class BERTBaseUncased(pl.LightningModule):
    best_accuracy = 0

    def __init__(self):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)  # Binary classification problem

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert(ids, attention_mask=mask,
                          return_dict=False,
                          token_type_ids=token_type_ids)
        # out1 sequence of hidden states for each and every hidden token 512 vectors of size 768
        # sequence of hidden-states at the output of the last layer shape = [batch_size,sequence_length, hidden_size]
        # out2 pooler output shape = [batch_size, hidden_size] last layer hidden-state of the first token of the sequence further processed
        bo = self.bert_drop(o2)
        output = self.out(bo)
        return output

    def configure_optimizers(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.decay"]
        optimizer_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.00},
        ]
        optimizer = AdamW(optimizer_parameters, lr=3e-5)
        return optimizer

    @staticmethod
    def loss_fn(outputs, targets):
        return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

    def training_step(self, dataset, batch_idx):
        ids = dataset["ids"]
        token_type_ids = dataset["token_type_ids"]
        masks = dataset["mask"]
        targets = dataset["target"]
        outputs = self(ids=ids, mask=masks, token_type_ids=token_type_ids)
        loss = BERTBaseUncased.loss_fn(outputs, targets)
        self.log('train_loss', loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, dataset, batch_idx):
        ids = dataset["ids"]
        token_type_ids = dataset["token_type_ids"]
        masks = dataset["mask"]
        targets = dataset["target"]
        outputs = self(ids=ids, mask=masks, token_type_ids=token_type_ids)
        loss = BERTBaseUncased.loss_fn(outputs, targets)
        outputs = torch.sigmoid(outputs).detach().cpu().numpy()
        outputs = np.array(outputs) >= 0.5
        targets = targets.detach().cpu().numpy()
        accuracy = metrics.accuracy_score(targets, outputs)
        accuracy = torch.tensor(accuracy)
        print(f"accuracy: {accuracy}")
        return {'loss': loss, 'accuracy': accuracy}

    def validation_epoch_end(self, val_step_outputs):
        avg_loss = torch.stack([x['loss'] for x in val_step_outputs]).mean()
        avg_accuracy = torch.stack([x['accuracy']
                                    for x in val_step_outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'avg_val_acc': avg_accuracy}
        return {'loss': avg_loss, 'progress_bar': tensorboard_logs}

# Input ids -> token indices numerical represesntations of tokens building the sequences that will be used as input by the model
# Attention Masks -> Optional argument to the model. This indicates to the model which tokens should be attended to and which should not. Don't attend to padded sequences
# Token type IDs -> Some models’ purpose is to do sequence classification or question answering. These require two different sequences to be joined in a single “input_ids” entry, which usually is performed with the help of special tokens, such as the classifier ([CLS]) and separator ([SEP]) tokens.
