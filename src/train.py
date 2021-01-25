import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from sklearn import metrics, model_selection
from transformers import AdamW, get_linear_schedule_with_warmup

import config
import dataset
from model import BERTBaseUncased


def run():
    df = pd.read_csv(config.TRAINING_FILE).fillna("none")
    df.sentiment = df.sentiment.apply(
        lambda x: 1 if x == "positive" else 0
    )
    df_train, df_valid = model_selection.train_test_split(
        df, test_size=0.1, random_state=42, stratify=df.sentiment.values)  # Same ratio of +ve and -ve index
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)
    train_dataset = dataset.BERTDataset(
        review=df_train.review.values, target=df_train.sentiment.values)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4,
    )

    valid_dataset = dataset.BERTDataset(
        review=df_valid.review.values, target=df_valid.sentiment.values)
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=4,
    )
    model = BERTBaseUncased()
    trainer = Trainer(gpus=1)
    trainer.fit(model, train_dataloader=train_data_loader,
                val_dataloaders=[valid_data_loader])


if __name__ == "__main__":
    run()
