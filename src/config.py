from transformers import BertTokenizer, BertModel
MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
# ACCUMULATION = 2 #??
BERT_PATH = "bert-base-uncased"
MODEL_PATH = "model.bin"
TRAINING_FILE = "/home/sanchit/sentiment/input/IMDB Dataset.csv"
TOKENIZER = BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
EPOCHS = 10