from transformers import BertTokenizer, BertModel
MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
# ACCUMULATION = 2 #??
BERT_PATH = "../input/bert_base_uncased"
MODEL_PATH = "model.bin"
TRAINING_FILE = "../input/imdb.csv"
TOKENIZER = BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)