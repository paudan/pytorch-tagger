import sys
sys.path.append('..')
import os
import json
import torch
from transformers import BertConfig, BertTokenizer
from pytorch_tagger import BERT_LSTM_CRF
from pytorch_tagger.datasets import BertDataset

BERT_MODEL_DIR = 'embeddings'
MODEL_PATH = 'bert-pos'

config = BertConfig.from_pretrained(MODEL_PATH, cache_dir=BERT_MODEL_DIR)
with open(os.path.join(MODEL_PATH, 'params.json'), 'r', encoding='utf-8') as f:
    params = json.load(f)
model = BERT_LSTM_CRF.from_pretrained(MODEL_PATH, labels_map=config.id2label, use_bilstm=params['use_bilstm'], rnn_dim=params['rnn_dim'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(torch.device(device))
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
dst = BertDataset(tokenizer)
example = [['I', 'have', 'an', 'option', 'to', 'buy', 'stocks'], ['This', 'is', 'for', 'testing']]
input = dst.transform_input(example, config.label2id)
print(model.predict_tags(input))
