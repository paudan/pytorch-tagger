import os
import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizer, BertModel
from pytorch_tagger.datasets import BertDataset

BERT_MODEL_DIR = 'FinBERT-FinVocab-Uncased'
BERT_CACHE_DIR = 'embeddings'

config = BertConfig.from_json_file(os.path.join(BERT_MODEL_DIR, 'config.json'))
model = BertModel.from_pretrained(BERT_MODEL_DIR)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
model.to(device)
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', cache_dir=BERT_CACHE_DIR)
dst = BertDataset(tokenizer)
example = [['I', 'have', 'an', 'option', 'to', 'buy', 'stocks'], ['This', 'is', 'for', 'testing']]
input = dst.transform_input(example)
model.eval()
dataloader = DataLoader(input)
for b_i, (input_ids, input_mask, segment_ids) in enumerate(dataloader):
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, segment_ids, input_mask)
        print(outputs)