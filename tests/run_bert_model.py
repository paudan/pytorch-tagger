import sys
sys.path.append('..')
import torch
from torch.utils.data import DataLoader
from pytorch_tagger.datasets import BertDataset
from pytorch_tagger.utils import load_model
from transformers import AutoTokenizer

model_dir = 'bert-pos-tagger-attention'
CACHE_DIR = 'embeddings'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_model(model_dir, torch.device(device))
labels_map = model.labels_map

examples = [['I', 'have', 'an', 'option', 'to', 'buy', 'stocks'],
            ['This', 'is', 'only', 'for', 'testing']]
tokenizer = AutoTokenizer.from_pretrained(model_dir, do_lower_case=False, cache_dir=CACHE_DIR)
predict_dst = BertDataset(examples, tokenizer, labels_map=labels_map)
val_dataloader = DataLoader(predict_dst, batch_size=len(examples))
for step, batch in enumerate(val_dataloader):
    with torch.no_grad():
        print(model.predict_tags(batch))

