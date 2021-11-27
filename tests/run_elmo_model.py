import sys
sys.path.append('..')
import torch
from torch.utils.data import DataLoader
from pytorch_tagger.datasets import ElmoDataset
from pytorch_tagger.utils import load_model

model_dir = 'elmo-pos-tagger-lstm'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_model(model_dir, torch.device(device))
labels_map = model.labels_map

examples = [['I', 'have', 'an', 'option', 'to', 'buy', 'Microsoft', 'stocks', 'at', 'NYSE'],
            ['This', 'is', 'only', 'for', 'testing']]
predict_dst = ElmoDataset(examples, labels_map=labels_map)
val_dataloader = DataLoader(predict_dst, batch_size=1)
for step, batch in enumerate(val_dataloader):
    with torch.no_grad():
        print(model.predict_tags(batch))

single_example = ['I', 'have', 'an', 'option', 'to', 'buy', 'more', 'stocks']
embed = ElmoDataset.process_example(examples[0], labels_map, pad=False)
embed = embed.input_ids.unsqueeze(0)
print(model.predict_tags(embed))
