#!/usr/bin/python3

__author__ = "Paulius Danenas"
__maintainer__ = "Paulius Danenas"
__email__ = "danpaulius@gmail.com"

import os
import json
import pickle
import random
import numpy as np
import torch
from pytorch_lightning import Trainer


def set_seed(seed, use_gpu=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_gpu:
        torch.cuda.manual_seed_all(seed)

def load_model(model_dir, device):
    with open(os.path.join(model_dir, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    model.to(device)
    return model

def save_model(model_dir, model, save_state=False):
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    if save_state is True:
        if hasattr(model, 'trainer') and isinstance(model.trainer, Trainer):
            model.trainer.save_checkpoint(os.path.join(model_dir, 'weights.pth'), weights_only=True)
        else:
            torch.save(model.state_dict(), os.path.join(model_dir, 'state.zip'))
    with open(os.path.join(model_dir, 'params.json'), 'w') as f:
        json.dump({'hidden_size': model.hidden_size, 'bidirectional': model.bidirectional}, f)
    with open(os.path.join(model_dir, 'labels_map.json'), 'w') as f:
        json.dump(model.labels_map, f)

def save_transformer_model(model_dir, model, save_state=False):
    save_model(model_dir, model, save_state)
    model.tokenizer.save_pretrained(model_dir)
    model.config.id2label = model.id2label
    model.config.label2id = model.labels_map
    model.config.save_pretrained(model_dir)