#!/usr/bin/python3

__author__ = "Paulius Danenas"
__maintainer__ = "Paulius Danenas"
__email__ = "danpaulius@gmail.com"

import os
import shutil
import json
import logging
import itertools
from abc import abstractmethod
from collections import namedtuple
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers.optimization import get_linear_schedule_with_warmup
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import (EarlyStopping, ModelCheckpoint, DeviceStatsMonitor,
                                         TQDMProgressBar, LearningRateMonitor)
from sklearn.metrics import precision_score, recall_score, f1_score

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
module = __import__('transformers')

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
torch.cuda.empty_cache()


class AbstractModelTrainer:

    def __init__(self, model: LightningModule, labels_map, use_gpu=True):
        self.model = model
        self.labels_map = labels_map
        device = 'cuda' if use_gpu is True and torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            logger.info(torch.cuda.get_device_name(0))
            logger.info('Available device count: {}'.format(torch.cuda.device_count()))
            logger.info('Memory Usage:')
            logger.info('Allocated: {} GB'.format(round(torch.cuda.memory_allocated(0)/1024**3,1)))
            logger.info('Cached: {} GB'.format(round(torch.cuda.memory_reserved(0)/1024**3,1)))

    def init_dir(self, output_dir):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    @abstractmethod
    def save_model(self, output_dir):
        raise NotImplementedError()

    def fit(self, train_data, eval_data, epochs=10, output_dir='bert_pos', warmup_steps=500,
            learning_rate=3e-4, gradient_accumulation_steps=1, max_steps=-1, train_batch_size=32):
        train_dataloader = DataLoader(train_data, batch_size=train_batch_size)
        val_dataloader = DataLoader(eval_data, batch_size=1)
        self.init_dir(output_dir)
        optimizer = Adam(lr=learning_rate, params=self.model.parameters())
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps)
        trainer = Trainer(
            logger=[CSVLogger(output_dir),TensorBoardLogger(output_dir)],
            default_root_dir=output_dir,
            gpus=torch.cuda.device_count(),
            auto_lr_find=True,
            accumulate_grad_batches=gradient_accumulation_steps,
            max_epochs=epochs,
            num_sanity_val_steps=0,
            callbacks=[
                EarlyStopping(monitor='valid_mean_loss', patience=10),
                ModelCheckpoint(monitor='valid_mean_loss', dirpath=os.path.join(output_dir, 'checkpoints')),
                DeviceStatsMonitor(),
                TQDMProgressBar(),
                LearningRateMonitor(logging_interval='epoch')
            ]
        )
        trainer.optimizers.append(optimizer)
        trainer.lr_schedulers.append(scheduler)
        trainer.fit(self.model, train_dataloader, val_dataloader)

    def evaluate(self, data, all_tokens):
        input, orig_labels = tuple(data)
        pred_labels = self.model.predict(input)
        all_true = list(itertools.chain(*orig_labels))
        all_pred = list(itertools.chain(*pred_labels))
        all_true, all_pred = self.filter_predictions(all_true, all_pred, all_tokens)
        Metrics = namedtuple('Metrics', ['precision', 'recall', 'fscore'])
        return Metrics(precision=precision_score(all_true, all_pred, average='micro'),
                       recall=recall_score(all_true, all_pred, average='micro'),
                       fscore=f1_score(all_true, all_pred, average='micro'))

    def filter_predictions(self, all_true, all_pred, all_tokens):
        return all_true, all_pred


class BertModelTrainer(AbstractModelTrainer):

    def __init__(self, config, model, tokenizer, labels_map, use_gpu=True):
        super(BertModelTrainer, self).__init__(model, labels_map, use_gpu)
        self.config = config
        self.tokenizer = tokenizer

    def save_model(self, output_dir):
        label2id = self.labels_map
        id2label = {value:key for key,value in label2id.items()}
        model_to_save = self.model._model.module if hasattr(self.model._model, 'module') else self.model._model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.config.id2label = id2label
        self.config.label2id = label2id
        self.config.save_pretrained(output_dir)
        params = {'rnn_dim': 128, 'use_bilstm': True}
        with open(os.path.join(output_dir, 'params.json'), 'w', encoding='utf-8') as f:
            json.dump(params, f)

    def filter_predictions(self, all_true, all_pred, all_tokens):
        # Exclude BERT specific tags
        skip_mask = list(map(lambda x: x not in [self.tokenizer.cls_token, self.tokenizer.sep_token], all_tokens))
        all_true = list(itertools.compress(all_true, skip_mask))
        all_pred = list(itertools.compress(all_pred, skip_mask))
        return all_true, all_pred


class ElmoModelTrainer(AbstractModelTrainer):

    def save_model(self, output_dir):
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        torch.save(self.model, os.path.join(output_dir, 'model.pth'))

