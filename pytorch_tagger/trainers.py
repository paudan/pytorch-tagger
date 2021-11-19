#!/usr/bin/python3

__author__ = "Paulius Danenas"
__maintainer__ = "Paulius Danenas"
__email__ = "danpaulius@gmail.com"

import os
import shutil
import logging
from tqdm import tqdm, trange
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from transformers.optimization import get_linear_schedule_with_warmup
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import (EarlyStopping, ModelCheckpoint, DeviceStatsMonitor,
                                         TQDMProgressBar, LearningRateMonitor)
from .utils import set_seed

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
module = __import__('transformers')

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)
torch.cuda.empty_cache()


class LightningModelTrainer:

    def __init__(self, model: LightningModule, labels_map, use_gpu=True):
        self.model = model
        if labels_map is None:
            raise ValueError('Labels map is not set')
        self.labels_map = labels_map
        device = 'cuda' if use_gpu is True and torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            logger.info(torch.cuda.get_device_name(0))
            logger.info('Available device count: {}'.format(torch.cuda.device_count()))

    def init_dir(self, output_dir):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    def fit(self, train_data, eval_data, epochs=10, output_dir='bert_pos', warmup_steps=500,
            learning_rate=3e-4, gradient_accumulation_steps=1, max_steps=-1, train_batch_size=32, early_stop=10):
        train_dataloader = DataLoader(train_data, batch_size=train_batch_size)
        val_dataloader = DataLoader(eval_data, batch_size=1)
        self.init_dir(output_dir)
        optimizer = Adam(lr=learning_rate, params=self.model.parameters())
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps)
        self.trainer = Trainer(
            logger=[CSVLogger(output_dir),TensorBoardLogger(output_dir)],
            profiler=None,
            default_root_dir=output_dir,
            gpus=torch.cuda.device_count(),
            auto_lr_find=True,
            accumulate_grad_batches=gradient_accumulation_steps,
            max_epochs=epochs,
            num_sanity_val_steps=0,
            callbacks=[
                EarlyStopping(monitor='valid_mean_loss', patience=early_stop),
                ModelCheckpoint(monitor='valid_mean_loss', dirpath=os.path.join(output_dir, 'checkpoints')),
                DeviceStatsMonitor(),
                TQDMProgressBar(),
                LearningRateMonitor(logging_interval='epoch')
            ]
        )
        self.trainer.optimizers.append(optimizer)
        self.trainer.lr_schedulers.append(scheduler)
        self.trainer.fit(self.model, train_dataloader, val_dataloader)

    @staticmethod
    def load_model(model_dir, model, device, **params):
         return model.load_from_checkpoint(os.path.join(model_dir, 'model.pth'), **params, map_location=device)


class SimpleModelTrainer:

    def __init__(self, model, labels_map, use_gpu=True):
        self.model = model
        if labels_map is None:
            raise ValueError('Labels map is not set')
        self.labels_map = labels_map
        self.device = torch.device('cuda' if use_gpu is True and torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            logger.info(torch.cuda.get_device_name(0))
            logger.info('Available device count: {}'.format(torch.cuda.device_count()))
        self.model.to(self.device)
        self.writer = SummaryWriter()

    def init_dir(self, output_dir):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    def fit(self, train_data, eval_data, epochs=10, output_dir='bert_pos', warmup_steps=500, logging_steps=10,
            learning_rate=3e-4, gradient_accumulation_steps=1, max_steps=-1, train_batch_size=32,
            eval_batch_size=32, early_stop=10):
        set_seed(1234)
        train_dataloader = DataLoader(train_data, batch_size=train_batch_size)
        val_dataloader = DataLoader(eval_data, batch_size=eval_batch_size)
        self.init_dir(output_dir)
        optimizer = Adam(lr=learning_rate, params=self.model.parameters())
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps)
        self.model.train()
        global_step = 0
        best_loss = float('Inf')
        current_stop = 0
        with trange(epochs, desc="Epoch") as ebar:
            for ep in ebar:
                self.model.train()
                tr_loss, current_loss = .0, .0
                with tqdm(train_dataloader, desc="Train") as bbar:
                    for step, batch in enumerate(bbar):
                        input_ids, label_ids = self.model.push_to_device(batch)
                        outputs = self.model(input_ids, label_ids)
                        mean_loss = (outputs + tr_loss) / (step+1)
                        tr_loss += outputs.item()
                        mean_loss.backward()
                        bbar.set_postfix({'mean_loss': mean_loss.item()})
                        if (step + 1) % gradient_accumulation_steps == 0:
                            optimizer.step()
                            scheduler.step()
                            # self.model.zero_grad()
                            global_step += 1
                            if logging_steps > 0 and global_step % logging_steps == 0:
                                self.writer.add_scalar("Train/loss", mean_loss, global_step)
                # Validation
                self.model.eval()
                with tqdm(val_dataloader, desc="Validation") as bbar:
                    for step, batch in enumerate(bbar):
                        with torch.no_grad():
                            input_ids, label_ids = self.model.push_to_device(batch)
                            outputs = self.model(input_ids, label_ids)
                            mean_loss = (outputs + tr_loss) / (step+1)
                            tr_loss += outputs.item()
                            bbar.set_postfix({'mean_loss': mean_loss.item()})
                    if mean_loss < best_loss:
                        best_loss = mean_loss
                        current_stop = 0
                    else:
                        current_stop += 1
                if current_stop == early_stop:
                    break
        self.writer.close()


