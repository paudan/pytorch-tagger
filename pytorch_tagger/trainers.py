#!/usr/bin/python3

__author__ = "Paulius Danenas"
__maintainer__ = "Paulius Danenas"
__email__ = "danpaulius@gmail.com"

import os
import shutil
import itertools
import json
import logging
from abc import abstractmethod
from collections import namedtuple
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm import tqdm, trange
from sklearn.metrics import precision_score, recall_score, f1_score

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
module = __import__('transformers')

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class AbstractModelTrainer:

    def __init__(self, model, labels_map, use_gpu=True):
        self.model = model
        self.labels_map = labels_map
        device = 'cuda' if use_gpu is True and torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model.to(self.device)
        self.n_gpu = torch.cuda.device_count()
        if self.device.type == 'cuda':
            logger.info(torch.cuda.get_device_name(0))
            logger.info('Available device count: {}'.format(self.n_gpu))
            logger.info('Memory Usage:')
            logger.info('Allocated: {} GB'.format(round(torch.cuda.memory_allocated(0)/1024**3,1)))
            logger.info('Cached: {} GB'.format(round(torch.cuda.memory_reserved(0)/1024**3,1)))
            if self.n_gpu > 1:
                self.model = torch.nn.DataParallel(self.model)
        self.writer = SummaryWriter()

    def init_dir(self, output_dir):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    @abstractmethod
    def save_model(self, output_dir):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, dataloader, model):
        raise NotImplementedError()

    @abstractmethod
    def fit_model(self, batch):
        raise NotImplementedError()

    def filter_predictions(self, all_tokens, all_true, all_pred):
        return all_true, all_pred

    def fit(self, train_data, eval_data, epochs=10, output_dir='bert_pos', warmup_steps=500, logging_steps=500,
            learning_rate=3e-4, gradient_accumulation_steps=1, max_steps=-1, train_batch_size=32, early_stop=10):
        eval_tokens, eval_labels, eval_data = eval_data
        train_dataloader = DataLoader(train_data, batch_size=train_batch_size)
        self.init_dir(output_dir)
        optimizer = Adam(lr=learning_rate, params=self.model.parameters())
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps)
        t_total = len(train_data) // gradient_accumulation_steps * epochs

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_data))
        logger.info("  Num Epochs = %d", epochs)
        logger.info("  Total optimization steps = %d", t_total)
        self.model.train()
        global_step = 0
        best_f1 = 0.0
        with trange(epochs, desc="Epoch") as tr:
            for ep in tr:
                tr_loss, logging_loss = 0.0, 0.0
                self.model.train()
                for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                    outputs = self.fit_model(batch)
                    loss = outputs
                    if self.n_gpu > 1:
                        loss = loss.mean() # mean() to average on multi-gpu.
                    if gradient_accumulation_steps > 1:
                        loss = loss / gradient_accumulation_steps
                    loss.backward()
                    tr_loss += loss.item()
                    if (step + 1) % gradient_accumulation_steps == 0:
                        optimizer.step()
                        scheduler.step()  # Update learning rate schedule
                        self.model.zero_grad()
                        global_step += 1
                        if logging_steps > 0 and global_step % logging_steps == 0:
                            tr_loss_avg = (tr_loss-logging_loss)/logging_steps
                            self.writer.add_scalar("Train/loss", tr_loss_avg, global_step)
                            logging_loss = tr_loss
                    tr.set_postfix(total_loss=tr_loss, mean_loss=tr_loss/(step+1))

                # Evaluation
                overall = self.evaluate(eval_data, self.model, eval_tokens, eval_labels)
                f1_score = overall.fscore
                self.writer.add_scalar("Eval/precision", overall.prec, ep)
                self.writer.add_scalar("Eval/recall", overall.rec, ep)
                self.writer.add_scalar("Eval/f1_score", overall.fscore, ep)
                logger.debug(f"Eval/precision: {overall.prec}, eval/recall: {overall.rec}, eval/f1_score: {overall.fscore}\n")
                if f1_score >= best_f1:
                    logger.debug(f"----------the best f1 is {f1_score}---------")
                    best_f1 = f1_score
                    self.save_model(output_dir)
        self.writer.close()

    def evaluate(self, eval_data, model, orig_tokens, orig_labels):
        model.eval()
        pred_labels = self.predict(eval_data, model)
        all_tokens = list(itertools.chain(*orig_tokens))
        all_true = list(itertools.chain(*orig_labels))
        all_pred = list(itertools.chain(*pred_labels))
        all_true, all_pred = self.filter_predictions(all_tokens, all_true, all_pred)
        Metrics = namedtuple('Metrics', ['prec', 'rec', 'fscore'])
        return Metrics(prec=precision_score(all_true, all_pred, average='micro'),
                       rec=recall_score(all_true, all_pred, average='micro'),
                       fscore=f1_score(all_true, all_pred, average='micro'))


class BertModelTrainer(AbstractModelTrainer):

    def __init__(self, config, model, tokenizer, labels_map, use_gpu=True):
        super(BertModelTrainer, self).__init__(model, labels_map, use_gpu)
        self.config = config
        self.tokenizer = tokenizer

    def save_model(self, output_dir):
        label2id = self.labels_map
        id2label = {value:key for key,value in label2id.items()}
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        self.config.id2label = id2label
        self.config.label2id = label2id
        self.config.save_pretrained(output_dir)
        params = {'rnn_dim': 128, 'use_bilstm': True}
        with open(os.path.join(output_dir, 'params.json'), 'w', encoding='utf-8') as f:
            json.dump(params, f)

    def fit_model(self, batch):
        batch = tuple(t.to(self.device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        return self.model(input_ids, label_ids, segment_ids, input_mask)

    def predict(self, eval_data, model):
        label2id = self.labels_map
        id2label = {value:key for key,value in label2id.items()}
        pred_labels = []
        for b_i, (input_ids, input_mask, segment_ids, _) in enumerate(tqdm(eval_data, desc="Evaluating")):
            input_ids = input_ids.unsqueeze(0)
            input_mask = input_mask.unsqueeze(0)
            segment_ids = segment_ids.unsqueeze(0)
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            with torch.no_grad():
                logits = model.predict(input_ids, segment_ids, input_mask)
            for l in logits:
                pred_labels.append([id2label[idx] for idx in l])
        return pred_labels

    def filter_predictions(self, all_tokens, all_true, all_pred):
        # Exclude BERT specific tags
        skip_mask = list(map(lambda x: x not in [self.tokenizer.cls_token, self.tokenizer.sep_token], all_tokens))
        all_true = list(itertools.compress(all_true, skip_mask))
        all_pred = list(itertools.compress(all_pred, skip_mask))
        return all_true, all_pred


class ElmoModelTrainer(AbstractModelTrainer):

    def __init__(self, model, labels_map, use_gpu=True):
        super(ElmoModelTrainer, self).__init__(model, labels_map, use_gpu)

    def save_model(self, output_dir):
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        torch.save(self.model, os.path.join(output_dir, 'model.pth'))

    def fit_model(self, batch):
        input_ids, label_ids = tuple(batch)
        input_ids = input_ids.to(self.device)
        label_ids = label_ids.to(self.device)
        return self.model(input_ids, label_ids)

    def predict(self, eval_data, model):
        label2id = self.labels_map
        id2label = {value:key for key,value in label2id.items()}
        pred_labels = []
        for b_i, input_ids in enumerate(tqdm(eval_data, desc="Evaluating")):
            input_ids = input_ids.unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = model.predict(input_ids)
            for l in logits:
                pred_labels.append([id2label[idx] for idx in l])
        return pred_labels
