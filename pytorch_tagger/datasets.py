#!/usr/bin/python3

__author__ = "Paulius Danenas"
__maintainer__ = "Paulius Danenas"
__email__ = "danpaulius@gmail.com"

import itertools
from collections import namedtuple
import torch
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import IterableDataset
from allennlp.modules.elmo import batch_to_ids


def collate_eval_fn(batch):
    return default_collate([i[0] for i in batch]), \
           default_collate([i[1] for i in batch]), \
           [i[2] for i in batch], [i[3] for i in batch]


class BertDataset(IterableDataset):

    def __init__(self, data, tokenizer, labels=None, labels_map=None, max_seq_length=256, return_inputs=False):
        self.tokenizer = tokenizer
        self.data = data
        self.labels = labels
        self.labels_map = labels_map
        self.max_seq_length = max_seq_length
        self.return_inputs = return_inputs

    @staticmethod
    def tags_list(labels):
        label_list = set(list(itertools.chain(*labels)))
        label_list.add("''")
        return label_list

    @staticmethod
    def labels_map(tags):
        return {label : i for i, label in enumerate(sorted(tags))}

    def process_example(self, example, label=None):
        InputFeatures = namedtuple('InputFeatures', ['input_ids', 'input_mask', 'segment_ids', 'label_id', 'tokens', 'labels'])
        example = list(example)
        if len(example) >= self.max_seq_length - 1:
            example = example[0:(self.max_seq_length - 2)]
            if label is not None:
                label = label[0:(self.max_seq_length - 2)]
        tokens = [self.tokenizer.cls_token] + example + [self.tokenizer.sep_token]
        segment_ids = [0] * len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        if label is not None:
            orig_labels = [''] + label + ['']
            label_ids = [self.labels_map["''"]] + [self.labels_map[l] for l in label] + [self.labels_map["''"]]
        else:
            label_ids, orig_labels = None, None
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            if label is not None:
                label_ids.append(0)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        label_ids = torch.tensor(label_ids, dtype=torch.long)
        return InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                             label_id=label_ids, tokens=tokens, labels=orig_labels)

    def create_stream(self):
        for example, label in zip(self.data, self.labels):
            f = self.process_example(example, label)
            if self.return_inputs:
                yield (f.input_ids, f.input_mask, f.segment_ids), f.label_id, f.tokens, f.labels
            else:
                yield (f.input_ids, f.input_mask, f.segment_ids), f.label_id

    def __iter__(self):
        return self.create_stream()

    def __len__(self):
        return len(self.data)


class ElmoDataset(IterableDataset):

    def __init__(self, data, labels=None, labels_map=None, max_seq_length=256, return_inputs=False):
        self.data = data
        self.labels = labels
        self.labels_map = labels_map
        self.max_seq_length = max_seq_length
        self.return_inputs = return_inputs

    def process_example(self, example, label=None, pad=True):
        InputFeatures = namedtuple('InputFeatures', ['input_ids', 'label_id', 'tokens', 'labels'])
        example = list(example)
        if len(example) >= self.max_seq_length - 1:
            example = example[0:self.max_seq_length]
            if label is not None:
                label = label[0:self.max_seq_length]
        input_ids = batch_to_ids([example])
        if label is not None:
            label_ids, orig_labels = [self.labels_map[l] for l in label], label
        else:
            label_ids, orig_labels = None, None
        # Pad both IDs and labels to common size
        input_ids = torch.squeeze(input_ids, 0)
        if pad is True:
            shape = torch.Size([self.max_seq_length, input_ids.shape[1]])
            reshaped = torch.zeros(shape, dtype=torch.long)
            reshaped[:input_ids.shape[0], :] = input_ids
            labels_reshaped = torch.zeros(self.max_seq_length, dtype=torch.long)
            labels_reshaped[:len(label_ids)] = torch.Tensor(label_ids)
            labels_reshaped = torch.unsqueeze(labels_reshaped, 1)
            del input_ids, label_ids
        else:
            reshaped, labels_reshaped = input_ids, label_ids
        return InputFeatures(input_ids=reshaped, label_id=labels_reshaped, tokens=example, labels=orig_labels)

    def create_stream(self):
        for example, label in zip(self.data, self.labels):
            f = self.process_example(example, label)
            if self.return_inputs:
                yield f.input_ids, f.label_id, f.tokens, f.labels
            else:
                yield f.input_ids, f.label_id

    def __iter__(self):
        return self.create_stream()

    def __len__(self):
        return len(self.data)
