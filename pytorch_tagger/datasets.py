#!/usr/bin/python3

__author__ = "Paulius Danenas"
__maintainer__ = "Paulius Danenas"
__email__ = "danpaulius@gmail.com"

import itertools
from collections import namedtuple
import torch
from torch.utils.data import TensorDataset, Dataset
from allennlp.modules.elmo import batch_to_ids


class BertDataset(object):

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @staticmethod
    def tags_list(labels):
        label_list = set(list(itertools.chain(*labels)))
        label_list.add("''")
        return label_list

    @staticmethod
    def labels_map(tags):
        return {label : i for i, label in enumerate(sorted(tags))}

    def process_example(self, example, labels_map=None, label=None, max_seq_length=256):
        InputFeatures = namedtuple('InputFeatures', ['input_ids', 'input_mask', 'segment_ids', 'label_id', 'tokens', 'labels'])
        example = list(example)
        if len(example) >= max_seq_length - 1:
            example = example[0:(max_seq_length - 2)]
            if label is not None:
                label = label[0:(max_seq_length - 2)]
        tokens = [self.tokenizer.cls_token] + example + [self.tokenizer.sep_token]
        segment_ids = [0] * len(tokens)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        if label is not None:
            orig_labels = [''] + label + ['']
            label_ids = [labels_map["''"]] + [labels_map[l] for l in label] + [labels_map["''"]]
        else:
            label_ids, orig_labels = None, None
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            if label is not None:
                label_ids.append(0)
        return InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids,
                             label_id=label_ids, tokens=tokens, labels=orig_labels)

    def transform(self, examples, labels, labels_map, max_seq_length=256):
        if labels is None:
            raise ValueError('labels cannot be None')
        features = [self.process_example(example, labels_map, label, max_seq_length) for example, label in zip(examples, labels)]
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
        examples = [f.tokens for f in features]
        labels = [f.labels for f in features]
        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        return data, examples, labels

    def transform_input(self, examples, max_seq_length=256):
        features = [self.process_example(example, None, None, max_seq_length) for example in examples]
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        return TensorDataset(all_input_ids, all_input_mask, all_segment_ids)


class ElmoDataset(Dataset):

    def __init__(self, data, labels=None, labels_map=None, max_seq_length=256):
        self.data = data
        self.labels = labels
        self.labels_map = labels_map
        self.max_seq_length = max_seq_length

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
        else:
            reshaped, labels_reshaped = input_ids, label_ids
        return InputFeatures(input_ids=reshaped, label_id=labels_reshaped, tokens=example, labels=orig_labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        tags = self.labels[idx] if self.labels else None
        f = self.process_example(example, tags)
        return f.input_ids, f.label_id

    def transform(self, examples, labels):
        features = [self.process_example(example, label, pad=False) for example, label in zip(examples, labels)]
        all_input_ids = [f.input_ids for f in features]
        all_label_ids = [f.label_id for f in features]
        examples = [f.tokens for f in features]
        labels = [f.labels for f in features]
        return all_input_ids, all_label_ids, examples, labels