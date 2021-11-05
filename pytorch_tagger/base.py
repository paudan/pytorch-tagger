#!/usr/bin/python3

__author__ = "Paulius Danenas"
__maintainer__ = "Paulius Danenas"
__email__ = "danpaulius@gmail.com"

from abc import abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.nn.modules.transformer import MultiheadAttention
from transformers import BertPreTrainedModel, AutoModel
from allennlp.modules.conditional_random_field import ConditionalRandomField
from allennlp.modules.token_embedders.elmo_token_embedder import ElmoTokenEmbedder

module = __import__('transformers')


class BaseRNNMixin:
    input_dim: int
    dropout_prob: float
    hidden_size: int
    bidirectional: bool
    num_labels: int

    @abstractmethod
    def init_hidden(self):
        pass

    @abstractmethod
    def forward_hidden(self, embed_input):
        pass


class BaseLstmCRFMixin(BaseRNNMixin):

    def init_hidden(self):
        self.dropout = nn.Dropout(self.dropout_prob)
        out_dim = self.hidden_size
        self.rnn = nn.LSTM(self.input_dim, self.hidden_size, num_layers=1, bidirectional=self.bidirectional, batch_first=True)
        if self.bidirectional:
            out_dim = self.hidden_size*2
        self.hidden2tag = nn.Linear(out_dim, self.num_labels)

    def forward_hidden(self, embed_input):
        sequence_output = embed_input
        sequence_output = self.dropout(sequence_output)
        sequence_output, _ = self.rnn(sequence_output)
        return self.hidden2tag(sequence_output)


class BaseAttentiveCRFMixin(BaseRNNMixin):

    def init_hidden(self):
        self.dropout = nn.Dropout(self.dropout_prob)
        out_dim = self.hidden_size
        self.rnn = nn.LSTM(self.input_dim, self.hidden_size, num_layers=1, bidirectional=self.bidirectional, batch_first=True)
        if self.bidirectional:
            out_dim = self.hidden_size*2
        self.self_attention = MultiheadAttention(out_dim, num_heads=8)
        self.hidden2tag = nn.Linear(out_dim, self.num_labels)

    def forward_hidden(self, embed_input):
        output = self.dropout(embed_input)
        output, _ = self.rnn(output)
        output, _ = self.self_attention(output, output, output)
        return self.hidden2tag(output)


class Base_BERT_CRF(BertPreTrainedModel, BaseRNNMixin):

    def __init__(self, config, labels_map=None, bidirectional=False, hidden_size=128):
        super(Base_BERT_CRF, self).__init__(config)
        self.labels_map = labels_map
        if labels_map is None:
            self.labels_map = config.id2label
            if labels_map is None:
                raise ValueError('Labels map is not set')
        num_labels = len(labels_map)
        self.bert = AutoModel.from_config(config)
        self.input_dim = config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.bidirectional = bidirectional
        self.crf = ConditionalRandomField(num_labels)
        self.init_hidden()

    def forward(self, input_ids, tags, token_type_ids=None, input_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)
        outputs = outputs[0]
        emissions = self.forward_hidden(outputs)
        loss = -1*self.crf(emissions, tags, mask=input_mask.byte())
        return loss

    def predict(self, input_ids, token_type_ids=None, input_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)
        outputs = outputs[0]
        emissions = self.forward_hidden(outputs)
        viterbi = self.crf.viterbi_tags(emissions, input_mask.byte())
        return [entry[0] for entry in viterbi]

    def predict_tags(self, input: TensorDataset):
        results = []
        for _, (input_ids, input_mask, segment_ids) in enumerate(input):
            input_ids = input_ids.unsqueeze(0)
            input_mask = input_mask.unsqueeze(0)
            segment_ids = segment_ids.unsqueeze(0)
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            with torch.no_grad():
                logits = self.predict(input_ids, segment_ids, input_mask)
            tags = [[self.labels_map[idx] for idx in l] for l in logits][0]
            results.append(tags[1:-1])  # Strip CLS/SEP symbols
        return results


class Base_ELMO_CRF(nn.Module, BaseRNNMixin):

    def __init__(self, options_file, weights_file, labels_map, hidden_size=128, dropout_prob=0.2, bidirectional=True, use_gpu=True):
        super(Base_ELMO_CRF, self).__init__()
        if labels_map is None:
            raise ValueError('Labels map is not set')
        self.labels_map = labels_map
        self.num_labels = len(labels_map)
        device = 'cuda' if use_gpu is True and torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        self.embed = ElmoTokenEmbedder(options_file, weights_file)
        self.input_dim = self.embed.get_output_dim()
        self.crf = ConditionalRandomField(self.num_labels)
        self.init_hidden()

    def forward(self, inputs, tags):
        embed_input = self.embed(inputs)
        emissions = self.forward_hidden(embed_input)
        tags = torch.squeeze(tags, 2)
        mask = torch.ones(*tags.size(), dtype=torch.bool).to(self.device)
        mask[tags == -1] = 0
        loss = -1*self.crf(emissions, tags, mask)
        return loss

    def predict(self, inputs):
        embed_input = self.embed(inputs)
        emissions = self.forward_hidden(embed_input)
        viterbi = self.crf.viterbi_tags(emissions)
        return [entry[0] for entry in viterbi]

    def predict_tags(self, input: TensorDataset):
        results = []
        for _, input_ids in enumerate(input):
            input_ids = input_ids.to(self.device)
            with torch.no_grad():
                logits = self.predict(input_ids)
            tags = [[self.labels_map[idx] for idx in l] for l in logits][0]
            results.append(tags)
        return results