#!/usr/bin/python3

__author__ = "Paulius Danenas"
__maintainer__ = "Paulius Danenas"
__email__ = "danpaulius@gmail.com"

import os
import glob
import random
import pickle
import itertools
import logging
import numpy as np
import torch
from nltk.corpus.reader.conll import ConllCorpusReader
from allennlp_models.tagging.dataset_readers.ontonotes_ner import OntonotesNamedEntityRecognition

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def set_seed(seed, use_gpu=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_gpu:
        torch.cuda.manual_seed_all(seed)


def process_conllu_file(file):
    data = list()
    with open(file, "r") as f:
        for line in f:
            if line.startswith('#begin'):
                continue
            elif len(line.strip()) == 0:
                data.append(('', ''))
            else:
                split = line.split()
                if len(split) > 5:
                    data.append((split[3], split[4]))
    size = len(data)
    idx_list = [idx + 1 for idx, val in enumerate(data) if val == ('', '')]
    sentences = set([tuple(data[i: j-1]) for i, j in zip([0] + idx_list, idx_list + ([size] if idx_list[-1] != size else []))])
    return list(zip(*[tuple(zip(*item)) for item in sentences]))

def process_conll_file(fpath):
    reader = ConllCorpusReader(os.path.dirname(fpath), os.path.basename(fpath), columntypes=['words', 'pos', 'ignore',  'chunk'])
    sents = reader.iob_sents()
    tokens, iobs = [], []
    for sent in sents:
        if len(sent) == 0: continue
        tokens.append([t[0] for t in sent])
        iobs.append([t[2] for t in sent])
    return tokens, iobs

def _create_dataset(output, output_file):
    x_train = list(itertools.chain.from_iterable(x[0] for x in output))
    y_train = list(itertools.chain.from_iterable(x[1] for x in output))
    del output
    with open(output_file, 'wb') as f:
        pickle.dump((x_train, y_train), f)


def process_ontonotes(ontonotes_path: str, output_path: str='.', skip_dirs=None):

    def select_skipped(dir):
        dirlist = set(map(lambda x: x[0], os.walk(dir)))
        selected = filter(lambda x: all('{sep}{dir}{sep}'.format(sep=os.path.sep, dir=d) not in x for d in skip_dirs), dirlist)
        files = itertools.chain(*map(lambda x: glob.glob(x + '/*.gold_conll'), selected))
        return list(files)

    DATA_TRAIN_PATH = os.path.join(ontonotes_path, 'data', 'train')
    DATA_VALID_PATH = os.path.join(ontonotes_path, 'data', 'development')
    DATA_TEST_PATH = os.path.join(ontonotes_path, 'data', 'test')
    if skip_dirs is not None:
        files_train = select_skipped(DATA_TRAIN_PATH)
        files_valid = select_skipped(DATA_VALID_PATH)
        files_test = select_skipped(DATA_TEST_PATH)
    else:
        files_train = glob.glob(DATA_TRAIN_PATH + '/**/*.gold_conll', recursive=True)
        files_valid = glob.glob(DATA_VALID_PATH + '/**/*.gold_conll', recursive=True)
        files_test = glob.glob(DATA_TEST_PATH + '/**/*.gold_conll', recursive=True)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    logger.info('Creating training dataset')
    _create_dataset(list(map(process_conllu_file, files_train)), os.path.join(output_path, 'data_train.pkl'))
    logger.info('Creating validation dataset')
    _create_dataset(list(map(process_conllu_file, files_valid)), os.path.join(output_path, 'data_valid.pkl'))
    logger.info('Creating testing dataset')
    _create_dataset(list(map(process_conllu_file, files_test)), os.path.join(output_path, 'data_test.pkl'))


def process_conll_corpus(corpus_path: str, output_path: str='.'):
    DATA_TRAIN_PATH = os.path.join(corpus_path, 'data', 'train')
    DATA_TEST_PATH = os.path.join(corpus_path, 'data', 'test')
    fpaths = lambda x, y: glob.glob(x + f'/**/*.{y}', recursive=True)
    files_train = fpaths(DATA_TRAIN_PATH, 'txt') + fpaths(DATA_TRAIN_PATH, 'conll')
    files_test = fpaths(DATA_TEST_PATH, 'txt') + fpaths(DATA_TEST_PATH, 'conll')
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    logger.info('Creating training dataset')
    _create_dataset(list(map(process_conll_file, files_train)), os.path.join(output_path, 'data_train.pkl'))
    logger.info('Creating testing dataset')
    _create_dataset(list(map(process_conll_file, files_test)), os.path.join(output_path, 'data_test.pkl'))


def process_ontonotes_ner(corpus_path: str, output_path: str='.', domains=('bc', 'bn', 'mz', 'nw', 'wb')):
    train_data, train_tags = list(), list()
    valid_data, valid_tags = list(), list()
    test_data, test_tags = list(), list()
    for domain in domains:
        logger.info(f'Processing corpus for domain {domain}')
        rec = OntonotesNamedEntityRecognition(domain_identifier=domain)
        train_dst = rec.read(os.path.join(corpus_path, 'train'))
        for inst in train_dst.instances:
            train_data.append([t.text for t in inst.fields.get('tokens').tokens])
            train_tags.append(inst.fields.get('tags').labels)
        valid_dst = rec.read(os.path.join(corpus_path, 'development'))
        for inst in valid_dst.instances:
            valid_data.append([t.text for t in inst.fields.get('tokens').tokens])
            valid_tags.append(inst.fields.get('tags').labels)
        test_dst = rec.read(os.path.join(corpus_path, 'test'))
        for inst in test_dst.instances:
            test_data.append([t.text for t in inst.fields.get('tokens').tokens])
            test_tags.append(inst.fields.get('tags').labels)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    with open(os.path.join(output_path, 'data_train.pkl'), 'wb') as f:
        pickle.dump((train_data, train_tags), f)
    with open(os.path.join(output_path, 'data_valid.pkl'), 'wb') as f:
        pickle.dump((valid_data, valid_tags), f)
    with open(os.path.join(output_path, 'data_test.pkl'), 'wb') as f:
        pickle.dump((test_data, test_tags), f)