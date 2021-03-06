#!/usr/bin/python3

__author__ = "Paulius Danenas"
__maintainer__ = "Paulius Danenas"
__email__ = "danpaulius@gmail.com"

import os
import sys
import argparse
import pickle
import torch
from torch.utils.data import DataLoader
from pytorch_tagger import ELMO_Attentive_CRF, ELMO_LSTM_CRF, ELMO_Transformer_CRF
from pytorch_tagger.datasets import ElmoDataset, BertDataset, collate_eval_fn
from pytorch_tagger.trainers import SimpleModelTrainer
from pytorch_tagger.utils import set_seed, load_model, save_model

OPTIONS_FILE = "elmo/elmo_2x2048_256_2048cnn_1xhighway_options.json"
WEIGHTS_FILE = "elmo/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"
CACHE_DIR = 'embeddings'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default='lstm', type=str, help="Model architecture", choices=['lstm', 'attention','transformer'])
    parser.add_argument("--options-file", default=OPTIONS_FILE, type=str, help="ELMO model options file")
    parser.add_argument("--weights-file", default=WEIGHTS_FILE, type=str, help="ELMO model weights file")
    parser.add_argument("--train-file", default=None, type=str, help="Training data file (Python pickle format)")
    parser.add_argument("--eval-file", default=None, type=str, help="Validation data file (Python pickle format)")
    parser.add_argument("--test-file", default=None, type=str, help="Testing data file (Python pickle format)", required=False)
    parser.add_argument("--output-dir", default='bert-pos', type=str, help="Output directory for the trained model and configuration")
    parser.add_argument("--cache-dir", default=CACHE_DIR, type=str, help="Location to store the downloaded pre-trained models")
    parser.add_argument("--max-seq-length", default=256, type=int, help="Maximum possible sequence size. Shorter sequences will be padded while longer sequences will be truncated")
    parser.add_argument("--do-train", action='store_true', help="Indicates if training will be performed")
    parser.add_argument("--do-test", action='store_true', help="Indicates if testing will be performed")
    parser.add_argument("--use-gpu", action='store_true', help="Indicates if GPU will be used")
    parser.add_argument("--train-batch-size", default=32, type=int, help="Training batch size")
    parser.add_argument("--eval-batch-size", default=32, type=int, help="Evaluation batch size")
    parser.add_argument("--learning-rate", default=3e-5, type=float, help="Learning rate")
    parser.add_argument("--num-epochs", default=10, type=int, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Initial seed for reproducibility")
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--warmup-steps", default=500, type=int, help="Number of warm-up steps for the optimizer")
    parser.add_argument("--max-steps", default=-1, type=int, help="Max training steps for optimization scheduler")
    parser.add_argument("--lower-case", action='store_true', help="Indicates if testing will be performed")
    parser.add_argument("--logging-steps", default=500, type=int, help="Logging frequency")
    parser.add_argument("--use-bilstm", action='store_true', help="Indicates if bidirectional LSTM will be used as the intermediate layer")
    parser.add_argument("--rnn-dim", default=128, type=int, help="Hidden layer dimension")
    parser.add_argument("--num_layers", default=2, type=int, help="Number of hidden LSTM layers")
    parser.add_argument("--early-stop", default=10, type=int, help="Stop after this number of epochs if no performance improvements are observed")
    parser.set_defaults(lower_case=True)
    parser.set_defaults(use_bilstm=False)
    args = parser.parse_args()
    if args.do_train is True and (args.train_file is None or not os.path.isfile(args.train_file)):
        print("Training file must be set if training option is selected")
        sys.exit()
    if args.do_test is True and (args.test_file is None or not os.path.isfile(args.test_file)):
        print("Test file must be set if training option is selected")
        sys.exit()

    if os.path.isdir(args.cache_dir):
        CACHE_DIR = args.cache_dir
    os.environ['TORCH_HOME'] = os.path.abspath(CACHE_DIR)
    torch.hub.set_dir(os.path.abspath(CACHE_DIR))

    set_seed(args.seed, use_gpu=args.use_gpu)
    with open(args.train_file, "rb") as f:
        x_train, y_train = pickle.load(f)
    with open(args.eval_file, "rb") as f:
        x_valid, y_valid = pickle.load(f)
    x_train, y_train = list(map(list, x_train)), list(map(list, y_train))
    x_valid, y_valid = list(map(list, x_valid)), list(map(list, y_valid))
    if args.test_file is not None:
        with open(args.test_file, "rb") as f:
            x_test, y_test = pickle.load(f)
        x_test, y_test = list(map(list, x_test)), list(map(list, y_test))
        # No testing mode yet
        x_train, y_train = x_train + x_test, y_train + y_test

    # If testing data includes unseen tags, add them to training
    tags_train = BertDataset.tags_list(y_train)
    tags_test = BertDataset.tags_list(y_valid)
    tags = set.union(set(tags_train), set(tags_test))
    labels_map = BertDataset.labels_map(tags)
    train_dst = ElmoDataset(x_train, y_train, labels_map=labels_map, max_seq_length=args.max_seq_length)
    eval_dst = ElmoDataset(x_valid, y_valid, labels_map=labels_map, max_seq_length=args.max_seq_length)
    params = {
        'options_file': args.options_file,
        'weights_file': args.weights_file,
        'bidirectional': args.use_bilstm,
        'hidden_size': args.rnn_dim,
        'labels_map': labels_map,
        'num_layers': args.num_layers
    }
    if args.model_type == 'lstm':
        model = ELMO_LSTM_CRF(**params)
    elif args.model_type == 'attention':
        model = ELMO_Attentive_CRF(**params)
    elif args.model_type == 'transformer':
        model = ELMO_Transformer_CRF(**params)
    else:
        raise Exception("Model type is not valid")
    trainer = SimpleModelTrainer(model, labels_map, use_gpu=args.use_gpu)
    trainer.fit(train_dst, eval_dst,
        epochs=args.num_epochs,
        output_dir=args.output_dir,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        train_batch_size=args.train_batch_size,
        early_stop=args.early_stop
    )
    save_model(args.output_dir, model, save_state=True)
    # Load model
    # device = 'cuda' if args.use_gpu is True and torch.cuda.is_available() else 'cpu'
    # model = load_model(args.output_dir, torch.device(device))
    eval_dst = ElmoDataset(x_valid, y_valid, labels_map=labels_map, max_seq_length=args.max_seq_length, return_inputs=True)
    dataloader = DataLoader(eval_dst, batch_size=args.eval_batch_size, collate_fn=collate_eval_fn)
    print(model.evaluate_dataloader(dataloader))

