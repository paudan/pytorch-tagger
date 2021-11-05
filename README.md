# BERT-BiLSTM-CRF

PyTorch implementation of BERT-BiLSTM-CRF taggers for POS and NER tasks. Currently, only POS tagger is implemented.

## Usage

The tool can be used as from the command line by running `bilstm-crf-pos.py`; the options are given below.

```
usage: bilstm-crf-pos.py [-h] [--train-file TRAIN_FILE]
                         [--eval-file EVAL_FILE] [--test-file TEST_FILE]
                         [--model-name MODEL_NAME] [--output-dir OUTPUT_DIR]
                         [--cache-dir CACHE_DIR]
                         [--max-seq-length MAX_SEQ_LENGTH] [--do-train]
                         [--do-test] [--use-gpu]
                         [--train-batch-size TRAIN_BATCH_SIZE]
                         [--eval-batch-size EVAL_BATCH_SIZE]
                         [--learning-rate LEARNING_RATE]
                         [--num-epochs NUM_EPOCHS] [--seed SEED]
                         [--gradient-accumulation-steps GRADIENT_ACCUMULATION_STEPS]
                         [--warmup-steps WARMUP_STEPS] [--max-steps MAX_STEPS]
                         [--lower-case] [--logging-steps LOGGING_STEPS]
                         [--use-bilstm] [--rnn-dim RNN_DIM]

optional arguments:
  -h, --help            show this help message and exit
  --train-file TRAIN_FILE
                        Training data file (Python pickle format)
  --eval-file EVAL_FILE
                        Validation data file (Python pickle format)
  --test-file TEST_FILE
                        Testing data file (Python pickle format)
  --model-name MODEL_NAME
                        BERT model which will be used for training (from
                        TorchHub/HuggingFace)
  --output-dir OUTPUT_DIR
                        Output directory for the trained model and
                        configuration
  --cache-dir CACHE_DIR
                        Location to store the downloaded pre-trained models
  --max-seq-length MAX_SEQ_LENGTH
                        Maximum possible sequence size. Shorter sequences will
                        be padded while longer sequences will be truncated
  --do-train            Indicates if training will be performed
  --do-test             Indicates if testing will be performed
  --use-gpu             Indicates if GPU will be used
  --train-batch-size TRAIN_BATCH_SIZE
                        Training batch size
  --eval-batch-size EVAL_BATCH_SIZE
                        Evaluation batch size
  --learning-rate LEARNING_RATE
                        Learning rate
  --num-epochs NUM_EPOCHS
                        Number of training epochs
  --seed SEED           Initial seed for reproducibility
  --gradient-accumulation-steps GRADIENT_ACCUMULATION_STEPS
                        Gradient accumulation steps
  --warmup-steps WARMUP_STEPS
                        Number of warm-up steps for the optimizer
  --max-steps MAX_STEPS
                        Max training steps for optimization scheduler
  --lower-case          Indicates if testing will be performed
  --logging-steps LOGGING_STEPS
                        Logging frequency
  --use-bilstm          Indicates if bidirectional LSTM will be used as the
                        intermediate layer
  --rnn-dim RNN_DIM     LSTM hidden layer dimension

```

At this moment, only Python `pickle` files are supported as input which contain sequences of tokens as lists, together with lists representing tags for each these tokens. 

&copy; Paulius Danenas (danpaulius(eta)gmail.com), 2021

