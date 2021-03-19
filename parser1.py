import argparse
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--train_file', default='data/aclImdb/train1')
parser.add_argument('--test_file', default='data/aclImdb/test1')
parser.add_argument('--config', default='bert-base-uncased')

parser.add_argument('--epochs', default=5, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--warmup_steps', default=0.1, type=float)
parser.add_argument('--decay_weight', default=0.01, type=float)
parser.add_argument('--logging_dir', default='./logging')
parser.add_argument('--logging_steps', default=100, type=int)
parser.add_argument('--out_dir', default='./result/model')
parser.add_argument('--learning_rate', default=5e-5, type=float)
parser.add_argument('--grad_accumulate', default=1, type=int)


args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
args.device = device
