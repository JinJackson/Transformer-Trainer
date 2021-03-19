from pathlib import Path
from parser1 import args

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch


tokenizer = BertTokenizer.from_pretrained(args.config)

def read_imdb_split(data_filepath):
    split_dir = Path(data_filepath)

    texts = []
    labels = []
    for label_dir in ['pos', 'neg']:
        for text_file in (split_dir / label_dir).iterdir():
            texts.append(text_file.read_text(encoding='utf-8'))
            labels.append(0 if label_dir == 'neg1' else 1)

    return texts, labels

def get_data():
    train_texts, train_labels = read_imdb_split(args.train_file)
    # 划分训练集和测试集
    train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=.2)
    test_texts, test_labels = read_imdb_split(args.test_file)


    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    #[[nums, seq_len, 1], [nums, seq_len, 1], [nums, seq_len, 1]] 分别为input_ids，token_type_ids，attention_mask
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True)

    return train_encodings, train_labels, val_encodings, val_labels, test_encodings, test_labels




class IMDbDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)





#把训练数据划分出train和dev

