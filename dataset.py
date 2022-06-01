from dataclasses import dataclass

import argparse
from typing import Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from mytokenizer import MyTokenizer
from splitter_train import split
from at_base import shift_tokens_right

@dataclass
class DataCollatorForSeq2Seq:
    src_tok : MyTokenizer
    morph_tok : MyTokenizer
    tag_tok : MyTokenizer
    max_len : int = 202
    label_pad_token_id : int = -100

    def __call__(self, features):
        # features : List[Dict]
        morph_labels = [feature["morph_labels"] for feature in features] if "morph_labels" in features[0].keys() else None
        tag_labels = [feature["tag_labels"] for feature in features] if "tag_labels" in features[0].keys() else None

        batch_size = len(morph_labels)
        # labels : List, len(labels) = batch_size

        if morph_labels is not None and tag_labels is not None:
            # if [a,b,c] comes,
            for feature in features:
                feature["morph_labels"] = feature["morph_labels"] + self.morph_tok.index("<eos>")
                feature["tag_labels"] = feature["tag_labels"] + self.tag_tok.index("<eos>")
                # [a, b, c, <eos>]
                remainder = [self.label_pad_token_id] * (self.max_len - len(feature["morph_labels"]))
                assert (self.max_len - len(feature["morph_labels"])) == (self.max_len - len(feature["tag_labels"]))
                feature["morph_labels"] = np.concatenate([feature["morph_labels"], remainder])
                feature["tag_labels"] = np.concatenate([feature["tag_labels"], remainder])
                # [a,b,c,<eos> , -100, -100]

        # prepare input_ids 
        for feature in features:
            # attention_mask
            attention_mask = [1] * len(feature["input_ids"])
            attention_mask_pad = [0] * (self.max_len - len(feature["input_ids"]))
            feature["attention_mask"] = np.concatenate([attention_mask, attention_mask_pad])
            # [1,1,1,0,0]

            # input padding
            input_ids_padding = [self.src_tok.index("<pad>")] * (self.max_len - len(feature["input_ids"]))
            feature["input_ids"] = np.concatenate([feature["input_ids"], input_ids_padding])
            # [i,j,k, <pad>, <pad>]

        # list of dictionary to dictionary
        new_dict = {}
        for key in features[0].keys():
            nps = [feature[key] for feature in features]
            if key == "attention_mask":
                new_dict[key] = torch.FloatTensor(np.stack(nps).reshape(batch_size, -1))
            else :
                new_dict[key] = torch.LongTensor(np.stack(nps).reshape(batch_size, -1))
            
        features = new_dict

        # prepare_decoder_input_ids
        if morph_labels is not None and tag_labels is not None:
            morph_input_ids = shift_tokens_right(features["morph_labels"],
                                                self.morph_tok.index("<pad>"), 
                                                self.morph_tok.index("<bos>"))
            tag_input_ids = shift_tokens_right(features["tag_labels"],
                                                self.tag_tok.index("<pad>"), 
                                                self.tag_tok.index("<bos>"))
        
            features["morph_input_ids"] = morph_input_ids
            features["tag_input_ids"] = tag_input_ids
        # [<bos>, a,b,c, <eos>, <pad>]

        '''
        input_ids
        morph_labels
        tag_labels

        ->

        input_ids
        attention_mask

        morph_labels
        morph_input_ids

        tag_labels
        tag_input_ids
        
        
        '''

        return features
                  
class KMADataset(Dataset):
    def __init__(self, filepath, src_tok, morph_tok, tag_tok, max_len, ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        self.max_len = max_len
        self.filepath = filepath

        self.src_tok = src_tok
        self.morph_tok = morph_tok
        self.tag_tok = tag_tok

        self.srcs, self.morphs, self.tags = self.load()

    def load(self):
        srcs = []
        morphs = []
        tags = []

        src_f = open(self.filepath + "_src.txt", 'r', encoding="UTF-8-sig")
        morph_f = open(self.filepath + "_morph.txt", 'r', encoding="UTF-8-sig")
        tag_f = open(self.filepath + "_tag.txt", 'r', encoding="UTF-8-sig")

        for src, morph, tag in zip(src_f, morph_f, tag_f):
            # split sentence which has over length.
            src_bufs, morph_bufs, tag_bufs = split(src.strip(), morph.strip(), tag.strip())

            for src_buf, morph_buf, tag_buf in zip(src_bufs, morph_bufs, tag_bufs):
                srcs.append(src_buf)
                morphs.append(morph_buf)
                tags.append(tag_buf)
        return srcs, morphs, tags

    def __getitem__(self, index):
        src_input_ids = self.src_tok.encode(list(self.srcs[index]))
        morph_labels = self.morph_tok.encode(list(self.morphs[index]))
        tag_labels = self.tag_tok.encode(self.tags[index].split(" "))

        return {'input_ids': np.array(src_input_ids, dtype=np.int_),
                'morph_labels': np.array(morph_labels, dtype=np.int_),
                'tag_labels': np.array(tag_labels, dtype=np.int_)}
        
    def __len__(self):
        assert len(self.srcs) == len(self.morphs) == len(self.tags), f"len dif"
        return len(self.srcs)

class KMAModule(pl.LightningDataModule):
    def __init__(self, train_file, valid_file, src_tok, morph_tok, tag_tok, max_len, batch_size=8, num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_len = max_len

        self.train_file_path = train_file
        self.valid_file_path = valid_file

        self.src_tok = src_tok
        self.morph_tok = morph_tok
        self.tag_tok = tag_tok

        self.data_collator = DataCollatorForSeq2Seq(src_tok = self.src_tok, morph_tok=self.morph_tok, tag_tok=self.tag_tok, max_len= self.max_len)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--num_workers',
                            type=int,
                            default=5,
                            help='num of worker for dataloader')
        return parser

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        self.train = KMADataset(self.train_file_path, self.src_tok, self.morph_tok, self.tag_tok, self.max_len)
        self.valid = KMADataset(self.valid_file_path, self.src_tok, self.morph_tok, self.tag_tok, self.max_len)
    
    def train_dataloader(self):
        train = DataLoader(self.train, collate_fn=self.data_collator, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return train

    def val_dataloader(self):
        val = DataLoader(self.valid, collate_fn=self.data_collator, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return val
    
    

    
    # only runs in single GPU, and DDP
    '''        
    def transfer_batch_to_device(self, batch, device):
        # batch must be dict
        for name in batch.keys():
            batch[name] = batch[name].to(device)
        return batch
        return super().transfer_batch_to_device(batch, device)
    '''
