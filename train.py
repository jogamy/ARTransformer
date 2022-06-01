import argparse
import logging
import os
import numpy as np
from tqdm import tqdm
import math

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

from at_base import ATransformer
from dataset import KMAModule
from mytokenizer import MyTokenizer

parser = argparse.ArgumentParser(description='KoBART Summarization')

parser.add_argument('--resume_from_checkpoint ',
                    type=str,
                    help='resume',
                    default=None)

parser.add_argument('--checkpoint_path',
                    type=str,
                    help='checkpoint path')

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--train_file',
                            type=str,
                            default='data/train',
                            help='train file')

        parser.add_argument('--valid_file',
                            type=str,
                            default='data/valid',
                            help='valid file')

        parser.add_argument('--batch_size',
                            type=int,
                            default=32,
                            help='')
        return parser

class Base(pl.LightningModule):
    def __init__(self, args, **kwargs) -> None:
        super(Base, self).__init__()
        self.save_hyperparameters(args)
        self.args = args

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument('--batch-size',
                            type=int,
                            default=32,
                            help='batch size for training (default: 32)')

        parser.add_argument('--lr',
                            type=float,
                            default=5e-4,
                            help='The initial learning rate')

        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.05,
                            help='warmup ratio')

        parser.add_argument('--model_path',
                            type=str,
                            default=None,
                            help='kobart model path')

        parser.add_argument('--n_heads', type=int, default=8)
        parser.add_argument('--n_layers', type=int, default=6)
        parser.add_argument('--d_model', type=int, default=512)
        parser.add_argument('--feedforward', type=int, default=2048)
        parser.add_argument('--dropout', type=int, default=0.1)
        parser.add_argument("--max_len", type=int, default=256, help="Maximum length of the output utterances")

        return parser

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        num_workers = self.hparams.num_workers
        data_len = len(self.train_dataloader().dataset)
        logging.info(f'number of workers {num_workers}, data length {data_len}')
        num_train_steps = int(data_len / (self.hparams.batch_size) * self.hparams.max_epochs)
        logging.info(f'num_train_steps : {num_train_steps}')
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        logging.info(f'num_warmup_steps : {num_warmup_steps}')
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler,
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

class Model(Base):
    def __init__(self, args, **kwargs):
        super(Model, self).__init__(args, **kwargs)
        
        src_tok = MyTokenizer()
        src_tok.read_vocab(args.train_file + '_src_vocab.txt')
        self.src_tok = src_tok

        morph_tok = MyTokenizer(extra_special_symbols=["<bos>", "<eos>"])
        morph_tok.read_vocab(args.train_file + '_morph_vocab.txt')
        self.morph_tok = morph_tok

        tag_tok = MyTokenizer(extra_special_symbols=["<bos>", "<eos>"])
        tag_tok.read_vocab(args.train_file +'_tag_vocab.txt')
        self.tag_tok = tag_tok

        self.pad_token_id = src_tok.index("<pad>")

        self.model = ATransformer(args, self.src_tok, self.morph_tok, self.tag_tok)

    def forward(self, inputs):
        return self.model(
                        input_ids = inputs["input_ids"],
                        attention_mask = inputs["attention_mask"],

                        morph_input_ids = inputs["morph_input_ids"],
                        morph_attention_mask = None,
                        morph_labels = inputs["morph_labels"],

                        tag_input_ids = inputs["tag_input_ids"],
                        tag_attention_mask = None,
                        tag_labels = inputs["tag_labels"],

                        morph_past_key_values = None,
                        tag_past_key_values = None,
                        encoder_outputs = None,
                        morph_embeds = None,
                        tag_embeds = None, 

                        use_cache = None,
                        output_attentions = None,
                        output_hidden_states = None
        )

    def training_step(self, batch, batch_idx):
        outs = self(batch)
        loss = outs[-1]
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outs = self(batch)
        '''
        outs[0] = morph_logit
        outs[1] = tag_logit
        outs[2] = outs[-1] = loss
        '''
        loss = outs[-1]
        return (loss)

    def validation_epoch_end(self, outputs):
        losses = []
        for loss in outputs:
            losses.append(loss)
        self.log('val_loss', torch.stack(losses).mean(), prog_bar=True)

if __name__ == '__main__':
    parser = Base.add_model_specific_args(parser)
    parser = ArgsBase.add_model_specific_args(parser)
    parser = KMAModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    logging.info(args)

    model = Model(args)

    dm = KMAModule(args.train_file,
                    args.valid_file,
                    model.src_tok, model.morph_tok, model.tag_tok,
                    args.max_len,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers)
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss", patience=20, mode="min")   

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                                       dirpath=args.default_root_dir,
                                                       filename='version_4/{epoch:02d}-{val_loss:.3f}',
                                                       verbose=True,
                                                       save_last=True,
                                                       mode='min',
                                                       save_top_k=-1)
    
    tb_logger = pl_loggers.TensorBoardLogger(os.path.join(args.default_root_dir, 'tb_logs'))
    lr_logger = pl.callbacks.LearningRateMonitor()
    trainer = pl.Trainer.from_argparse_args(args, gpus=args.gpus, accelerator="dp", logger=tb_logger,
                                            callbacks=[checkpoint_callback, early_stop_callback, lr_logger])
    trainer.fit(model, dm)    