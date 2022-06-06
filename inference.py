import argparse
from tqdm import tqdm
import yaml
import math

import torch
import torch.nn.functional as F

from mytokenizer import MyTokenizer
from train import Model

from FileManager import FileManager
from splitter_inf import split


def duplicate_encoder_out(encoder_out, att_mask, bsz, beam_size):
    new_encoder_out = encoder_out.unsqueeze(2).repeat(beam_size, 1, 1, 1).view(bsz * beam_size, encoder_out.size(1), -1)
    new_att_mask = att_mask.unsqueeze(1).repeat(beam_size, 1, 1).view(bsz * beam_size, -1)
    return new_encoder_out, new_att_mask

def predict_length_beam(predicted_lengths, length_beam_size):
    beam_probs = predicted_lengths.topk(length_beam_size, dim=1)[0]
    beam = predicted_lengths.topk(length_beam_size, dim=1)[1]
    beam[beam < 2] = 2 
    beam = beam[0].tolist()
    beam_probs = beam_probs[0].tolist()
    return beam, beam_probs

def make_enc_input(input_ids, tok, max_len):
    attention_mask = [1] * len(input_ids) \
                        + [0] * (max_len - len(input_ids))
    input_ids = input_ids + [tok.index("<pad>")] * (max_len - len(input_ids))

    return input_ids, attention_mask


def make_dec_input(input_ids, tok, max_len):
    attention_mask = [1] * len(input_ids) \
                        + [0] * (max_len - len(input_ids))
    attention_mask_ = []
    attention_mask_.append(attention_mask)    
    attention_mask = attention_mask_
    attention_mask = torch.tensor(attention_mask)
    attention_mask = attention_mask.cuda()

    input_ids = input_ids + [tok.index("<pad>")] * (max_len - len(input_ids))

    input_ids_ = []
    input_ids_.append(input_ids)
    input_ids = input_ids_

    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.cuda()

    return input_ids, attention_mask
    
def tag_gen(tag):
    probs = F.softmax(tag, dim=-1) # beam * len * tagVOCAB
    max_probs, idx = probs.max(dim=-1) # beam * len
    return idx, max_probs, probs

def morph_gen(morph):
    probs = F.softmax(morph, dim=-1)
    max_probs, idx = probs.max(dim=-1)
    return idx, max_probs, probs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams", default=None, type=str)
    parser.add_argument("--model_binary", default=None, type=str)
    parser.add_argument("--testfile", default=None, type=str)
    parser.add_argument("--outputfile", default=None, type=str)
    parser.add_argument("--gold_len", default=False, type=bool)
    parser.add_argument("--length_beam_size", default=3, type=int)
    args = parser.parse_args()

    with open(args.hparams) as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)
        hparams.update(vars(args))

    args = argparse.Namespace(**hparams)

    inf = Model.load_from_checkpoint(args.model_binary, args=args)
    model = inf.model
    model = model.cuda() 
    model.eval()

    src_tok = inf.src_tok
    morph_tok = inf.morph_tok
    tag_tok = inf.tag_tok

    Fm = FileManager(beam_size=args.length_beam_size, file_name_list=["beam"])
  
    srcs = []
    f = open(args.testfile + '_src.txt', 'r', encoding="utf-8-sig")
    for src in f:
        srcs.append(src.strip())
    f.close()


    for src in tqdm(srcs, total=len(srcs)):
        beam = []
        for i in range(args.length_beam_size):
            beam.append([])

        sents = split(src, args.max_len)
        
        for sent in sents:
            input_ids = src_tok.encode(list(sent)) 
            input_ids, attention_mask = make_enc_input(input_ids, src_tok, args.max_len)
            # input_ids : [i,j,h, <pad>]
            # attention_mask : [1,1,1,0,0,0]

            attention_mask = torch.tensor(attention_mask)
            attention_mask = attention_mask.unsqueeze(0)
            attention_mask = attention_mask.cuda()

            input_ids = torch.tensor(input_ids)
            input_ids = input_ids.unsqueeze(0)
            input_ids = input_ids.cuda()

            enc_outputs, _, _ = model.encoder(input_ids, attention_mask)

            enc_outputs, attention_mask = duplicate_encoder_out(enc_outputs, attention_mask, enc_outputs.size(0), args.length_beam_size)

            morph_result = ["<bos>"]
            tag_result = ["<bos>"]
       
            for i in range(args.max_len):

                # Make decoder input through REsult
                morph_input_ids = morph_tok.encode(morph_result)
                tag_input_ids = tag_tok.encode(tag_result)
                morph_input_ids, morph_attention_mask = make_dec_input(morph_input_ids, morph_tok, args.max_len)
                tag_input_ids, tag_attention_mask = make_dec_input(tag_input_ids, tag_tok, args.max_len)

                morph_outputs = model.morph_decoder(
                    input_ids=morph_input_ids, 
                    attention_mask=morph_attention_mask,
                    encoder_hidden_states=enc_outputs,
                    encoder_attention_mask=attention_mask
                )
                
                tag_outputs = model.tag_decoder(
                    input_ids=tag_input_ids, 
                    attention_mask=tag_attention_mask,
                    encoder_hidden_states=enc_outputs,
                    encoder_attention_mask=attention_mask
                )

                morph_logits = model.morph_projection(morph_outputs[0])
                tag_logits = model.tag_projection(tag_outputs[0])

                morph_token, _, _ = morph_gen(morph_logits)
                tag_token, _, _ = tag_gen(tag_logits)

                morph_token[0][i+1:] = morph_tok.index("<pad>")
                tag_token[0][i+1:] = tag_tok.index("<pad>")

                # update result in AR way
                morph_inf = morph_tok.decode(morph_token[0].tolist(), False)
                tag_inf = tag_tok.decode(tag_token[0].tolist(), False)

                morph_result = ["<bos>"] + morph_inf[:i+1]
                tag_result = ["<bos>"] + tag_inf[:i+1]

                if morph_result[-1] == "<eos>" and tag_result[-1] == "<eos>":
                    break
            
            morph_result = morph_result[1:-1]
            tag_result = tag_result[1:-1]
            tag_result = Fm.force_BI_to_tag(tag_result)
            
                        
            sentence = Fm.make_sentence(morph_result, tag_result)
            beam[0] = sentence
            

        Fm.sent_to_beam_write("".join(beam[0]), 0)

    Fm.file_close()