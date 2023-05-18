import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoConfig, AutoModel, AutoTokenizer
from opt_einsum import contract

def process_long_input(model, input_ids, attention_mask, start_tokens, end_tokens):
    # Split the input to 2 overlapping chunks. Now BERT can encode inputs of which the length are up to 1024.
    c = input_ids.size().detach().cpu().numpy() #[4, 360]
    start_tokens = torch.tensor(start_tokens).to(input_ids) # tensor([101], device='cuda:0')
    end_tokens = torch.tensor(end_tokens).to(input_ids)
    len_start = start_tokens.size(0) # 1
    len_end = end_tokens.size(0)
    if c <= 512:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        sequence_output = output[0] #
        attention = output[-1][-1]#
    else:
        new_input_ids, new_attention_mask, num_seg = [], [], []
        seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
        for i, l_i in enumerate(seq_len):
            if l_i <= 512:
                new_input_ids.append(input_ids[i, :512])
                new_attention_mask.append(attention_mask[i, :512])
                num_seg.append(1)
            else:
                input_ids1 = torch.cat([input_ids[i, :512 - len_end], end_tokens], dim=-1)
                input_ids2 = torch.cat([start_tokens, input_ids[i, (l_i - 512 + len_start): l_i]], dim=-1)
                attention_mask1 = attention_mask[i, :512]
                attention_mask2 = attention_mask[i, (l_i - 512): l_i]
                new_input_ids.extend([input_ids1, input_ids2])
                new_attention_mask.extend([attention_mask1, attention_mask2])
                num_seg.append(2)
        input_ids = torch.stack(new_input_ids, dim=0)
        attention_mask = torch.stack(new_attention_mask, dim=0)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        sequence_output = output[0]
        attention = output[-1][-1]
        i = 0
        new_output, new_attention = [], []
        for (n_s, l_i) in zip(num_seg, seq_len):
            if n_s == 1:
                output = F.pad(sequence_output[i], (0, 0, 0, c - 512))
                att = F.pad(attention[i], (0, c - 512, 0, c - 512))
                new_output.append(output)
                new_attention.append(att)
            elif n_s == 2:
                output1 = sequence_output[i][:512 - len_end]
                mask1 = attention_mask[i][:512 - len_end]
                att1 = attention[i][:, :512 - len_end, :512 - len_end]
                output1 = F.pad(output1, (0, 0, 0, c - 512 + len_end))
                mask1 = F.pad(mask1, (0, c - 512 + len_end))
                att1 = F.pad(att1, (0, c - 512 + len_end, 0, c - 512 + len_end))

                output2 = sequence_output[i + 1][len_start:]
                mask2 = attention_mask[i + 1][len_start:]
                att2 = attention[i + 1][:, len_start:, len_start:]
                output2 = F.pad(output2, (0, 0, l_i - 512 + len_start, c - l_i))
                mask2 = F.pad(mask2, (l_i - 512 + len_start, c - l_i))
                att2 = F.pad(att2, [l_i - 512 + len_start, c - l_i, l_i - 512 + len_start, c - l_i])
                mask = mask1 + mask2 + 1e-10
                output = (output1 + output2) / mask.unsqueeze(-1)
                att = (att1 + att2)
                att = att / (att.sum(-1, keepdim=True) + 1e-10)
                new_output.append(output)
                new_attention.append(att)
            i += n_s
        sequence_output = torch.stack(new_output, dim=0)
        attention = torch.stack(new_attention, dim=0)
    return sequence_output, attention

def preprocess(args,input_ids,attention_mask):
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-cased"
    )
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    if config.transformer_type == "bert":
        start_tokens = [config.cls_token_id]  # [101]

        end_tokens = [config.sep_token_id]  # [102]

    elif config.transformer_type == "roberta":
        start_tokens = [config.cls_token_id]
        end_tokens = [config.sep_token_id, config.sep_token_id]
    sequence_output, attention = process_long_input(model, input_ids, attention_mask, start_tokens, end_tokens)

    return sequence_output,attention


def get_hrt(args, sequence_output, attention, entity_pos, hts):
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    offset = 1 if args.transformer_type in ["bert", "roberta"] else 0
    h, _, c = attention.size()
    # 遍历一个文档中所有实体
    entity_embs, entity_atts = [], []
    for e in entity_pos:
        if len(e) > 1:  # 多次出现的实体
            e_emb, e_att = [], []
            # 实体所有提及
            for start, end in e:
                if start + offset < c:
                    # In case the entity mention is truncated due to limited max seq length.
                    e_emb.append(sequence_output[ start + offset])
                    e_att.append(attention[ :, start + offset])
            if len(e_emb) > 0:  # 公式2
                e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                e_att = torch.stack(e_att, dim=0).mean(0)
            else:
                e_emb = torch.zeros(config.hidden_size).to(sequence_output)
                e_att = torch.zeros(h, c).to(attention)
        else:
            start, end = e[0]
            if start + offset < c:
                e_emb = sequence_output[start + offset]
                e_att = attention[ :, start + offset]
            else:
                e_emb = torch.zeros(config.hidden_size).to(sequence_output)
                e_att = torch.zeros(h, c).to(attention)
        entity_embs.append(e_emb)
        entity_atts.append(e_att)

    entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
    entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]

    ht_i = torch.LongTensor(hts).to(sequence_output.device)  # [n_e*(n_e-1),2]
    hs = torch.index_select(entity_embs, 0, ht_i[:, 0])  # 头尾实体嵌入[n_e*(n_e-1),768]
    ts = torch.index_select(entity_embs, 0, ht_i[:, 1])  # 公式7

    h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
    t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
    ht_att = (h_att * t_att).mean(0)  # 公式3  # q_so
    ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)  # a_so [n_e*(n_e-1),360]
    rs = contract("ld,rl->rd", sequence_output, ht_att)  # 公式4 c_so  公式7

    can_vec = torch.stack([hs,ts,rs],dim=1)
    return can_vec