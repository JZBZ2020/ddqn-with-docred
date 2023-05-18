import argparse
from tqdm import tqdm
import networkx as nx
import numpy as np
import torch
import random
import os
from structure import *
from config import set_com_args, set_dqn_args, set_bert_args
import argparse
from claims import preprocess,get_hrt
# import ujson as json
from transformers import AutoConfig, AutoModel, AutoTokenizer
import json
docred_rel2id = json.load(open('./dataset/rel2id.json', 'r'))

tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-cased"
    )
parser = argparse.ArgumentParser()
set_com_args(parser)  # 设置参数
set_dqn_args(parser)
set_bert_args(parser)
args = parser.parse_args()

def set_seed(args):
    random.seed(args.seed) # 66
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # if args.n_gpu > 0 and torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(args.seed)


def output_path_inf(path,G):
    """
    输出路径上共现句子信息
    :param path:
    :return:
    """
    path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    co_occur_sent_set = set()
    for edge in path_edges:
        co_occur_sent_set = co_occur_sent_set.union(set(G[edge[0]][edge[1]]['sent']))
    co_occur_sent_list = list(co_occur_sent_set)
    co_occur_sent_list.sort()
    return co_occur_sent_list

def get_entity_pair_path_info(doc, entity1, entity2):
    """
    可能问题：1.构造文件图结构，计算实体节点的eigenvector centrality策略，是否考虑边上共现句子信息数量
    2.扩展路径的策略
    :param doc:
    :param entity1:
    :param entity2:
    :return:
    """
    # Construct graph and calculate eigenvector centrality scores for each node
    G = nx.Graph()
    entities = doc['vertexSet']
    # print(len(doc['vertexSet'])) #文档中所有实体个数
    for en_id,entity in enumerate(entities):
        entity_id = entity[0]['name']
        G.add_node(en_id)
    for sent_idx, sent in enumerate(doc['sents']):
        nodes_in_sent = []
        for entity_id,entity in enumerate(entities):
            sent_id_ = []  # 元素数量为0或1
            for mention in entity:
                sent_id = mention['sent_id']
                if sent_id == sent_idx:  # 在entity中只要有一个提及sent_id==ent_idx，就把entity_id加进去，仅添加一次
                    if sent_id not in sent_id_:
                        sent_id_.append(sent_id)
                        nodes_in_sent.append(entity_id)
        nodes_in_sent = list(set(nodes_in_sent))
        if len(nodes_in_sent)<=1:
            continue
        else:# 如果没有构造边，边包含两个属性sent,weight;存在边，更新边的信息
            for i in range(len(nodes_in_sent)):
                for j in range(i+1, len(nodes_in_sent)):
                    if not G.has_edge(nodes_in_sent[i],nodes_in_sent[j]):
                        G.add_edge(nodes_in_sent[i], nodes_in_sent[j], sent=[sent_idx],weight =1)
                    else:
                        G[nodes_in_sent[i]][nodes_in_sent[j]]['sent'].append(sent_idx)
                        G[nodes_in_sent[i]][nodes_in_sent[j]]['weight']+=1
    eig_centrality = nx.eigenvector_centrality(G,max_iter=100000,weight="weight")

    # Beam search with width 2 to find path between entity nodes
    initial_path = [(0,[entity1])]

    beam_width = 2
    while True:

        paths = []
        for curr_path in initial_path:
            curr_cost,curr_node = curr_path
            for neighbor in list(G.neighbors(curr_node[-1])):
                if neighbor not in curr_node:
                    new_cost = curr_cost + np.log(eig_centrality[neighbor]) if eig_centrality[neighbor] != 0 else curr_cost
                    new_path = curr_node + [neighbor]  # None
                    paths.append((new_cost, new_path))

        if len(paths)==0:
            #头结点或两条路径无法扩展或无法到达尾结点
            # all_entities_name = search_samemeaning_entities(doc,entity1)+search_samemeaning_entities(doc,entity2)
            co_occur_sent_list = list(set([mention['sent_id'] for id in [entity1,entity2] for mention in doc['vertexSet'][id]]))
            co_occur_sent_list.sort()
            return co_occur_sent_list
        else:          # 输出候选路径上尾实体信息，如果存在目标实体：如果目标候选路径为1，输出路径上的共现句子信息；如果目标候选路径为多个，输出分值最大的。
            candidate_path_tail = [sp[1][-1] for sp in paths]
            if entity2 in candidate_path_tail:
                score_list =[]
                path_list = []
                #记录候选路径中所有尾实体为目标实体的路径的信息
                for i,(score,path) in enumerate(paths):
                    if path[-1]==entity2:
                        score_list.append(score)
                        path_list.append(path)
                max_score = max(score_list)
                # 目标候选路径中最大分值的路径不止一个
                if score_list.count(max_score)!=1:
                    path_sents_info = []
                    for i,score in enumerate(score_list):
                        if score == max_score:
                            path_sents_info += output_path_inf(path_list[i],G)
                    return list(set(path_sents_info))
                else:
                    path_sents_info = output_path_inf(path_list[score_list.index(max_score)],G)
                    return path_sents_info
        paths.sort(key=lambda a:a[0],reverse=True)# descent sort
        initial_path = paths[:beam_width]
    # return co_occur_sent_list


def sentences_token(args, sents_id):
    """
    获取文档每句话模型计算后的tokens
    :param args: 预训练模型参数
    :param sents_id: 每句话未处理之前的id列表
    :return:
    """
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type
    # inputs = tokenizer.batch_encode_plus(sents_id)
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    set_seed(args)
    input_id = [tokenizer.convert_tokens_to_ids(sents) for sents in sents_id]
    input_id = [tokenizer.build_inputs_with_special_tokens(sents) for sents in input_id] # [101,...,102]
    max_len = max([len(s) for s in input_id])
    input_ids = [s + [0] * (max_len - len(s)) for s in input_id]
    input_mask = [[1.0] * len(s) + [0.0] * (max_len - len(s)) for s in input_id]
    input_ids = torch.tensor(input_ids,dtype=torch.long)
    input_mask = torch.tensor(input_mask,dtype=torch.float)
    # bert
    output = model(
        input_ids=input_ids,
        attention_mask=input_mask,
        output_attentions=True,
    )
    sequence_output = output[1]  # [8,50,768] [8,768]
    return sequence_output


def read_docred(file_in, tokenizer, max_seq_length=1024):
    i_line = 0 # 文档个数
    pos_samples = 0 # 所有文档中存在关系正例实体对个数
    neg_samples = 0 # 所有文档中存在关系负例实体对个数
    features = [] # 返回所有文档的input_ids,entity_pos,relations,hts,title
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)

    for sample in tqdm(data, desc="Example"):
        sents = [] #文档中标签化后的token，长列表
        sents_ = []
        sent_map = [] #列表字典， 文档中每个句子token_id映射

        entities = sample['vertexSet']
        entity_start, entity_end = [], [] #文档中实体提及的起始位置和终止位置
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]# 实体提及句子id
                pos = mention["pos"]# 实体提及位置
                entity_start.append((sent_id, pos[0],))
                entity_end.append((sent_id, pos[1] - 1,))#结束下标
        for i_s, sent in enumerate(sample['sents']):
            # 对句子中出现的实体对前后加上特殊token[*]
            new_map = {} # 字典，句子中token_id映射
            # key:token在句中位置下标 value:累加已处理token长列表sents的长度
            # sents_.append([])
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                if (i_s, i_t) in entity_start:
                    tokens_wordpiece = ["*"] + tokens_wordpiece
                if (i_s, i_t) in entity_end:
                    tokens_wordpiece = tokens_wordpiece + ["*"]
                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
                # sents_[-1].extend(tokens_wordpiece)
            new_map[i_t + 1] = len(sents)
            sent_map.append(new_map)
        # 获取每句话的内容，整合完整的str
        sents_str = []
        for i_s, sent in enumerate(sample['sents']):
            sent_str = ' '.join(sent)
            sents_str.append([sent_str])
            sents_.append([])
            for i_t, token in enumerate(sent):
                tokens_word =tokenizer.tokenize(token)
                sents_[-1].extend(tokens_word)



        sents_token = sentences_token(args=args,sents_id=sents_)

        train_triple = {}
        # 处理文档中多个实体对以及实体对多关系
        # key:(h,t) value:[{r,evidence},]
        if "labels" in sample:
            for label in sample['labels']:
                evidence = label['evidence']
                r = int(docred_rel2id[label['r']])
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r, 'evidence': evidence}]
                else: # 实体对之间存在多关系
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r, 'evidence': evidence})

        entity_pos = [] # 列表列表，文档实体提及在sents的位置下标元组
        for e in entities:
            entity_pos.append([])
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                entity_pos[-1].append((start, end,))

        sents = sents[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
        # attention_mask
        attention_mask=torch.tensor([1.0]*len(input_ids),dtype=torch.float)
        input_ids = torch.tensor(input_ids,dtype=torch.long)
        sequence_output, attention = preprocess(args,input_ids,attention_mask)


        relations, hts, evidences,actions = [], [], [],[]
        # relation:每个实体对多个关系列表（出现的关系标1）（正例）；实体之间不存在关系[1,0,0..](负例)
        for h, t in train_triple.keys():
            relation = [0] * len(docred_rel2id)# 关系类别长度的全0列表
            evidences_ = [[] for _ in range(len(docred_rel2id))]
            for mention in train_triple[h, t]:
                relation[mention["relation"]] = 1

                evidence = [(index,list(sents_token[index]),) for index in mention['evidence'] ]
                evidences_[mention["relation"]] = evidence
            relations.append(relation)
            hts.append([h, t])

            evidences.append(evidences_)
            ht_actions = [(index,list(sents_token[index]),) for index in get_entity_pair_path_info(sample,h,t) ] # [(id,[]),]
            actions.append(ht_actions)
            pos_samples += 1 # 实体对[h,t]关系为正

        for h in range(len(entities)):
            for t in range(len(entities)):
                if h != t and [h, t] not in hts: #h,t之间不存在关系
                    relation = [1] + [0] * (len(docred_rel2id) - 1)
                    evidences_ = [[] for _ in range(len(docred_rel2id))]
                    relations.append(relation)
                    evidences.append(evidences_)
                    hts.append([h, t])

                    ht_actions = [(index,list(sents_token[index]),) for index in get_entity_pair_path_info(sample, h, t)]
                    actions.append(ht_actions)
                    neg_samples += 1
        # 验证文档中实体对个数
        assert len(relations) == len(entities) * (len(entities) - 1)

        final_hts = get_hrt(args,sequence_output, attention, entity_pos, hts)
        claims = [final_hts[id] for id,ht in enumerate(hts)]



        i_line += 1
        feature = {'input_ids': input_ids,
                   'entity_pos': entity_pos,#列表元组，根据vertexSet确定每个实体提及在文档长列表sents中的位置(start,end)
                   'labels': relations, # 列表列表，根据labels确定实体对[h,t]关系列表
                   'hts': hts, #列表列表，[h,t]，用于relation关系列表的实体对映射
                   'evidences': evidences,# 列表列表,根据labels确定实体对[h,t]证据列表
                   'actions': actions,# 列表列表，每个ht候选动作集
                   'title': sample['title'],
                   # 'sent_map': sent_map # 列表字典
                   'claims' : claims # 每个ht嵌入向量
                   }
        features.append(feature)

    print("# of documents {}.".format(i_line))
    print("# of positive examples {}.".format(pos_samples))
    print("# of negative examples {}.".format(neg_samples))
    return features

class Dataset():
    def __init__(self,filedir):
        self.filedir = filedir
        self.tokenizer= AutoTokenizer.from_pretrained(
        "bert-base-cased"
    )
        self.data = read_docred(self.filedir,
                                  self.tokenizer, max_seq_length=1024)
    def load_data(self):
        data_list = [] # [{},]
        for doc_id,dict in enumerate(self.data):
            # [map(lambda x: x[1][i] if x[0] in ['hts','labels','evidences','actions','claims']  ,dict.items() ) for i in range(len(dict['hts']))]

            for idx in range(len(dict['hts'])):
                localdata = {}
                for key,value in dict.items():
                    if key in ['hts','labels','evidences','claims']:
                        localdata[key] = value[idx]
                    elif key in ['actions']:
                        localdata[key]=[Sentence(id=(dict['title'],int(id)),tokens=token) for id,token in value[idx]]
                data_list.append(localdata)
        return data_list


    # 把列表字典中claims,evidences,actions,提取
    def __getitem__(self, index: int):
        littledict = self.load_data()[index]
        # evidences
        evidences = []
        for i,num in enumerate(littledict['labels']):
            if num ==1 and littledict['evidences'][i]!=[]:
                evidences.extend(list(map(lambda x:Sentence(id=x[0],tokens=x[1] )   ,littledict['evidences'][i])))
        next_actions = [Action(sentence=act) for act in littledict['actions']]
        state = State(claim=Claim(id=littledict['hts'],tokens=littledict['claims']),
        truth_evi=evidences,
        candidate=[],
        count=0)

        return state,next_actions

    def __len__(self):
        return len(self.data)

def load_and_process_data(datadir:str,filename:str):
    filedir = os.path.join(datadir,filename)
    dataset = Dataset(filedir)
    return dataset


def collate_fn(batch):
    batch_state, batch_actions = [], []
    for state, actions in batch:
        batch_state.append(state)  #[(state),]
        batch_actions.append(actions) # [[Action,],]
    assert len(batch_state) == len(batch_actions)
    return batch_state, batch_actions


