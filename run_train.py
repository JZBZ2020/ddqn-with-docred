import os
import argparse
import json
import pickle
import random
from typing import List, Tuple
from tqdm import tqdm, trange
from time import sleep
from functools import reduce
import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
from state import load_and_process_data,Dataset,collate_fn
from config import set_com_args, set_dqn_args, set_bert_args
from logger import logger
from transformer_dqn import TransformerDQN
from memory import ReplayMemory
from environment import TargetEnv
def set_seed(args):
    random.seed(args.seed) # 66
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def train(args,
          agent,
          train_dataset: Dataset,
          epochs_trained: int = 0,
          acc_loss_trained_in_current_epoch: float = 0,
          steps_trained_in_current_epoch: int = 0,
          losses_trained_in_current_epoch: List[float] = []) -> None:

    logger.info('Training')
    env = TargetEnv(K=args.max_evi_size) # 环境更新
    # ReplayMemory
    memory = ReplayMemory(args.capacity)
    data_loader = DataLoader(train_dataset,
                             # num_workers=1,
                             num_workers=0,
                             collate_fn=collate_fn,
                             batch_size=args.train_batch_size,
                             shuffle=True)
    train_iterator = trange(int(args.num_train_epochs), desc='Epoch') #100
    for epoch in train_iterator:
        if epochs_trained > 0:
            epochs_trained -= 1
            sleep(0.1)
            continue

        epoch_iterator = tqdm(data_loader,
                              desc='Loss')

        # log_per_steps = len(epoch_iterator) // 5
        # log_per_steps = 100

        t_loss, t_steps = acc_loss_trained_in_current_epoch, steps_trained_in_current_epoch
        t_losses, losses = losses_trained_in_current_epoch, [] #每隔一些批次，记录当前每次训练损失，当前每次训练损失
        # get one batch data
        for step, (batch_state, batch_actions) in enumerate(epoch_iterator):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            cur_iter, tmp_iter = 0, random.randint(1, 5) # 记录该批次数据训练次数
            while True:
                # 选择动作 ([(Action)*batch_size],[float,])
                batch_selected_action, _ = agent.select_action(batch_state,
                                                               batch_actions,
                                                               net=agent.q_net,
                                                               is_greedy=False)
                batch_state_next, batch_actions_next = [], []
                # updata every state and push them into memory
                for state, selected_action, actions in zip(batch_state,
                                                           batch_selected_action,
                                                           batch_actions):
                    state_next, reward, done = env.step(state, selected_action)  # 更新状态，奖励
                    actions_next = \
                        list(filter(lambda x: selected_action.sentence.id != x.sentence.id, actions))
                    done = done if len(actions_next) else True
                    # 默认保持当前数据的done，若无后续动作集，设为完成
                    data = {'item': Transition(state=state,
                                               action=selected_action,
                                               next_state=state_next,
                                               reward=reward,
                                               next_actions=actions_next,
                                               done=done)}
                    if done: continue
                    # 数据已经完成，不保留该数据进入下个循环；当批次数据全部完成，跳出while循环
                    batch_state_next.append(state_next)
                    batch_actions_next.append(actions_next)

                batch_state = batch_state_next
                batch_actions = batch_actions_next
                cur_iter += 1
                # sample batch data and optimize model
                if len(memory) >= args.train_batch_size:
                    batch_rl = memory.sample(args.train_batch_size)
                    isweights = None
                    # list which len quals with size of sampled data from memory,in order to weighted every loss
                    # 每个被采数据的损失列表，损失均值，损失权重均值
                    td_error, [loss, wloss] = agent.update(batch_rl,
                                                           isweights,
                                                           )
                    t_loss += loss
                    t_steps += 1
                    losses.append(loss)
                    # tqdm descrtption
                    epoch_iterator.set_description('%.4f (%.4f)' % (wloss, loss))
                    epoch_iterator.refresh()
                # if all data in batch is done,next sample new batch data
                if len(batch_state) == 0: break
            # every 10 batch data before update target net
            if step and step % args.target_update == 0:
                agent.soft_update_of_target_network(args.tau)

            # save q_net,optimizer,scheduler,memory,a serious loss
            if step and step % args.save_steps == 0:
                save_dir = os.path.join(args.output_dir, f'{epoch}-{step}-{t_loss / t_steps}')
                agent.save(save_dir)
                with open(os.path.join(save_dir, 'memory.pk'), 'wb') as fw:
                    pickle.dump(memory, fw)
                with open(os.path.join(save_dir, 'loss.txt'), 'w') as fw:
                    fw.write('\n'.join(list(map(str, losses))))
                t_losses.extend(losses)
                losses = []

        epoch_iterator.close() # 一次大循环

        acc_loss_trained_in_current_epoch = 0
        losses_trained_in_current_epoch = []

        save_dir = os.path.join(args.output_dir, f'{epoch + 1}-0-{t_loss / t_steps}')
        if steps_trained_in_current_epoch == 0:
            agent.save(save_dir)
            with open(os.path.join(save_dir, 'memory.pk'), 'wb') as fw:
                pickle.dump(memory, fw)
            with open(os.path.join(save_dir, 'loss.txt'), 'w') as fw:
                fw.write('\n'.join(list(map(str, losses))))
            losses = []


    train_iterator.close()

def evaluate(args: dict, agent, datafilename):
    agent.eval()
    evl_data = load_and_process_data(args.data_dir,'dev.json')
    # 加载dev 每次提取一个data
    data_loader = DataLoader(evl_data, collate_fn=collate_fn, batch_size=args.test_batch_size, shuffle=False)
    epoch_iterator = tqdm(data_loader,
                          disable=args.local_rank not in [-1, 0])

    logger.info('Evaluating')
    with torch.no_grad():
        """
        读取一批数据，迭代贪心选择最大动作，更新状态，直到所有数据收敛；
        每次迭代，已经到头的数据不带入下一次动作选择，记录每次迭代q值和状态；
        对每个数据进行分类，按照q值最大选取状态（证据集）
        """
        step = 0
        pred,pred_list = [],[]
        for batch_state, batch_actions in epoch_iterator:
            q_value_seq, state_seq = [], []
            # mask_list = [True] * len(batch_state)
            all_qvalue_state = []

            for _ in range(max([len(actions) for actions in batch_actions])):
                batch_selected_action, batch_q_value = \
                    agent.select_action(batch_state,
                                        batch_actions,
                                        net=agent.q_net,
                                        is_greedy=True)

                batch_state_next, batch_actions_next = [], []

                for state, selected_action, actions,q_value in zip(batch_state,
                                                           batch_selected_action,
                                                           batch_actions,batch_q_value):
                    state_next = TargetEnv.new_state(state, selected_action)
                    all_qvalue_state.append((q_value,state_next))

                    actions_next = \
                        list(filter(lambda x: selected_action.sentence.id != x.sentence.id, actions))

                    if len(actions_next) == 0:
                        continue
                    else:
                        batch_state_next.append(state_next)
                        batch_actions_next.append(actions_next)

                # state_seq.append(batch_state_next)

                if len(batch_actions_next) == 0:
                    break

                batch_state = batch_state_next
                batch_actions = batch_actions_next
            # 对每个数据进行分类，按照q值最大选取状态（证据集）
            batch_qvalue_state_dict = {}
            all_hts= list(map(lambda x : x.claim.id ,batch_state))
            for hts in all_hts:
                batch_qvalue_state_dict[hts]=[]
            for qvalue,state in all_qvalue_state:
                batch_qvalue_state_dict[state.claim.id].append((qvalue,state))
            batch_pred_state = []
            for hts in all_hts:
                qvalue = list(map(lambda a:a[0] ,batch_qvalue_state_dict[hts]))
                states = list(map(lambda a:a[1] ,batch_qvalue_state_dict[hts]))
                batch_pred_state.append(states[qvalue.index(max(qvalue))])
            pred.extend(batch_pred_state)
            step+=1
            if step and step % 10 ==0:
                pred_list.append(pred)
                with open(os.path.join(args.save_path,'pred_state.txt'),'w') as fw:
                    pickle.dump(*pred,fw)
                pred = []

def run_dqn(args):
    Agent = TransformerDQN# TransformerDQN
    agent = Agent(args)
    agent.to(args.device)

    train_dataset = load_and_process_data(args.data_dir,'train_annotated.json')
    epochs_trained = 0
    acc_loss_trained_in_current_epoch = 0
    steps_trained_in_current_epoch = 0
    losses_trained_in_current_epoch = []
    train(args,
          agent,
          train_dataset,
          epochs_trained,
          acc_loss_trained_in_current_epoch,
          steps_trained_in_current_epoch,
          losses_trained_in_current_epoch)


def main():
    parser = argparse.ArgumentParser()
    set_com_args(parser)# 设置参数
    set_dqn_args(parser)
    set_bert_args(parser)
    args = parser.parse_args()

    args.logger = logger
    args.do_lower_case = bool(args.do_lower_case)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    logger.info(vars(args))

    # Set seed
    set_seed(args)

    logger.info("Training/evaluation parameters %s", args)

    # run dqn
    run_dqn(args)

if __name__ == '__main__':
    main()
