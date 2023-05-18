#!/usr/bin/env python3
# coding=utf-8
from structure import State, Evidence, Action
from typing import Tuple, Set


def get_id_from_evidence(e) :
    return list(set(map(lambda sent: sent.id, e))) # 提取所有证据的id并去重


class BaseEnv:
    def __init__(self, K=3):
        self.K = K

    def jaccard(self, e1: Evidence, e2: Evidence) -> float:
        sents1 = get_id_from_evidence(e1)
        sents2 = get_id_from_evidence(e2)
        return (len(sents1 & sents2) + 1.0) / (len(sents1 | sents2) + 1.0)

    @classmethod
    def new_state(cls, state: State, action: Action) -> State:
                return State(claim=state.claim,
                 truth_evi = state.truth_evi,
                 # evidence_set=state.evidence_set,
                 candidate=state.candidate + [action.sentence],

                 count=state.count + 1)

    def is_done(self, state: State) -> bool:
        return state.count == self.K

    def score(self, state: State) -> float:
        return NotImplementedError()

    def reward(self, state_now: State, state_next: State) -> float:
        return NotImplementedError()

    def step(self, state: State, action: Action) -> Tuple[State, float, bool]:
        return NotImplementedError()


class TargetEnv(BaseEnv):
    def __init__(self,  K=3):
        super(FeverEnv, self).__init__(K)
        # self.label2id = label2id

    def reward(self, state: State, action: Action)-> float:
        truth_evi_id = get_id_from_evidence(state.truth_evi)
        # candidate id
        candidate_id = [sent.id for sent in state.candidate]
        if self.done(state)!=True and action.sentence.id in truth_evi_id:
            return 1.0
        elif self.done(state) == True and all([map(lambda x: x in truth_evi_id, candidate_id)]):
            return 1.0
        else:
            return 0



    def step(self, state: State, action: Action) -> Tuple[State, float, bool]:
        state_next = BaseEnv.new_state(state, action)
        done = self.is_done(state_next)
        return state_next, self.reward(state_next, action), done








