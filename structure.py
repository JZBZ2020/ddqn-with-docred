#coding: utf8 
from collections import namedtuple
from typing import List, Tuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'next_actions', 'done'))
Action = namedtuple('Action', ('sentence'))
State = namedtuple('State', ('claim', 'truth_evi',  'candidate',  'count'))
Sentence = namedtuple('Sentence', ('id', 'tokens'))
Claim = namedtuple('Claim', ('id', 'tokens'))
Evidence = List[Sentence]
EvidenceSet = List[Evidence]
DataSet = List[Tuple[Claim, int, Evidence, List[Sentence]]]

Sentence.__new__.__defaults__ = (None, None)
