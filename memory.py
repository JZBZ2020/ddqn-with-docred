#!/usr/bin/python3
# coding: utf-8
import random
import math
import numpy as np
from collections import defaultdict
from typing import List, Tuple
from structure import Transition

import pdb

from logger import logger


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory = [None] * capacity
        self.position = 0
        self.length = 0

    def reset(self) -> None:
        self.position = 0
        self.length = 0
        self.sequences = {}

    def push(self, item: Transition) -> None:
        self.memory[self.position] = item
        self.position = (self.position + 1) % self.capacity
        self.length = min(self.length + 1, self.capacity)

    def sample(self, batch_size: int) -> List[Transition]:
        batch = random.sample(self.memory[:self.length],
                               min(batch_size, self.length))
        return batch

    def __len__(self) -> int:
        return self.length
