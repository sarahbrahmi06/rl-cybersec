"""
Blue Agent (v2 — 80-dim state, 18 actions, Prioritised Replay)
===============================================================
DQN-based defensive agent with:
- Same DQN architecture as RedAgent (80→18)
- Prioritised Experience Replay (PER): transitions with high anomaly scores
  are sampled more often, biasing the agent toward learning under attack
"""

import numpy as np
import random
from collections import deque
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    STATE_DIM, ACTION_DIM, LEARNING_RATE, GAMMA,
    EPSILON_START, EPSILON_MIN, EPSILON_DECAY,
    MEMORY_SIZE, BATCH_SIZE, TARGET_UPDATE_FREQ,
    HIDDEN_DIM_1, HIDDEN_DIM_2,
    N_INTERFACES,
)

# Anomaly index slice in state vector (last N dims = anomaly scores)
_ANOMALY_START = STATE_DIM - N_INTERFACES   # = 70


class DQN(nn.Module):
    """
    Deep Q-Network with two hidden layers.

    Input  : STATE_DIM  = 80
    Output : ACTION_DIM = 18
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim,    HIDDEN_DIM_1), nn.ReLU(),
            nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2), nn.ReLU(),
            nn.Linear(HIDDEN_DIM_2, action_dim),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PrioritisedReplayBuffer:
    """
    Simple prioritised experience replay buffer.

    Transitions recorded when the observed anomaly score (last N dims of state)
    is above a threshold are stored in a "hot" priority queue sampled at higher
    frequency.  All other transitions go into the normal replay memory.

    Sampling ratio: 30 % from hot (anomaly) memory, 70 % from normal memory
    (subject to availability).
    """

    ANOMALY_PRIORITY_THRESHOLD = 0.45   # anomaly score that triggers priority
    HOT_RATIO = 0.30                    # fraction of batch from priority queue

    def __init__(self, maxlen: int = MEMORY_SIZE):
        self.normal = deque(maxlen=maxlen)
        self.hot    = deque(maxlen=maxlen // 2)

    def append(self, transition: tuple):
        """Store transition; route to hot queue if anomaly score is elevated."""
        state = transition[0]
        anomaly_mean = float(np.mean(state[_ANOMALY_START:]))

        if anomaly_mean >= self.ANOMALY_PRIORITY_THRESHOLD:
            self.hot.append(transition)
        else:
            self.normal.append(transition)

    def sample(self, batch_size: int) -> list:
        """
        Return a mixed batch.  Falls back gracefully if one queue is too small.
        """
        n_hot    = min(int(batch_size * self.HOT_RATIO), len(self.hot))
        n_normal = min(batch_size - n_hot, len(self.normal))

        batch  = random.sample(list(self.hot),    n_hot)   if n_hot    > 0 else []
        batch += random.sample(list(self.normal), n_normal) if n_normal > 0 else []

        # If still short (both queues small), top up from whatever is available
        shortage = batch_size - len(batch)
        if shortage > 0:
            pool = list(self.hot) + list(self.normal)
            batch += random.choices(pool, k=min(shortage, len(pool)))

        return batch

    def __len__(self) -> int:
        return len(self.normal) + len(self.hot)


class BlueAgent:
    """
    DQN-based defensive (Blue) agent.

    Features
    --------
    - Target network updated every TARGET_UPDATE_FREQ training steps
    - ε-greedy exploration with exponential decay
    - Prioritised Experience Replay (anomaly-weighted sampling)
    - 80-dim state observation → 18-action discrete output
    """

    def __init__(self, epsilon: float = None):
        self.epsilon       = epsilon if epsilon is not None else EPSILON_START
        self.epsilon_min   = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.gamma         = GAMMA
        self.memory        = PrioritisedReplayBuffer(maxlen=MEMORY_SIZE)

        self.model        = DQN()
        self.target_model = copy.deepcopy(self.model)
        self.optimizer    = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.loss_fn      = nn.MSELoss()

        self._train_steps = 0

    # ── Inference ────────────────────────────────────────

    def act(self, state: np.ndarray) -> int:
        """ε-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randrange(ACTION_DIM)
        with torch.no_grad():
            q = self.model(torch.FloatTensor(state))
        return q.argmax().item()

    # ── Memory ───────────────────────────────────────────

    def remember(self, state, action, reward, next_state):
        """Store a (s, a, r, s') transition in the prioritised replay buffer."""
        self.memory.append((state, action, reward, next_state))

    # ── Training ─────────────────────────────────────────

    def train(self, batch_size: int = None) -> float | None:
        """Sample a prioritised mini-batch and run one gradient update step."""
        batch_size = batch_size or BATCH_SIZE
        if len(self.memory) < batch_size:
            return None

        batch  = self.memory.sample(batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states      = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        rewards     = torch.FloatTensor(rewards)
        actions     = torch.LongTensor(actions)

        # Current Q-values for taken actions
        q_vals = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q-values from frozen target network
        with torch.no_grad():
            next_q  = self.target_model(next_states).max(1)[0]
        targets = rewards + self.gamma * next_q

        loss = self.loss_fn(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Periodically sync target network
        self._train_steps += 1
        if self._train_steps % TARGET_UPDATE_FREQ == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()

    # ── Persistence ──────────────────────────────────────

    def save(self, path: str = "models/blue_model.pth"):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str = "models/blue_model.pth"):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.target_model.load_state_dict(self.model.state_dict())
