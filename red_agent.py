"""
Red Agent (v2 — 80-dim state, 18 actions)
==========================================
DQN-based offensive agent with target network and configurable dimensions.
Architecture scaled to match the new 80-dim state and 18-action space.
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
)


class DQN(nn.Module):
    """
    Deep Q-Network with two hidden layers.

    Input  : STATE_DIM  = 80
    Output : ACTION_DIM = 18
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim,  HIDDEN_DIM_1), nn.ReLU(),
            nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2), nn.ReLU(),
            nn.Linear(HIDDEN_DIM_2, action_dim),
        )
        # Kaiming initialisation for ReLU networks
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RedAgent:
    """
    DQN-based offensive (Red) agent.

    Features
    --------
    - Target network updated every TARGET_UPDATE_FREQ training steps
    - ε-greedy exploration with exponential decay
    - Experience replay (deque-based memory)
    - 80-dim state observation → 18-action discrete output
    """

    def __init__(self, epsilon: float = None):
        self.epsilon       = epsilon if epsilon is not None else EPSILON_START
        self.epsilon_min   = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.gamma         = GAMMA
        self.memory        = deque(maxlen=MEMORY_SIZE)

        self.model        = DQN()
        self.target_model = copy.deepcopy(self.model)   # frozen target network
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
        """Store a (s, a, r, s') transition in replay memory."""
        self.memory.append((state, action, reward, next_state))

    # ── Training ─────────────────────────────────────────

    def train(self, batch_size: int = None) -> float | None:
        """Sample a mini-batch and run one gradient update step."""
        batch_size = batch_size or BATCH_SIZE
        if len(self.memory) < batch_size:
            return None

        batch  = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states      = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        rewards     = torch.FloatTensor(rewards)
        actions     = torch.LongTensor(actions)

        # Current Q-values for taken actions
        q_vals = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()

        # Target Q-values from frozen target network (Bellman update)
        with torch.no_grad():
            next_q  = self.target_model(next_states).max(1)[0]
        targets = rewards + self.gamma * next_q

        loss = self.loss_fn(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
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

    def save(self, path: str = "models/red_model.pth"):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str = "models/red_model.pth"):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.target_model.load_state_dict(self.model.state_dict())
