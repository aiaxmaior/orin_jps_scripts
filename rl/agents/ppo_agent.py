"""
PPO (Proximal Policy Optimization) agent for ADAS/DMS.

Implements PPO for safe driving policy learning with:
- Clipped objective for stable updates
- Separate actor and critic networks
- GAE (Generalized Advantage Estimation)
- Safety constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Dict, List, Tuple, Optional


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.

    Architecture:
        - Shared feature extractor
        - Actor head (policy network)
        - Critic head (value network)
    """

    def __init__(
        self,
        state_dim: int,
        action_space_type: str = "discrete",
        num_actions: int = 175,  # For discrete
        action_dim: int = 3,  # For continuous (throttle, brake, steering)
        hidden_dims: List[int] = [512, 256, 128],
    ):
        """
        Args:
            state_dim: Dimension of state vector
            action_space_type: 'discrete' or 'continuous'
            num_actions: Number of discrete actions
            action_dim: Dimension of continuous action space
            hidden_dims: Hidden layer sizes
        """
        super().__init__()

        self.action_space_type = action_space_type
        self.num_actions = num_actions
        self.action_dim = action_dim

        # Shared feature extractor
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),  # Stabilizes training
            ])
            prev_dim = hidden_dim

        self.shared_net = nn.Sequential(*layers)

        # Actor head (policy)
        if action_space_type == "discrete":
            self.actor = nn.Sequential(
                nn.Linear(prev_dim, prev_dim // 2),
                nn.ReLU(),
                nn.Linear(prev_dim // 2, num_actions),
            )
        else:  # continuous
            # Output mean and log_std for each action dimension
            self.actor_mean = nn.Sequential(
                nn.Linear(prev_dim, prev_dim // 2),
                nn.ReLU(),
                nn.Linear(prev_dim // 2, action_dim),
                nn.Tanh(),  # Bounded actions [-1, 1]
            )
            self.actor_log_std = nn.Parameter(
                torch.zeros(action_dim)
            )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Linear(prev_dim // 2, 1),
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            action_logits/mean: Policy output
            value: State value V(s)
        """
        features = self.shared_net(state)

        if self.action_space_type == "discrete":
            action_logits = self.actor(features)
            value = self.critic(features).squeeze(-1)
            return action_logits, value
        else:
            action_mean = self.actor_mean(features)
            value = self.critic(features).squeeze(-1)
            return action_mean, value

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            state: State tensor [batch, state_dim]
            deterministic: Use greedy/mean action (for evaluation)

        Returns:
            action: Sampled action
            log_prob: Log probability of action
            value: State value V(s)
        """
        if self.action_space_type == "discrete":
            action_logits, value = self.forward(state)
            action_probs = F.softmax(action_logits, dim=-1)

            if deterministic:
                action = torch.argmax(action_probs, dim=-1)
            else:
                dist = Categorical(action_probs)
                action = dist.sample()

            log_prob = F.log_softmax(action_logits, dim=-1)
            log_prob = log_prob.gather(-1, action.unsqueeze(-1)).squeeze(-1)

            return action, log_prob, value

        else:  # continuous
            action_mean, value = self.forward(state)
            action_std = torch.exp(self.actor_log_std)

            if deterministic:
                action = action_mean
                log_prob = torch.zeros_like(action[:, 0])
            else:
                dist = Normal(action_mean, action_std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)

            return action, log_prob, value

    def evaluate_actions(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of given actions.

        Used during PPO updates.

        Returns:
            log_prob: Log probability of actions
            value: State values
            entropy: Policy entropy (for exploration bonus)
        """
        if self.action_space_type == "discrete":
            action_logits, value = self.forward(state)
            action_probs = F.softmax(action_logits, dim=-1)

            dist = Categorical(action_probs)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            return log_prob, value, entropy

        else:  # continuous
            action_mean, value = self.forward(state)
            action_std = torch.exp(self.actor_log_std)

            dist = Normal(action_mean, action_std)
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)

            return log_prob, value, entropy


class PPOAgent:
    """PPO agent with clipped objective."""

    def __init__(
        self,
        state_dim: int,
        action_space_type: str = "discrete",
        num_actions: int = 175,
        action_dim: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cuda",
    ):
        """
        Args:
            state_dim: State dimension
            action_space_type: 'discrete' or 'continuous'
            num_actions: Number of discrete actions
            action_dim: Continuous action dimension
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_epsilon: PPO clip epsilon (0.1-0.3)
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Max gradient norm for clipping
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Create policy network
        self.policy = ActorCritic(
            state_dim=state_dim,
            action_space_type=action_space_type,
            num_actions=num_actions,
            action_dim=action_dim,
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr, eps=1e-5
        )

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select action from policy.

        Args:
            state: State array
            deterministic: Use deterministic policy (for eval)

        Returns:
            action: Selected action
            log_prob: Log probability
            value: State value
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, value = self.policy.get_action(
                state_tensor, deterministic
            )

        return (
            action.cpu().numpy()[0],
            log_prob.item(),
            value.item()
        )

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation.

        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value of next state (0 if terminal)

        Returns:
            advantages: GAE advantages
            returns: Discounted returns
        """
        advantages = []
        gae = 0

        values = values + [next_value]

        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
                gae = delta + self.gamma * self.gae_lambda * gae

            advantages.insert(0, gae)

        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        returns = advantages + torch.tensor(values[:-1], dtype=torch.float32).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        epochs: int = 10,
        batch_size: int = 64,
    ) -> Dict[str, float]:
        """
        PPO update with multiple epochs.

        Args:
            states: State tensor [N, state_dim]
            actions: Action tensor [N] or [N, action_dim]
            old_log_probs: Old log probabilities [N]
            advantages: Advantages [N]
            returns: Returns [N]
            epochs: Number of update epochs
            batch_size: Mini-batch size

        Returns:
            Dict with loss metrics
        """
        num_samples = states.size(0)

        # Track metrics
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        for epoch in range(epochs):
            # Random mini-batches
            indices = np.arange(num_samples)
            np.random.shuffle(indices)

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]

                # Get batch
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Evaluate actions
                log_probs, values, entropy = self.policy.evaluate_actions(
                    batch_states, batch_actions
                )

                # PPO policy loss (clipped objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE)
                value_loss = F.mse_loss(values, batch_returns)

                # Entropy bonus (encourages exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # Track metrics
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        return {
            'loss': total_loss / num_updates,
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
        }

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Saved checkpoint to {path}")

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {path}")


if __name__ == "__main__":
    # Test PPO agent
    print("Testing PPO Agent...")

    # Create agent
    agent = PPOAgent(
        state_dim=1024,
        action_space_type="discrete",
        num_actions=175,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Test action selection
    state = np.random.randn(1024)
    action, log_prob, value = agent.select_action(state)

    print(f"State shape: {state.shape}")
    print(f"Action: {action}")
    print(f"Log prob: {log_prob:.4f}")
    print(f"Value: {value:.4f}")

    # Test update
    states = torch.randn(128, 1024)
    actions = torch.randint(0, 175, (128,))
    old_log_probs = torch.randn(128)
    advantages = torch.randn(128)
    returns = torch.randn(128)

    metrics = agent.update(
        states, actions, old_log_probs, advantages, returns,
        epochs=3, batch_size=32
    )

    print(f"\nUpdate metrics: {metrics}")
    print("\nPPO Agent test passed!")
