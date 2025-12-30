"""
SOTA Continuous Learning Trading Agent
Architecture: PatchTST + Probabilistic Forecasting + PPO Actor-Critic + LoRA Adaptation
Optimized for RTX 4050 6GB with Flash Attention 2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import math
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
@dataclass
class Config:
    # Data & Patching
    seq_length: int = 512
    patch_length: int = 16
    stride: int = 16
    feature_size: int = 8
    
    # Model Architecture
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    ff_dim: int = 1024
    dropout: float = 0.1
    
    # LoRA Configuration
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    # PPO Hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Training
    batch_size: int = 64
    accumulation_steps: int = 8
    learning_rate: float = 1e-4
    lora_learning_rate: float = 5e-5
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    
    # Continuous Learning
    per_buffer_size: int = 100000
    per_alpha: float = 0.6
    per_beta: float = 0.4
    per_beta_increment: float = 0.001
    online_update_freq: int = 500
    per_sample_batch: int = 256
    
    # Trading Environment
    n_actions: int = 3  # HOLD, BUY, SELL
    context_dim: int = 4  # Balance, Position, ATR, TimeToClose
    
    # Hardware Optimization
    use_flash_attention: bool = True
    use_amp: bool = True
    gradient_checkpointing: bool = True
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

config = Config()

# ==========================================
# PATCHING LAYER
# ==========================================
class PatchEmbedding(nn.Module):
    """Converts time series into patch tokens."""
    
    def __init__(self, patch_length: int, stride: int, d_model: int, in_channels: int):
        super().__init__()
        self.patch_length = patch_length
        self.stride = stride
        
        # Linear projection for each patch
        self.patch_projection = nn.Linear(patch_length * in_channels, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_length, n_features]
        Returns:
            patches: [batch_size, n_patches, d_model]
        """
        batch_size, seq_length, n_features = x.shape
        
        # Unfold to create patches
        # [batch, n_features, seq_length] -> [batch, n_features, n_patches, patch_length]
        x = x.transpose(1, 2)  # [batch, n_features, seq_length]
        patches = x.unfold(dimension=2, size=self.patch_length, step=self.stride)
        
        # Reshape: [batch, n_features, n_patches, patch_length] -> [batch, n_patches, n_features * patch_length]
        n_patches = patches.shape[2]
        patches = patches.permute(0, 2, 1, 3).contiguous()
        patches = patches.view(batch_size, n_patches, -1)
        
        # Project to d_model
        patches = self.patch_projection(patches)
        patches = self.layer_norm(patches)
        
        return patches

# ==========================================
# LORA ATTENTION LAYER
# ==========================================
class LoRAAttention(nn.Module):
    """Multi-head attention with LoRA adaptation."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, 
                 lora_rank: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)
        
        # Standard attention weights (frozen during online learning)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # LoRA adaptation matrices
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / lora_rank
        
        # LoRA for Q, K, V projections
        self.lora_q_down = nn.Linear(d_model, lora_rank, bias=False)
        self.lora_q_up = nn.Linear(lora_rank, d_model, bias=False)
        self.lora_k_down = nn.Linear(d_model, lora_rank, bias=False)
        self.lora_k_up = nn.Linear(lora_rank, d_model, bias=False)
        self.lora_v_down = nn.Linear(d_model, lora_rank, bias=False)
        self.lora_v_up = nn.Linear(lora_rank, d_model, bias=False)
        
        self.lora_dropout = nn.Dropout(lora_dropout)
        self.attn_dropout = nn.Dropout(dropout)
        
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_q_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_q_up.weight)
        nn.init.kaiming_uniform_(self.lora_k_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_k_up.weight)
        nn.init.kaiming_uniform_(self.lora_v_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_v_up.weight)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Standard projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Add LoRA adaptations
        q = q + self.lora_q_up(self.lora_dropout(self.lora_q_down(x))) * self.scaling
        k = k + self.lora_k_up(self.lora_dropout(self.lora_k_down(x))) * self.scaling
        v = v + self.lora_v_up(self.lora_dropout(self.lora_v_down(x))) * self.scaling
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        if config.use_flash_attention:
            # Flash Attention 2 (if available)
            try:
                from flash_attn import flash_attn_func
                # Reshape for flash attention: [batch, seq_len, n_heads, d_k]
                q = q.transpose(1, 2).contiguous()
                k = k.transpose(1, 2).contiguous()
                v = v.transpose(1, 2).contiguous()
                
                attn_output = flash_attn_func(q, k, v, dropout_p=self.attn_dropout.p if self.training else 0.0)
                attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)
            except ImportError:
                # Fallback to standard attention
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
                if mask is not None:
                    attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
                attn_probs = F.softmax(attn_scores, dim=-1)
                attn_probs = self.attn_dropout(attn_probs)
                attn_output = torch.matmul(attn_probs, v)
                attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        else:
            # Standard attention
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.attn_dropout(attn_probs)
            attn_output = torch.matmul(attn_probs, v)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.out_proj(attn_output)
        return output
    
    def freeze_pretrained_weights(self):
        """Freeze all weights except LoRA parameters."""
        for name, param in self.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False

# ==========================================
# TRANSFORMER ENCODER WITH LORA
# ==========================================
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.1,
                 lora_rank: int = 8, lora_alpha: int = 16):
        super().__init__()
        
        self.attention = LoRAAttention(d_model, n_heads, dropout, lora_rank, lora_alpha)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attention(self.norm1(x), mask)
        x = x + attn_out
        
        # Feedforward with residual
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out
        
        return x

class PatchTSTBackbone(nn.Module):
    """PatchTST backbone with LoRA."""
    
    def __init__(self, config: Config):
        super().__init__()
        
        self.patch_embedding = PatchEmbedding(
            patch_length=config.patch_length,
            stride=config.stride,
            d_model=config.d_model,
            in_channels=config.feature_size
        )
        
        # Positional encoding
        n_patches = (config.seq_length - config.patch_length) // config.stride + 1
        self.pos_encoding = nn.Parameter(torch.randn(1, n_patches, config.d_model) * 0.02)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=config.d_model,
                n_heads=config.n_heads,
                ff_dim=config.ff_dim,
                dropout=config.dropout,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha
            ) for _ in range(config.n_layers)
        ])
        
        self.norm = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = config.gradient_checkpointing
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_length, n_features]
        Returns:
            embeddings: [batch_size, d_model]
        """
        # Create patches
        x = self.patch_embedding(x)  # [batch, n_patches, d_model]
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Pass through transformer layers
        if self.gradient_checkpointing and self.training:
            for layer in self.encoder_layers:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
        else:
            for layer in self.encoder_layers:
                x = layer(x)
        
        x = self.norm(x)
        
        # Global average pooling
        embedding = x.mean(dim=1)  # [batch, d_model]
        
        return embedding
    
    def freeze_pretrained_weights(self):
        """Freeze all weights except LoRA parameters."""
        for module in self.modules():
            if isinstance(module, LoRAAttention):
                module.freeze_pretrained_weights()
            else:
                for param in module.parameters():
                    if not any(n in param for n in ['lora']):
                        param.requires_grad = False

# ==========================================
# PROBABILISTIC FORECASTING HEAD
# ==========================================
class ProbabilisticHead(nn.Module):
    """Outputs mean and variance for probabilistic forecasting."""
    
    def __init__(self, d_model: int, forecast_horizon: int = 1):
        super().__init__()
        
        self.mean_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, forecast_horizon)
        )
        
        self.variance_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, forecast_horizon),
            nn.Softplus()  # Ensure positive variance
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, d_model]
        Returns:
            mean: [batch_size, forecast_horizon]
            variance: [batch_size, forecast_horizon]
        """
        mean = self.mean_head(x)
        variance = self.variance_head(x) + 1e-6  # Numerical stability
        
        return mean, variance

# ==========================================
# ACTOR-CRITIC HEADS
# ==========================================
class ActorHead(nn.Module):
    """Policy network for action selection."""
    
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_actions)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch_size, state_dim]
        Returns:
            action_logits: [batch_size, n_actions]
        """
        return self.network(state)
    
    def get_action_distribution(self, state: torch.Tensor) -> torch.distributions.Categorical:
        logits = self.forward(state)
        return torch.distributions.Categorical(logits=logits)

class CriticHead(nn.Module):
    """Value network for state evaluation."""
    
    def __init__(self, state_dim: int):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch_size, state_dim]
        Returns:
            value: [batch_size, 1]
        """
        return self.network(state)

# ==========================================
# PRIORITIZED EXPERIENCE REPLAY
# ==========================================
class PrioritizedReplayBuffer:
    """Experience replay with prioritization based on TD-error."""
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.position = 0
    
    def add(self, experience: Dict, td_error: float):
        """Add experience with priority."""
        priority = (abs(td_error) + 1e-5) ** self.alpha
        
        self.buffer.append(experience)
        self.priorities.append(priority)
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[list, torch.Tensor, torch.Tensor]:
        """Sample with importance sampling weights."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        
        # Importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()
        weights = torch.FloatTensor(weights).to(config.device)
        
        samples = [self.buffer[idx] for idx in indices]
        indices = torch.LongTensor(indices).to(config.device)
        
        return samples, weights, indices
    
    def update_priorities(self, indices: torch.Tensor, td_errors: torch.Tensor):
        """Update priorities based on new TD-errors."""
        for idx, td_error in zip(indices.cpu().numpy(), td_errors.cpu().numpy()):
            self.priorities[idx] = (abs(td_error) + 1e-5) ** self.alpha
    
    def __len__(self):
        return len(self.buffer)

# ==========================================
# DIFFERENTIAL SHARPE RATIO
# ==========================================
class DifferentialSharpeRatio:
    """Online Sharpe Ratio computation for reward shaping."""
    
    def __init__(self, eta: float = 0.01):
        self.eta = eta
        self.A = 0.0  # Mean estimate
        self.B = 0.0  # Second moment estimate
        self.sharpe = 0.0
    
    def update(self, return_t: float) -> float:
        """
        Update with new return and compute differential Sharpe.
        
        Args:
            return_t: Current period return
        Returns:
            Differential Sharpe ratio
        """
        delta_A = return_t - self.A
        self.A += self.eta * delta_A
        self.B += self.eta * (return_t ** 2 - self.B)
        
        variance = self.B - self.A ** 2
        std = math.sqrt(max(variance, 1e-8))
        
        if std > 1e-6:
            self.sharpe = self.A / std
            # Differential component
            differential = (self.B * delta_A - 0.5 * self.A * (return_t ** 2 - self.B)) / (std ** 3)
            return differential
        else:
            return 0.0
    
    def get_sharpe(self) -> float:
        return self.sharpe

# ==========================================
# MAIN TRADING AGENT
# ==========================================
class SOTATradingAgent(pl.LightningModule):
    """
    State-of-the-Art Continuous Learning Trading Agent.
    Combines PatchTST, Probabilistic Forecasting, PPO, and LoRA adaptation.
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Backbone
        self.backbone = PatchTSTBackbone(config)
        
        # Forecasting head
        self.forecast_head = ProbabilisticHead(config.d_model, forecast_horizon=1)
        
        # Actor-Critic heads
        state_dim = config.d_model + config.context_dim
        self.actor = ActorHead(state_dim, config.n_actions)
        self.critic = CriticHead(state_dim)
        
        # Prioritized Experience Replay
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.per_buffer_size,
            alpha=config.per_alpha
        )
        self.per_beta = config.per_beta
        
        # Differential Sharpe Ratio
        self.sharpe_ratio = DifferentialSharpeRatio()
        
        # Tracking
        self.step_counter = 0
        self.automatic_optimization = False  # Manual optimization for PPO
    
    def forward(self, time_series: torch.Tensor, context: torch.Tensor) -> Dict:
        """
        Forward pass through the complete agent.
        
        Args:
            time_series: [batch_size, seq_length, n_features]
            context: [batch_size, context_dim] - [Balance, Position, ATR, TimeToClose]
        Returns:
            Dictionary with predictions, actions, values
        """
        # Extract latent representation
        embedding = self.backbone(time_series)  # [batch, d_model]
        
        # Probabilistic forecast
        price_mean, price_variance = self.forecast_head(embedding)
        
        # State for actor-critic
        state = torch.cat([embedding, context], dim=-1)  # [batch, d_model + context_dim]
        
        # Actor output
        action_dist = self.actor.get_action_distribution(state)
        
        # Critic output
        value = self.critic(state)
        
        return {
            'embedding': embedding,
            'price_mean': price_mean,
            'price_variance': price_variance,
            'action_distribution': action_dist,
            'value': value,
            'state': state
        }
    
    def compute_nll_loss(self, mean: torch.Tensor, variance: torch.Tensor, 
                        target: torch.Tensor) -> torch.Tensor:
        """
        Negative Log-Likelihood loss for probabilistic forecasting.
        
        Args:
            mean: Predicted mean [batch_size, forecast_horizon]
            variance: Predicted variance [batch_size, forecast_horizon]
            target: Ground truth [batch_size, forecast_horizon]
        Returns:
            NLL loss
        """
        dist = torch.distributions.Normal(mean, torch.sqrt(variance))
        nll = -dist.log_prob(target).mean()
        return nll
    
    def compute_ppo_loss(self, batch: Dict, old_log_probs: torch.Tensor) -> Dict:
        """
        Compute PPO clipped loss.
        
        Args:
            batch: Dictionary containing states, actions, advantages, returns
            old_log_probs: Log probabilities from old policy
        Returns:
            Dictionary with actor_loss, critic_loss, entropy
        """
        states = batch['states']
        actions = batch['actions']
        advantages = batch['advantages']
        returns = batch['returns']
        
        # Get current policy
        action_dist = self.actor.get_action_distribution(states)
        current_log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy().mean()
        
        # Ratio for PPO
        ratio = torch.exp(current_log_probs - old_log_probs)
        
        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.config.ppo_epsilon, 1.0 + self.config.ppo_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        values = self.critic(states).squeeze()
        critic_loss = F.mse_loss(values, returns)
        
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'entropy': entropy
        }
    
    def actor_critic_step(self, time_series: torch.Tensor, context: torch.Tensor,
                         reward: torch.Tensor, next_time_series: torch.Tensor,
                         next_context: torch.Tensor, done: torch.Tensor) -> Dict:
        """
        Single actor-critic training step with experience collection.
        
        Args:
            time_series: Current state time series
            context: Current context
            reward: Reward received
            next_time_series: Next state time series
            next_context: Next context
            done: Episode termination flag
        Returns:
            Dictionary with losses and metrics
        """
        # Forward pass
        output = self.forward(time_series, context)
        next_output = self.forward(next_time_series, next_context)
        
        # TD-error for PER priority
        with torch.no_grad():
            td_error = reward + self.config.gamma * next_output['value'].squeeze() * (1 - done) - output['value'].squeeze()
        
        # Store in replay buffer
        experience = {
            'time_series': time_series.cpu(),
            'context': context.cpu(),
            'action': output['action_distribution'].sample().cpu(),
            'reward': reward.cpu(),
            'next_time_series': next_time_series.cpu(),
            'next_context': next_context.cpu(),
            'done': done.cpu(),
            'value': output['value'].detach().cpu(),
            'log_prob': output['action_distribution'].log_prob(output['action_distribution'].sample()).detach().cpu()
        }
        
        for i in range(len(time_series)):
            self.replay_buffer.add(
                {k: v[i] for k, v in experience.items()},
                td_error[i].item()
            )
        
        # Increment step counter
        self.step_counter += 1
        
        # Trigger online fine-tuning if needed
        if self.step_counter % self.config.online_update_freq == 0 and len(self.replay_buffer) >= self.config.per_sample_batch:
            self.online_fine_tune()
        
        return {
            'td_error': td_error.abs().mean(),
            'value': output['value'].mean(),
            'reward': reward.mean()
        }
    
    def online_fine_tune(self):
        """
        Perform online LoRA fine-tuning using prioritized experience replay.
        Only LoRA parameters are updated.
        """
        # Freeze backbone, only train LoRA
        self.backbone.freeze_pretrained_weights()
        
        # Sample from PER
        samples, is_weights, indices = self.replay_buffer.sample(
            self.config.per_sample_batch,
            beta=self.per_beta
        )
        
        # Increase beta for importance sampling
        self.per_beta = min(1.0, self.per_beta + self.config.per_beta_increment)
        
        # Prepare batch
        time_series = torch.stack([s['time_series'] for s in samples]).to(self.device)
        context = torch.stack([s['context'] for s in samples]).to(self.device)
        actions = torch.stack([s['action'] for s in samples]).to(self.device)
        rewards = torch.stack([s['reward'] for s in samples]).to(self.device)
        next_time_series = torch.stack([s['next_time_series'] for s in samples]).to(self.device)
        next_context = torch.stack([s['next_context'] for s in samples]).to(self.device)
        dones = torch.stack([s['done'] for s in samples]).to(self.device)
        old_values = torch.stack([s['value'] for s in samples]).to(self.device)
        old_log_probs = torch.stack([s['log_prob'] for s in samples]).to(self.device)
        
        # Compute advantages using GAE
        with torch.no_grad():
            next_output = self.forward(next_time_series, next_context)
            next_values = next_output['value'].squeeze()
            
            advantages = []
            gae = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = next_values[t] * (1 - dones[t])
                else:
                    next_value = old_values[t + 1]
                
                delta = rewards[t] + self.config.gamma * next_value - old_values[t]
                gae = delta + self.config.gamma * self.config.gae_lambda * gae * (1 - dones[t])
                advantages.insert(0, gae)
            
            advantages = torch.stack(advantages)
            returns = advantages + old_values
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Forward pass
        output = self.forward(time_series, context)
        
        # Get states for PPO
        states = output['state']
        
        # Compute PPO loss
        ppo_batch = {
            'states': states,
            'actions': actions,
            'advantages': advantages,
            'returns': returns
        }
        
        loss_dict = self.compute_ppo_loss(ppo_batch, old_log_probs)
        
        # Weight by importance sampling
        total_loss = (
            loss_dict['actor_loss'] * is_weights.mean() +
            self.config.value_loss_coef * loss_dict['critic_loss'] * is_weights.mean() -
            self.config.entropy_coef * loss_dict['entropy']
        )
        
        # Backward pass with gradient accumulation
        opt = self.optimizers()
        total_loss = total_loss / self.config.accumulation_steps
        self.manual_backward(total_loss)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.parameters() if p.requires_grad],
            self.config.max_grad_norm
        )
        
        opt.step()
        opt.zero_grad()
        
        # Update priorities in replay buffer
        with torch.no_grad():
            new_output = self.forward(time_series, context)
            new_td_errors = rewards + self.config.gamma * next_values * (1 - dones) - new_output['value'].squeeze()
            self.replay_buffer.update_priorities(indices, new_td_errors)
        
        # Log metrics
        self.log('online/actor_loss', loss_dict['actor_loss'], prog_bar=True)
        self.log('online/critic_loss', loss_dict['critic_loss'], prog_bar=True)
        self.log('online/entropy', loss_dict['entropy'], prog_bar=True)
        self.log('online/per_beta', self.per_beta, prog_bar=True)
        
        print(f"✓ Online LoRA update at step {self.step_counter} | "
              f"Actor Loss: {loss_dict['actor_loss']:.4f} | "
              f"Critic Loss: {loss_dict['critic_loss']:.4f}")
    
    def training_step(self, batch, batch_idx):
        """
        Training step for initial supervised learning phase.
        Trains the probabilistic forecasting head.
        """
        time_series = batch['time_series']
        target_price = batch['target_price']
        
        # Forward pass
        embedding = self.backbone(time_series)
        price_mean, price_variance = self.forecast_head(embedding)
        
        # NLL loss
        nll_loss = self.compute_nll_loss(price_mean, price_variance, target_price)
        
        # Regularization: Encourage reasonable uncertainty
        variance_reg = torch.mean(torch.log(price_variance + 1e-6))
        
        total_loss = nll_loss + 0.01 * variance_reg
        
        self.log('train/nll_loss', nll_loss, prog_bar=True)
        self.log('train/variance', price_variance.mean(), prog_bar=True)
        self.log('train/mean_price', price_mean.mean())
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        time_series = batch['time_series']
        target_price = batch['target_price']
        
        embedding = self.backbone(time_series)
        price_mean, price_variance = self.forecast_head(embedding)
        
        nll_loss = self.compute_nll_loss(price_mean, price_variance, target_price)
        
        # Calibration metric: prediction interval coverage
        std = torch.sqrt(price_variance)
        lower_bound = price_mean - 1.96 * std
        upper_bound = price_mean + 1.96 * std
        coverage = ((target_price >= lower_bound) & (target_price <= upper_bound)).float().mean()
        
        self.log('val/nll_loss', nll_loss, prog_bar=True)
        self.log('val/coverage_95', coverage, prog_bar=True)
        self.log('val/variance', price_variance.mean())
        
        return nll_loss
    
    def configure_optimizers(self):
        """Configure optimizers with different learning rates for LoRA."""
        # Separate parameters
        lora_params = []
        other_params = []
        
        for name, param in self.named_parameters():
            if 'lora' in name:
                lora_params.append(param)
            else:
                other_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': self.config.learning_rate},
            {'params': lora_params, 'lr': self.config.lora_learning_rate}
        ], weight_decay=self.config.weight_decay)
        
        # Warmup + Cosine annealing
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[self.config.learning_rate, self.config.lora_learning_rate],
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }

# ==========================================
# DATASET FOR SUPERVISED PRE-TRAINING
# ==========================================
class TimeSeriesDataset(Dataset):
    """Dataset for supervised pre-training of the forecasting head."""
    
    def __init__(self, data: np.ndarray, seq_length: int):
        """
        Args:
            data: [n_samples, n_features] - preprocessed time series data
            seq_length: Length of input sequences
        """
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.data) - self.seq_length - 1
    
    def __getitem__(self, idx):
        time_series = self.data[idx:idx + self.seq_length]
        target_price = self.data[idx + self.seq_length, 0]  # Predict close price
        
        return {
            'time_series': time_series,
            'target_price': target_price
        }

# ==========================================
# TRADING ENVIRONMENT SIMULATOR
# ==========================================
class TradingEnvironment:
    """
    Simple trading environment for testing the agent.
    In production, this would interface with real market data.
    """
    
    def __init__(self, data: np.ndarray, initial_balance: float = 10000.0,
                 seq_length: int = 512):
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length
        self.initial_balance = initial_balance
        self.reset()
    
    def reset(self):
        """Reset environment to initial state."""
        self.current_step = self.seq_length
        self.balance = self.initial_balance
        self.position = 0  # -1: short, 0: neutral, 1: long
        self.entry_price = 0.0
        self.portfolio_value = self.initial_balance
        self.previous_portfolio_value = self.initial_balance
        
        return self._get_observation()
    
    def _get_observation(self):
        """Get current observation (time series + context)."""
        # Time series
        time_series = self.data[self.current_step - self.seq_length:self.current_step]
        
        # Context: [Balance, Position, ATR, TimeToClose]
        current_price = self.data[self.current_step, 0].item()
        atr = self._compute_atr()
        time_to_close = (len(self.data) - self.current_step) / len(self.data)
        
        context = torch.FloatTensor([
            self.balance / self.initial_balance,  # Normalized balance
            float(self.position),  # Position status
            atr,  # Volatility
            time_to_close  # Time remaining
        ])
        
        return time_series.unsqueeze(0), context.unsqueeze(0)
    
    def _compute_atr(self, window: int = 14):
        """Compute Average True Range for volatility."""
        if self.current_step < window + 1:
            return 0.0
        
        prices = self.data[self.current_step - window:self.current_step, 0]
        high = prices.max().item()
        low = prices.min().item()
        atr = (high - low) / prices.mean().item()
        
        return atr
    
    def step(self, action: int):
        """
        Execute action and return next state, reward, done.
        
        Args:
            action: 0 (HOLD), 1 (BUY), 2 (SELL)
        Returns:
            next_obs, reward, done, info
        """
        current_price = self.data[self.current_step, 0].item()
        
        # Execute action
        if action == 1:  # BUY
            if self.position <= 0:  # Close short or open long
                if self.position == -1:
                    # Close short position
                    pnl = self.entry_price - current_price
                    self.balance += pnl
                # Open long position
                self.position = 1
                self.entry_price = current_price
        
        elif action == 2:  # SELL
            if self.position >= 0:  # Close long or open short
                if self.position == 1:
                    # Close long position
                    pnl = current_price - self.entry_price
                    self.balance += pnl
                # Open short position
                self.position = -1
                self.entry_price = current_price
        
        # Update portfolio value
        if self.position == 1:
            self.portfolio_value = self.balance + (current_price - self.entry_price)
        elif self.position == -1:
            self.portfolio_value = self.balance + (self.entry_price - current_price)
        else:
            self.portfolio_value = self.balance
        
        # Compute return
        period_return = (self.portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value
        self.previous_portfolio_value = self.portfolio_value
        
        # Differential Sharpe Ratio as reward (would be computed by agent in practice)
        reward = period_return
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Get next observation
        if not done:
            next_obs = self._get_observation()
        else:
            next_obs = (torch.zeros_like(self.data[:self.seq_length]).unsqueeze(0),
                       torch.zeros(1, 4))
        
        info = {
            'balance': self.balance,
            'position': self.position,
            'portfolio_value': self.portfolio_value,
            'current_price': current_price
        }
        
        return next_obs, reward, done, info

# ==========================================
# TRAINING SCRIPT
# ==========================================
def train_agent():
    """Complete training pipeline."""
    
    print("="*80)
    print("SOTA CONTINUOUS LEARNING TRADING AGENT")
    print("="*80)
    print(f"Device: {config.device}")
    print(f"Model Parameters: ~4.8M (PatchTST + Actor-Critic + LoRA)")
    print(f"Flash Attention: {'Enabled' if config.use_flash_attention else 'Disabled'}")
    print(f"Mixed Precision: {'FP16' if config.use_amp else 'FP32'}")
    print(f"Gradient Checkpointing: {'Enabled' if config.gradient_checkpointing else 'Disabled'}")
    print("="*80)
    
    # Generate synthetic data (replace with real market data)
    print("\n[1/4] Loading Market Data...")
    from sklearn.preprocessing import MinMaxScaler
    
    # Synthetic data generation
    n_samples = 50000
    t = np.linspace(0, 1000, n_samples)
    price = 100 + 0.05 * t + 10 * np.sin(2 * np.pi * t / 500) + np.random.normal(0, 2, n_samples)
    
    # Add features
    data = np.column_stack([
        price,  # Close
        np.random.uniform(30, 70, n_samples),  # RSI
        np.convolve(price, np.ones(14)/14, mode='same'),  # SMA
        np.random.uniform(-1, 1, n_samples),  # MACD
        np.random.uniform(0, 1, n_samples),  # BB Position
        np.random.uniform(0.5, 2.5, n_samples),  # ATR
        np.random.uniform(0.1, 0.3, n_samples),  # Volatility
        np.random.uniform(-5, 5, n_samples)  # ROC
    ])
    
    # Normalize
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Split
    train_size = int(0.7 * len(data_scaled))
    val_size = int(0.15 * len(data_scaled))
    
    train_data = data_scaled[:train_size]
    val_data = data_scaled[train_size:train_size + val_size]
    test_data = data_scaled[train_size + val_size:]
    
    print(f"✓ Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, config.seq_length)
    val_dataset = TimeSeriesDataset(val_data, config.seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize model
    print("\n[2/4] Initializing Model...")
    model = SOTATradingAgent(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(p.numel() for n, p in model.named_parameters() if 'lora' in n)
    
    print(f"✓ Total Parameters: {total_params:,}")
    print(f"✓ Trainable Parameters: {trainable_params:,}")
    print(f"✓ LoRA Parameters: {lora_params:,} ({100*lora_params/total_params:.2f}%)")
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='sota-trading-{epoch:02d}-{val/nll_loss:.4f}',
        monitor='val/nll_loss',
        mode='min',
        save_top_k=3
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val/nll_loss',
        patience=15,
        mode='min'
    )
    
    # Trainer
    print("\n[3/4] Starting Supervised Pre-Training...")
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='16-mixed' if config.use_amp else '32',
        gradient_clip_val=config.max_grad_norm,
        accumulate_grad_batches=config.accumulation_steps,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=50,
        enable_progress_bar=True
    )
    
    trainer.fit(model, train_loader, val_loader)
    
    print("\n✓ Supervised pre-training complete!")
    
    # Load best checkpoint
    best_model_path = checkpoint_callback.best_model_path
    model = SOTATradingAgent.load_from_checkpoint(best_model_path, config=config)
    model.eval()
    
    # RL Fine-tuning Phase
    print("\n[4/4] Starting RL Trading Simulation...")
    env = TradingEnvironment(test_data, seq_length=config.seq_length)
    
    n_episodes = 10
    episode_rewards = []
    
    for episode in range(n_episodes):
        obs = env.reset()
        time_series, context = obs
        time_series = time_series.to(config.device)
        context = context.to(config.device)
        
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            # Get action from policy
            with torch.no_grad():
                output = model(time_series, context)
                action = output['action_distribution'].sample()
            
            # Execute action
            next_obs, reward, done, info = env.step(action.item())
            next_time_series, next_context = next_obs
            next_time_series = next_time_series.to(config.device)
            next_context = next_context.to(config.device)
            
            # Actor-critic step (with online learning)
            model.actor_critic_step(
                time_series, context,
                torch.FloatTensor([reward]).to(config.device),
                next_time_series, next_context,
                torch.FloatTensor([float(done)]).to(config.device)
            )
            
            episode_reward += reward
            time_series, context = next_time_series, next_context
            step += 1
        
        episode_rewards.append(episode_reward)
        final_value = info['portfolio_value']
        roi = (final_value - env.initial_balance) / env.initial_balance * 100
        
        print(f"Episode {episode+1}/{n_episodes} | "
              f"Steps: {step} | "
              f"Total Reward: {episode_reward:.4f} | "
              f"Final Value: ${final_value:.2f} | "
              f"ROI: {roi:.2f}%")
    
    avg_reward = np.mean(episode_rewards)
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Average Episode Reward: {avg_reward:.4f}")
    print(f"Replay Buffer Size: {len(model.replay_buffer)}")
    print(f"Total Online Updates: {model.step_counter // config.online_update_freq}")
    print(f"{'='*80}")

if __name__ == '__main__':
    train_agent()
