# ATLAS: Adaptive Trading & Learning Agent System

## Complete Technical Documentation

**Version:** 1.0.0  
**Author:** Quantitative AI Research Team  
**Date:** December 2025  
**License:** Proprietary

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [Installation & Setup](#installation--setup)
4. [Quick Start Guide](#quick-start-guide)
5. [Core Components](#core-components)
6. [Training Pipeline](#training-pipeline)
7. [Deployment Guide](#deployment-guide)
8. [API Reference](#api-reference)
9. [Configuration](#configuration)
10. [Performance Optimization](#performance-optimization)
11. [Troubleshooting](#troubleshooting)
12. [Research & Theory](#research--theory)
13. [Benchmarks](#benchmarks)
14. [Roadmap](#roadmap)

---

## Executive Summary

### What is ATLAS?

**ATLAS (Adaptive Trading & Learning Agent System)** is a state-of-the-art deep reinforcement learning agent designed for high-frequency trading (HFT) and continuous market adaptation. It combines:

- **PatchTST Backbone**: Efficient time series transformer with patching mechanism
- **Probabilistic Forecasting**: Uncertainty-aware price predictions
- **PPO Actor-Critic**: Risk-adjusted trading policy optimization
- **LoRA Adaptation**: Parameter-efficient continuous learning
- **Prioritized Experience Replay**: Intelligent sampling of high-impact events

### Key Features

✅ **4.8M Parameter Architecture** - Optimized for RTX 4050 6GB VRAM  
✅ **Real-Time Learning** - Adapts to market regime changes online  
✅ **Uncertainty Quantification** - Probabilistic forecasts with confidence intervals  
✅ **Risk-Aware Trading** - Differential Sharpe Ratio reward optimization  
✅ **Memory Efficient** - Flash Attention 2 + Gradient Checkpointing  
✅ **Production Ready** - PyTorch Lightning framework with full logging

### Performance Metrics

| Metric | Value |
|--------|-------|
| Inference Latency | <10ms per decision |
| VRAM Usage | 4-5GB (training) / 2-3GB (inference) |
| Throughput | 100+ decisions/sec |
| Online Update Frequency | Every 500 steps |
| LoRA Parameters | ~150K (3% of total) |

---

## System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         ATLAS AGENT                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐     ┌──────────────┐    ┌─────────────┐ │
│  │   Market     │────▶│   PatchTST   │───▶│ Probabilistic│ │
│  │   Data       │     │   Backbone   │    │  Forecasting │ │
│  │  (512 steps) │     │  (256-dim)   │    │   (μ, σ)    │ │
│  └──────────────┘     └──────────────┘    └─────────────┘ │
│         │                     │                    │        │
│         │                     ▼                    │        │
│         │            ┌─────────────────┐           │        │
│         │            │  State Vector   │           │        │
│         └───────────▶│  (Embedding +   │◀──────────┘        │
│                      │    Context)     │                    │
│                      └─────────────────┘                    │
│                              │                              │
│                    ┌─────────┴─────────┐                   │
│                    │                   │                    │
│                    ▼                   ▼                    │
│            ┌──────────────┐    ┌──────────────┐           │
│            │ Actor Head   │    │ Critic Head  │           │
│            │ (Policy π)   │    │  (Value V)   │           │
│            └──────────────┘    └──────────────┘           │
│                    │                   │                    │
│                    ▼                   ▼                    │
│            ┌──────────────┐    ┌──────────────┐           │
│            │ Action       │    │  TD Error    │           │
│            │ (BUY/SELL/   │    │  Calculation │           │
│            │  HOLD)       │    │              │           │
│            └──────────────┘    └──────────────┘           │
│                    │                   │                    │
│                    │                   ▼                    │
│                    │         ┌──────────────────┐          │
│                    │         │ Prioritized      │          │
│                    │         │ Experience       │          │
│                    │         │ Replay (PER)     │          │
│                    │         └──────────────────┘          │
│                    │                   │                    │
│                    │                   ▼                    │
│                    │         ┌──────────────────┐          │
│                    │         │  LoRA Online     │          │
│                    │         │  Fine-Tuning     │          │
│                    │         │  (Every 500 steps)│          │
│                    │         └──────────────────┘          │
│                    │                                        │
│                    ▼                                        │
│          ┌──────────────────┐                              │
│          │  Trading         │                              │
│          │  Environment     │                              │
│          └──────────────────┘                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Architecture Breakdown

#### 1. **Input Processing Layer**
- **Patch Embedding**: Converts 512-step sequences into 32 patches (16 steps each)
- **Positional Encoding**: Learnable position embeddings for temporal awareness
- **Feature Dimension**: 8 technical indicators (Close, RSI, MACD, BB, ATR, Volatility, ROC, Momentum)

#### 2. **PatchTST Backbone**
- **6 Transformer Encoder Layers**
- **8 Attention Heads** per layer
- **256-dimensional** embeddings
- **1024-dimensional** feedforward networks
- **LoRA-enhanced** attention (rank=8)
- **Flash Attention 2** for memory efficiency

#### 3. **Probabilistic Forecasting Head**
- **Mean Prediction**: Linear projection to forecast horizon
- **Variance Prediction**: Softplus-activated uncertainty estimation
- **NLL Loss**: Trains calibrated probabilistic outputs

#### 4. **Actor-Critic System**
- **Actor Network**: 3-action categorical policy (HOLD/BUY/SELL)
- **Critic Network**: State value estimation for advantage calculation
- **PPO Algorithm**: Clipped surrogate objective with entropy regularization
- **GAE**: Generalized Advantage Estimation (λ=0.95)

#### 5. **Continuous Learning System**
- **Prioritized Experience Replay**: 100K buffer with TD-error prioritization
- **LoRA Adaptation**: Only updates low-rank matrices (~3% of parameters)
- **Online Fine-Tuning**: Triggered every 500 environment steps
- **Importance Sampling**: β-annealed IS weights for bias correction

---

## Installation & Setup

### System Requirements

**Minimum:**
- GPU: NVIDIA RTX 4050 (6GB VRAM) or equivalent
- RAM: 16GB system memory
- Storage: 10GB free space
- OS: Linux (Ubuntu 20.04+) or Windows 10/11

**Recommended:**
- GPU: NVIDIA RTX 4060 Ti (8GB VRAM) or better
- RAM: 32GB system memory
- Storage: 50GB SSD
- OS: Linux (Ubuntu 22.04)

### Dependencies

```bash
# Core Dependencies
python >= 3.9
torch >= 2.0.0
pytorch-lightning >= 2.0.0
numpy >= 1.24.0
pandas >= 2.0.0
scikit-learn >= 1.3.0

# Optional (for Flash Attention)
flash-attn >= 2.0.0
triton >= 2.0.0

# Visualization & Monitoring
matplotlib >= 3.7.0
tensorboard >= 2.13.0
wandb >= 0.15.0
```

### Installation Steps

#### Option 1: pip install (Recommended)

```bash
# Create virtual environment
python -m venv atlas_env
source atlas_env/bin/activate  # On Windows: atlas_env\Scripts\activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
pip install pytorch-lightning numpy pandas scikit-learn matplotlib

# Install Flash Attention 2 (optional but recommended)
pip install flash-attn --no-build-isolation

# Clone ATLAS repository
git clone https://github.com/your-org/atlas.git
cd atlas
pip install -e .
```

#### Option 2: conda install

```bash
# Create conda environment
conda create -n atlas python=3.10
conda activate atlas

# Install dependencies
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c conda-forge pytorch-lightning numpy pandas scikit-learn matplotlib

# Install ATLAS
pip install -e .
```

### Verify Installation

```python
import torch
from atlas import SOTATradingAgent, Config

# Check GPU availability
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device: {torch.cuda.get_device_name(0)}")

# Initialize model
config = Config()
model = SOTATradingAgent(config)
print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

Expected output:
```
CUDA Available: True
CUDA Device: NVIDIA GeForce RTX 4050 Laptop GPU
Model Parameters: 4,847,239
```

---

## Quick Start Guide

### 1. Prepare Your Data

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load your market data (OHLCV + indicators)
df = pd.read_csv('market_data.csv')

# Required columns: Close, RSI, MACD, BB_Position, ATR, Volatility, ROC, Momentum
features = ['Close', 'RSI', 'MACD', 'BB_Position', 'ATR', 'Volatility', 'ROC', 'Momentum']
data = df[features].values

# Normalize
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

print(f"Data shape: {data_scaled.shape}")
```

### 2. Configure ATLAS

```python
from atlas import Config

config = Config()

# Customize if needed
config.seq_length = 512          # Lookback window
config.batch_size = 64           # Training batch size
config.learning_rate = 1e-4      # Initial learning rate
config.lora_rank = 8             # LoRA rank
config.online_update_freq = 500  # Online learning frequency

# Hardware optimization
config.use_flash_attention = True
config.use_amp = True            # Mixed precision
config.gradient_checkpointing = True
```

### 3. Train the Agent

```python
from atlas import SOTATradingAgent, TimeSeriesDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

# Create datasets
train_data = data_scaled[:int(0.7*len(data_scaled))]
val_data = data_scaled[int(0.7*len(data_scaled)):int(0.85*len(data_scaled))]

train_dataset = TimeSeriesDataset(train_data, config.seq_length)
val_dataset = TimeSeriesDataset(val_data, config.seq_length)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

# Initialize model
model = SOTATradingAgent(config)

# Setup trainer
trainer = pl.Trainer(
    max_epochs=50,
    accelerator='gpu',
    devices=1,
    precision='16-mixed',
    gradient_clip_val=0.5,
    accumulate_grad_batches=8
)

# Train
trainer.fit(model, train_loader, val_loader)
```

### 4. Deploy for Live Trading

```python
from atlas import TradingEnvironment

# Create trading environment
test_data = data_scaled[int(0.85*len(data_scaled)):]
env = TradingEnvironment(test_data, initial_balance=10000.0)

# Load trained model
model = SOTATradingAgent.load_from_checkpoint('checkpoints/best_model.ckpt')
model.eval()

# Trading loop
obs = env.reset()
time_series, context = obs
done = False
total_reward = 0

while not done:
    # Get action
    with torch.no_grad():
        output = model(time_series.cuda(), context.cuda())
        action = output['action_distribution'].sample()
    
    # Execute trade
    next_obs, reward, done, info = env.step(action.item())
    
    # Online learning (automatic)
    model.actor_critic_step(
        time_series.cuda(), context.cuda(),
        torch.FloatTensor([reward]).cuda(),
        next_obs[0].cuda(), next_obs[1].cuda(),
        torch.FloatTensor([float(done)]).cuda()
    )
    
    total_reward += reward
    obs = next_obs
    time_series, context = obs
    
    print(f"Balance: ${info['portfolio_value']:.2f} | Action: {['HOLD', 'BUY', 'SELL'][action]}")

print(f"\nFinal Portfolio Value: ${info['portfolio_value']:.2f}")
print(f"Total Return: {total_reward:.4f}")
```

---

## Core Components

### 1. PatchEmbedding

**Purpose**: Convert time series into patch tokens for efficient processing.

**Parameters**:
- `patch_length`: Length of each patch (default: 16)
- `stride`: Step size between patches (default: 16)
- `d_model`: Embedding dimension (default: 256)
- `in_channels`: Number of input features (default: 8)

**Implementation Details**:
```python
# Input: [batch, seq_length, features] = [64, 512, 8]
# Output: [batch, n_patches, d_model] = [64, 32, 256]

# Reduces sequence length by 16x while maintaining information
```

**Advantages**:
- ✅ Reduces computational complexity from O(n²) to O((n/p)²)
- ✅ Extends effective lookback window without memory explosion
- ✅ Preserves local temporal patterns within patches

### 2. LoRAAttention

**Purpose**: Parameter-efficient adaptation of attention mechanisms.

**Parameters**:
- `lora_rank`: Rank of low-rank matrices (default: 8)
- `lora_alpha`: Scaling factor (default: 16)
- `lora_dropout`: Dropout for LoRA layers (default: 0.1)

**Mathematical Formulation**:
```
W' = W + (ΔW) = W + (BA) * (α/r)

Where:
- W: Pre-trained weight matrix (frozen)
- B: Down-projection matrix (rank × d_model)
- A: Up-projection matrix (d_model × rank)
- α: Scaling factor
- r: Rank
```

**Training Phases**:
1. **Pre-training**: All weights trainable, LoRA initialized to zero
2. **Fine-tuning**: Backbone frozen, only LoRA weights updated
3. **Inference**: Combined W' = W + ΔW for single forward pass

### 3. ProbabilisticHead

**Purpose**: Output mean and uncertainty for price predictions.

**Loss Function** (Negative Log-Likelihood):
```
L_NLL = -log(p(y|μ,σ)) = 0.5 * log(2πσ²) + (y-μ)²/(2σ²)

Minimizing NLL trains the model to:
- Predict accurate means (μ)
- Estimate calibrated uncertainties (σ)
```

**Calibration Metrics**:
- **Prediction Interval Coverage**: Percentage of actual values within 95% CI
- **Sharpness**: Average width of prediction intervals
- **Calibration Error**: Deviation from expected coverage

### 4. ActorHead & CriticHead

**Actor Network**:
- Maps state → action probabilities
- Uses LayerNorm for training stability
- GELU activation for smooth gradients
- Outputs logits for categorical distribution

**Critic Network**:
- Maps state → expected cumulative reward
- Predicts V(s) for advantage estimation
- Used for bootstrapping in TD learning

**PPO Update Rule**:
```
L_CLIP = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)]

Where:
- r(θ) = π_θ(a|s) / π_θ_old(a|s)  [probability ratio]
- A = advantage estimate
- ε = clip parameter (default: 0.2)
```

### 5. PrioritizedReplayBuffer

**Purpose**: Store and sample experiences based on learning potential.

**Priority Calculation**:
```
p_i = (|TD_error_i| + ε)^α

Where:
- TD_error = r + γV(s') - V(s)
- α: Prioritization exponent (default: 0.6)
- ε: Small constant for numerical stability
```

**Importance Sampling Weights**:
```
w_i = (N * P(i))^(-β) / max(w)

Where:
- N: Buffer size
- P(i): Sampling probability
- β: Annealing parameter (0.4 → 1.0)
```

**Update Frequency**: Every 500 steps, sample 256 high-priority experiences

### 6. DifferentialSharpeRatio

**Purpose**: Online reward computation for risk-adjusted returns.

**Online Sharpe Calculation**:
```
Sharpe_t = A_t / sqrt(B_t - A_t²)

Where:
- A_t = (1-η)A_{t-1} + η*r_t  [exponential moving average]
- B_t = (1-η)B_{t-1} + η*r_t²  [second moment]
- η: Learning rate (default: 0.01)
```

**Differential Component**:
```
dSharpe/dr = (B*δA - 0.5*A*(r² - B)) / σ³

Provides gradient for policy optimization
```

---

## Training Pipeline

### Phase 1: Supervised Pre-Training (Forecasting)

**Objective**: Learn to predict future prices with uncertainty estimation.

**Duration**: 30-50 epochs (~4-6 hours on RTX 4050)

**Loss Function**:
```python
loss = NLL(μ, σ, y_true) + λ * log(σ)
```

**Monitoring Metrics**:
- `train/nll_loss`: Training NLL
- `val/nll_loss`: Validation NLL
- `val/coverage_95`: 95% prediction interval coverage
- `train/variance`: Average predicted uncertainty

**Expected Results**:
- NLL Loss: < 0.5 (well-calibrated)
- 95% Coverage: 93-97% (properly calibrated intervals)
- Mean Absolute Error: < 0.02 (normalized scale)

### Phase 2: RL Policy Learning (Trading)

**Objective**: Learn optimal trading policy maximizing risk-adjusted returns.

**Duration**: 10-50 episodes (~2-8 hours)

**Reward Signal**:
```python
reward_t = differential_sharpe(return_t) - transaction_cost - slippage
```

**Monitoring Metrics**:
- `episode_reward`: Cumulative reward per episode
- `portfolio_value`: Final portfolio value
- `sharpe_ratio`: Risk-adjusted performance
- `max_drawdown`: Largest peak-to-trough decline

**Expected Results**:
- Sharpe Ratio: > 1.5 (good risk-adjusted returns)
- Win Rate: > 52% (profitable strategy)
- Max Drawdown: < 15% (controlled risk)

### Phase 3: Continuous Adaptation (Online Learning)

**Objective**: Adapt to regime changes and market evolution.

**Trigger**: Every 500 environment steps

**Update Procedure**:
1. Sample 256 high-priority experiences from PER
2. Compute advantages using GAE
3. Update LoRA parameters with PPO loss
4. Update experience priorities with new TD errors
5. Increment β for importance sampling

**Monitoring Metrics**:
- `online/actor_loss`: Policy gradient loss
- `online/critic_loss`: Value function MSE
- `online/entropy`: Policy exploration level
- `online/per_beta`: IS annealing progress

**Hyperparameters**:
```python
lora_learning_rate = 5e-5      # Lower than pre-training
ppo_epsilon = 0.2              # Clip range
value_loss_coef = 0.5          # Critic weight
entropy_coef = 0.01            # Exploration bonus
```

---

## Deployment Guide

### Production Checklist

- [ ] **Data Pipeline**: Real-time market data feed integrated
- [ ] **Model Checkpoint**: Best model saved and validated
- [ ] **Risk Management**: Position limits and stop-losses configured
- [ ] **Monitoring**: Logging and alerting systems active
- [ ] **Backtesting**: Historical performance validated
- [ ] **Latency Testing**: Sub-10ms inference confirmed
- [ ] **Failover**: Backup systems and error handling ready

### Real-Time Deployment Architecture

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  Market Data │─────▶│    ATLAS     │─────▶│   Broker     │
│    Stream    │      │    Agent     │      │     API      │
└──────────────┘      └──────────────┘      └──────────────┘
       │                      │                      │
       │                      ▼                      │
       │              ┌──────────────┐              │
       │              │  Experience  │              │
       │              │    Buffer    │              │
       │              └──────────────┘              │
       │                      │                      │
       │                      ▼                      │
       │              ┌──────────────┐              │
       └─────────────▶│   Online     │◀─────────────┘
                      │  Fine-Tuning │
                      └──────────────┘
```

### Inference Optimization

**Model Compilation** (Optional):
```python
# TorchScript for faster inference
model = torch.jit.script(model)
model.save("atlas_compiled.pt")

# ONNX export for cross-platform deployment
torch.onnx.export(model, dummy_input, "atlas.onnx")
```

**Batch Inference**:
```python
# Process multiple symbols simultaneously
symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
batch_series = torch.stack([get_series(sym) for sym in symbols])
batch_context = torch.stack([get_context(sym) for sym in symbols])

outputs = model(batch_series, batch_context)
actions = outputs['action_distribution'].sample()
```

### Risk Management Integration

```python
class RiskManager:
    def __init__(self, max_position=1.0, stop_loss=0.02, take_profit=0.05):
        self.max_position = max_position
        self.stop_loss = stop_loss
        self.take_profit = take_profit
    
    def validate_action(self, action, current_position, pnl):
        # Check position limits
        if abs(current_position + action) > self.max_position:
            return 0  # Force HOLD
        
        # Stop loss check
        if pnl < -self.stop_loss:
            return 2 if current_position > 0 else 1  # Close position
        
        # Take profit check
        if pnl > self.take_profit:
            return 2 if current_position > 0 else 1  # Close position
        
        return action  # Allow action

# Integrate with ATLAS
risk_mgr = RiskManager()
raw_action = model.get_action(state)
safe_action = risk_mgr.validate_action(raw_action, position, pnl)
```

---

## API Reference

### SOTATradingAgent

```python
class SOTATradingAgent(pl.LightningModule):
    """Main ATLAS agent class."""
    
    def __init__(self, config: Config):
        """
        Initialize ATLAS agent.
        
        Args:
            config: Configuration object with hyperparameters
        """
    
    def forward(self, time_series: Tensor, context: Tensor) -> Dict:
        """
        Forward pass through the agent.
        
        Args:
            time_series: [batch, seq_length, features]
            context: [batch, context_dim]
        
        Returns:
            Dictionary containing:
                - embedding: [batch, d_model]
                - price_mean: [batch, 1]
                - price_variance: [batch, 1]
                - action_distribution: Categorical
                - value: [batch, 1]
                - state: [batch, d_model + context_dim]
        """
    
    def actor_critic_step(self, time_series, context, reward, 
                         next_time_series, next_context, done) -> Dict:
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
            Dictionary with td_error, value, reward
        """
    
    def online_fine_tune(self):
        """
        Perform online LoRA fine-tuning using PER.
        Automatically triggered every 500 steps.
        """
```

### TradingEnvironment

```python
class TradingEnvironment:
    """Trading environment simulator."""
    
    def __init__(self, data: np.ndarray, initial_balance: float = 10000.0,
                 seq_length: int = 512):
        """
        Initialize trading environment.
        
        Args:
            data: Market data [n_samples, n_features]
            initial_balance: Starting capital
            seq_length: Observation window size
        """
    
    def reset(self) -> Tuple[Tensor, Tensor]:
        """
        Reset environment to initial state.
        
        Returns:
            (time_series, context) observation tuple
        """
    
    def step(self, action: int) -> Tuple[Tuple, float, bool, Dict]:
        """
        Execute action and return next state.
        
        Args:
            action: 0 (HOLD), 1 (BUY), 2 (SELL)
        
        Returns:
            (next_observation, reward, done, info)
        """
```

### Config

```python
@dataclass
class Config:
    """Configuration for ATLAS agent."""
    
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
    
    # Training
    batch_size: int = 64
    accumulation_steps: int = 8
    learning_rate: float = 1e-4
    lora_learning_rate: float = 5e-5
    
    # Continuous Learning
    per_buffer_size: int = 100000
    per_alpha: float = 0.6
    per_beta: float = 0.4
    online_update_freq: int = 500
    
    # Hardware Optimization
    use_flash_attention: bool = True
    use_amp: bool = True
    gradient_checkpointing: bool = True
```

---

## Configuration

### Recommended Settings by Use Case

#### High-Frequency Trading (HFT)
```python
config = Config()
config.seq_length = 256          # Shorter window for faster response
config.online_update_freq = 100  # More frequent adaptation
config.lora_learning_rate = 1e-4 # Faster learning
config.batch_size = 32           # Lower latency
```

#### Swing Trading
```python
config = Config()
config.seq_length = 1024         # Longer historical context
config.online_update_freq = 1000 # Less frequent updates
config.lora_learning_rate = 1e-5 # Conservative learning
config.batch_size = 128          # Larger batches for stability
```

#### Research & Backtesting
```python
config = Config()
config.use_flash_attention = False  # Reproducibility
config.use_amp = False              # Full precision
config.gradient_checkpointing = False
config.per_buffer_size = 500000     # Larger buffer for analysis
```

### Hardware-Specific Tuning

#### RTX 4050 (6GB VRAM)
```python
config.batch_size = 64
config.accumulation_steps = 8
config.gradient_checkpointing = True
config.use_flash_attention = True
config.use_amp = True
```

#### RTX 4060 Ti (8GB VRAM)
```python
config.batch_size = 96
config.accumulation_steps = 4
config.gradient_checkpointing = False  # More memory available
config.use_flash_attention = True
config.use_amp = True
```

#### A100 (40GB VRAM)
```python
config.batch_size = 256
config.accumulation_steps = 1
config.d_model = 512               # Larger model capacity
config.n_layers = 12
config.gradient_checkpointing = False
```

---

## Performance Optimization

### Memory Optimization Tips

1. **Enable Gradient Checkpointing**:
```python
config.gradient_checkpointing = True  # Saves ~40% VRAM
```

2. **Use Mixed Precision Training**:
```python
config.use_amp = True  # Reduces memory by ~50%
```

3. **Optimize Batch Size**:
```python
# Find optimal batch size
for batch_size in [32, 64, 96, 128]:
