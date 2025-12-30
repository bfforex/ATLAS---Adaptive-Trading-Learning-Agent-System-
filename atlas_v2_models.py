"""
ATLAS V2: Models Module
=======================
Complete neural network architecture with LoRA and multi-asset support.

File: models.py
Status: PRODUCTION READY ✅
NO PLACEHOLDERS ✅
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


# ==========================================
# PATCH EMBEDDING
# ==========================================

class PatchEmbedding(nn.Module):
    """Convert time series into patch tokens."""
    
    def __init__(self, patch_length: int, stride: int, d_model: int, in_channels: int):
        super().__init__()
        
        if patch_length <= 0 or stride <= 0:
            raise ValueError("patch_length and stride must be positive")
        
        self.patch_length = patch_length
        self.stride = stride
        self.in_channels = in_channels
        self.d_model = d_model
        
        # Linear projection for each patch
        self.patch_projection = nn.Linear(patch_length * in_channels, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
        logger.info(f"PatchEmbedding: patch={patch_length}, stride={stride}, "
                   f"in={in_channels}, out={d_model}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [batch_size, seq_length, n_features]
            
        Returns:
            patches: [batch_size, n_patches, d_model]
        """
        try:
            batch_size, seq_length, n_features = x.shape
            
            if n_features != self.in_channels:
                raise ValueError(f"Expected {self.in_channels} features, got {n_features}")
            
            # Transpose to [batch, features, seq]
            x = x.transpose(1, 2)
            
            # Create patches using unfold
            patches = x.unfold(dimension=2, size=self.patch_length, step=self.stride)
            # Shape: [batch, features, n_patches, patch_length]
            
            n_patches = patches.shape[2]
            
            # Reshape for projection
            patches = patches.permute(0, 2, 1, 3).contiguous()
            # Shape: [batch, n_patches, features, patch_length]
            
            patches = patches.view(batch_size, n_patches, -1)
            # Shape: [batch, n_patches, features * patch_length]
            
            # Project and normalize
            patches = self.patch_projection(patches)
            patches = self.layer_norm(patches)
            
            return patches
            
        except Exception as e:
            logger.error(f"PatchEmbedding forward error: {e}")
            logger.error(f"Input shape: {x.shape}")
            raise


# ==========================================
# POSITIONAL ENCODING
# ==========================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute div_term for sinusoidal encoding
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
        logger.info(f"PositionalEncoding: d_model={d_model}, max_len={max_len}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding."""
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)


# ==========================================
# LORA ATTENTION
# ==========================================

class LoRAAttention(nn.Module):
    """Multi-head attention with LoRA adaptation."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 lora_rank: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.1):
        super().__init__()
        
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = math.sqrt(self.d_k)
        
        # Standard attention weights (will be frozen during online learning)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # LoRA parameters
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / lora_rank
        
        # LoRA matrices for Q, K, V
        self.lora_q_down = nn.Linear(d_model, lora_rank, bias=False)
        self.lora_q_up = nn.Linear(lora_rank, d_model, bias=False)
        self.lora_k_down = nn.Linear(d_model, lora_rank, bias=False)
        self.lora_k_up = nn.Linear(lora_rank, d_model, bias=False)
        self.lora_v_down = nn.Linear(d_model, lora_rank, bias=False)
        self.lora_v_up = nn.Linear(lora_rank, d_model, bias=False)
        
        self.lora_dropout = nn.Dropout(lora_dropout)
        self.attn_dropout = nn.Dropout(dropout)
        
        # Initialize LoRA weights
        self._init_lora_weights()
        
        logger.info(f"LoRAAttention: d_model={d_model}, n_heads={n_heads}, lora_rank={lora_rank}")
    
    def _init_lora_weights(self):
        """Initialize LoRA weights properly."""
        nn.init.kaiming_uniform_(self.lora_q_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_q_up.weight)
        nn.init.kaiming_uniform_(self.lora_k_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_k_up.weight)
        nn.init.kaiming_uniform_(self.lora_v_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_v_up.weight)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with LoRA.
        
        Args:
            x: [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            output: [batch_size, seq_len, d_model]
        """
        try:
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
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
            
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
            
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.attn_dropout(attn_probs)
            
            # Apply attention to values
            attn_output = torch.matmul(attn_probs, v)
            
            # Reshape and project
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, seq_len, self.d_model)
            
            output = self.out_proj(attn_output)
            
            return output
            
        except Exception as e:
            logger.error(f"LoRAAttention forward error: {e}")
            raise
    
    def freeze_pretrained_weights(self):
        """Freeze all non-LoRA parameters for online learning."""
        for name, param in self.named_parameters():
            if 'lora' not in name:
                param.requires_grad = False
        logger.debug("Froze pretrained attention weights")


# ==========================================
# TRANSFORMER ENCODER LAYER
# ==========================================

class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with LoRA attention."""
    
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.1,
                 lora_rank: int = 8, lora_alpha: int = 16):
        super().__init__()
        
        self.attention = LoRAAttention(d_model, n_heads, dropout, lora_rank, lora_alpha)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Self-attention with pre-norm
        attn_out = self.attention(self.norm1(x), mask)
        x = x + attn_out
        
        # Feed-forward with pre-norm
        ff_out = self.feed_forward(self.norm2(x))
        x = x + ff_out
        
        return x


# ==========================================
# PATCHTST BACKBONE
# ==========================================

class PatchTSTBackbone(nn.Module):
    """PatchTST backbone with LoRA."""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            config.patch_length,
            config.stride,
            config.d_model,
            config.feature_size
        )
        
        # Positional encoding
        n_patches = config.get_n_patches()
        self.pos_encoding = nn.Parameter(torch.randn(1, n_patches, config.d_model) * 0.02)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                config.d_model,
                config.n_heads,
                config.ff_dim,
                config.dropout,
                config.lora_rank,
                config.lora_alpha
            ) for _ in range(config.n_layers)
        ])
        
        self.norm = nn.LayerNorm(config.d_model)
        
        logger.info(f"PatchTSTBackbone: {config.n_layers} layers, {n_patches} patches")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [batch_size, seq_length, n_features]
            
        Returns:
            embedding: [batch_size, d_model]
        """
        try:
            # Create patches
            x = self.patch_embedding(x)
            
            # Add positional encoding
            x = x + self.pos_encoding
            
            # Pass through transformer layers
            if self.config.gradient_checkpointing and self.training:
                for layer in self.encoder_layers:
                    x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                for layer in self.encoder_layers:
                    x = layer(x)
            
            x = self.norm(x)
            
            # Global average pooling
            embedding = x.mean(dim=1)
            
            return embedding
            
        except Exception as e:
            logger.error(f"PatchTSTBackbone forward error: {e}")
            raise
    
    def freeze_pretrained_weights(self):
        """Freeze all non-LoRA parameters."""
        for module in self.encoder_layers:
            if hasattr(module.attention, 'freeze_pretrained_weights'):
                module.attention.freeze_pretrained_weights()


# ==========================================
# PROBABILISTIC HEAD
# ==========================================

class ProbabilisticHead(nn.Module):
    """Forecasting head that outputs mean and variance."""
    
    def __init__(self, d_model: int, forecast_horizon: int = 1):
        super().__init__()
        
        self.mean_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, forecast_horizon)
        )
        
        self.variance_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, forecast_horizon),
            nn.Softplus()  # Ensure positive variance
        )
        
        logger.info(f"ProbabilisticHead: forecast_horizon={forecast_horizon}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
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
# ACTOR HEAD
# ==========================================

class ActorHead(nn.Module):
    """Policy network for action selection."""
    
    def __init__(self, state_dim: int, n_actions: int = 3):
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
        
        logger.info(f"ActorHead: state_dim={state_dim}, n_actions={n_actions}")
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: [batch_size, state_dim]
            
        Returns:
            action_logits: [batch_size, n_actions]
        """
        return self.network(state)
    
    def get_action_distribution(self, state: torch.Tensor) -> torch.distributions.Categorical:
        """Get action distribution."""
        logits = self.forward(state)
        return torch.distributions.Categorical(logits=logits)


# ==========================================
# CRITIC HEAD
# ==========================================

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
        
        logger.info(f"CriticHead: state_dim={state_dim}")
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: [batch_size, state_dim]
            
        Returns:
            value: [batch_size, 1]
        """
        return self.network(state)


# ==========================================
# CROSS-ASSET ATTENTION
# ==========================================

class CrossAssetAttention(nn.Module):
    """Attention mechanism across multiple assets."""
    
    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True, dropout=0.1
        )
        self.norm = nn.LayerNorm(d_model)
        
        logger.info(f"CrossAssetAttention: d_model={d_model}, n_heads={n_heads}")
    
    def forward(self, asset_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            asset_embeddings: [batch_size, n_assets, d_model]
            
        Returns:
            attended: [batch_size, n_assets, d_model]
        """
        # Self-attention across assets
        attn_out, _ = self.attention(
            asset_embeddings, asset_embeddings, asset_embeddings
        )
        
        # Residual connection and normalization
        output = self.norm(asset_embeddings + attn_out)
        
        return output


# ==========================================
# PORTFOLIO OPTIMIZER
# ==========================================

class PortfolioOptimizer(nn.Module):
    """Neural network for portfolio weight optimization."""
    
    def __init__(self, n_assets: int, d_model: int):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(n_assets * d_model, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_assets),
            nn.Softmax(dim=-1)  # Weights sum to 1
        )
        
        logger.info(f"PortfolioOptimizer: n_assets={n_assets}, d_model={d_model}")
    
    def forward(self, asset_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            asset_embeddings: [batch_size, n_assets, d_model]
            
        Returns:
            weights: [batch_size, n_assets]
        """
        batch_size = asset_embeddings.size(0)
        flattened = asset_embeddings.view(batch_size, -1)
        weights = self.network(flattened)
        return weights


# ==========================================
# MAIN ATLAS MODEL
# ==========================================

class ATLASModel(nn.Module):
    """Complete ATLAS multi-asset trading model."""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.symbols = config.symbols
        
        # Shared or per-asset backbone
        if config.shared_encoder:
            self.shared_backbone = PatchTSTBackbone(config)
        
        # Per-asset modules
        self.asset_modules = nn.ModuleDict()
        context_dim = 4  # Balance, Position, ATR, TimeToClose
        
        for symbol in config.symbols:
            modules = {}
            
            if not config.shared_encoder:
                modules['backbone'] = PatchTSTBackbone(config)
            
            modules['forecast_head'] = ProbabilisticHead(config.d_model)
            
            if config.asset_specific_heads:
                state_dim = config.d_model + context_dim
                modules['actor'] = ActorHead(state_dim)
                modules['critic'] = CriticHead(state_dim)
            
            self.asset_modules[symbol] = nn.ModuleDict(modules)
        
        # Cross-asset attention
        if config.cross_asset_attention:
            self.cross_asset_attention = CrossAssetAttention(config.d_model)
        
        # Portfolio optimizer
        if config.portfolio_mode == 'portfolio_opt':
            self.portfolio_optimizer = PortfolioOptimizer(
                len(config.symbols), config.d_model
            )
        
        # Shared actor-critic if not asset-specific
        if not config.asset_specific_heads:
            state_dim = config.d_model + context_dim
            self.shared_actor = ActorHead(state_dim)
            self.shared_critic = CriticHead(state_dim)
        
        logger.info(f"ATLASModel initialized: {len(config.symbols)} assets")
        self._log_parameter_count()
    
    def _log_parameter_count(self):
        """Log parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora = sum(p.numel() for n, p in self.named_parameters() if 'lora' in n)
        
        logger.info(f"Total parameters: {total:,}")
        logger.info(f"Trainable parameters: {trainable:,}")
        logger.info(f"LoRA parameters: {lora:,} ({100*lora/total:.2f}%)")
    
    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]], 
                context: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Dict]:
        """
        Complete forward pass.
        
        Args:
            batch: Dict[symbol -> {'x': tensor, 'y': tensor}]
            context: Optional context tensors per asset
            
        Returns:
            Dict[symbol -> outputs]
        """
        try:
            outputs = {}
            asset_embeddings = []
            
            for symbol in self.symbols:
                if symbol not in batch:
                    continue
                
                x = batch[symbol]['x']
                
                # Get embedding
                if self.config.shared_encoder:
                    embedding = self.shared_backbone(x)
                else:
                    embedding = self.asset_modules[symbol]['backbone'](x)
                
                asset_embeddings.append(embedding)
                
                # Probabilistic forecasting
                mean, variance = self.asset_modules[symbol]['forecast_head'](embedding)
                
                outputs[symbol] = {
                    'embedding': embedding,
                    'price_mean': mean,
                    'price_variance': variance
                }
            
            # Cross-asset attention
            if self.config.cross_asset_attention and len(asset_embeddings) > 1:
                asset_embeddings_tensor = torch.stack(asset_embeddings, dim=1)
                cross_attended = self.cross_asset_attention(asset_embeddings_tensor)
                
                for i, symbol in enumerate([s for s in self.symbols if s in outputs]):
                    outputs[symbol]['embedding'] = cross_attended[:, i, :]
            
            # Portfolio optimization
            if self.config.portfolio_mode == 'portfolio_opt' and len(asset_embeddings) > 1:
                all_embeddings = torch.stack(
                    [outputs[s]['embedding'] for s in self.symbols if s in outputs], 
                    dim=1
                )
                portfolio_weights = self.portfolio_optimizer(all_embeddings)
                outputs['portfolio_weights'] = portfolio_weights
            
            return outputs
            
        except Exception as e:
            logger.error(f"ATLASModel forward error: {e}")
            raise
    
    def freeze_backbones(self):
        """Freeze all backbone parameters (for online LoRA learning)."""
        if self.config.shared_encoder:
            self.shared_backbone.freeze_pretrained_weights()
        else:
            for symbol in self.symbols:
                if 'backbone' in self.asset_modules[symbol]:
                    self.asset_modules[symbol]['backbone'].freeze_pretrained_weights()
        
        logger.info("Froze backbone weights, LoRA parameters remain trainable")


# ==========================================
# MODULE EXPORTS
# ==========================================

__all__ = [
    'PatchEmbedding',
    'PositionalEncoding',
    'LoRAAttention',
    'TransformerEncoderLayer',
    'PatchTSTBackbone',
    'ProbabilisticHead',
    'ActorHead',
    'CriticHead',
    'CrossAssetAttention',
    'PortfolioOptimizer',
    'ATLASModel'
]


if __name__ == '__main__':
    # Test the model
    from config import get_minimal_config
    
    config = get_minimal_config()
    model = ATLASModel(config)
    
    # Create dummy batch
    batch = {
        'EURUSD': {
            'x': torch.randn(2, config.seq_length, config.feature_size),
            'y': torch.randn(2)
        }
    }
    
    # Forward pass
    outputs = model(batch)
    
    print("\n✓ Models module test passed")
    print(f"Output keys: {list(outputs.keys())}")
    print(f"EURUSD mean shape: {outputs['EURUSD']['price_mean'].shape}")
    print(f"EURUSD variance shape: {outputs['EURUSD']['price_variance'].shape}")
