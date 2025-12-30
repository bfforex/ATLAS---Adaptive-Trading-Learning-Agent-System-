"""
ATLAS V2: Configuration Module
==============================
Complete configuration with validation and persistence.

File: config.py
Status: PRODUCTION READY ✅
"""

import torch
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ATLASConfig:
    """Complete ATLAS V2 configuration with validation."""
    
    # ==========================================
    # ASSET CONFIGURATION
    # ==========================================
    symbols: List[str] = field(default_factory=lambda: ['EURUSD', 'GBPUSD', 'BTCUSD'])
    
    # ==========================================
    # DATA PARAMETERS
    # ==========================================
    seq_length: int = 512
    patch_length: int = 16
    stride: int = 16
    feature_size: int = 8
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # ==========================================
    # MODEL ARCHITECTURE
    # ==========================================
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    ff_dim: int = 1024
    dropout: float = 0.2
    
    # ==========================================
    # LORA PARAMETERS
    # ==========================================
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    # ==========================================
    # MULTI-ASSET SETTINGS
    # ==========================================
    shared_encoder: bool = True
    asset_specific_heads: bool = True
    cross_asset_attention: bool = True
    portfolio_mode: str = 'correlated'  # 'independent', 'correlated', 'portfolio_opt'
    
    # ==========================================
    # TRAINING PARAMETERS
    # ==========================================
    batch_size: int = 64
    accumulation_steps: int = 8
    learning_rate: float = 1e-4
    lora_learning_rate: float = 5e-5
    weight_decay: float = 1e-5
    num_epochs: int = 100
    patience: int = 25
    min_epochs: int = 10
    gradient_clip_val: float = 0.5
    
    # ==========================================
    # PERSISTENT LEARNING
    # ==========================================
    memory_dir: str = 'memory'
    auto_resume: bool = True
    save_every_n_epochs: int = 5
    keep_n_checkpoints: int = 3
    
    # ==========================================
    # PPO PARAMETERS
    # ==========================================
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # ==========================================
    # EXPERIENCE REPLAY
    # ==========================================
    per_buffer_size: int = 100000
    per_alpha: float = 0.6
    per_beta: float = 0.4
    per_beta_increment: float = 0.001
    online_update_freq: int = 500
    per_sample_batch: int = 256
    
    # ==========================================
    # AUTOMATED IMPROVEMENT
    # ==========================================
    auto_optimize: bool = True
    performance_threshold: float = 1.5
    auto_adjust_hyperparams: bool = True
    early_stop_on_degradation: bool = True
    degradation_tolerance: int = 3
    min_improvement: float = 0.01
    
    # ==========================================
    # MARKET REGIMES
    # ==========================================
    enable_regime_detection: bool = True
    regime_detection_window: int = 50
    regimes: List[str] = field(default_factory=lambda: [
        'trending_bull', 'trending_bear', 'ranging_neutral', 
        'volatile_mixed', 'crisis'
    ])
    
    # ==========================================
    # HARDWARE OPTIMIZATION
    # ==========================================
    use_amp: bool = True
    gradient_checkpointing: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # ==========================================
    # LOGGING & MONITORING
    # ==========================================
    log_every_n_steps: int = 50
    val_check_interval: float = 1.0
    enable_progress_bar: bool = True
    enable_model_summary: bool = True
    
    # ==========================================
    # RISK MANAGEMENT
    # ==========================================
    max_position_size: float = 1.0
    max_leverage: float = 3.0
    stop_loss: float = 0.02
    take_profit: float = 0.05
    max_correlation_exposure: float = 0.7
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_parameters()
        self._create_directories()
        self._log_configuration()
    
    def _validate_parameters(self):
        """Comprehensive parameter validation."""
        errors = []
        
        # Validate positive integers
        positive_ints = [
            'seq_length', 'patch_length', 'stride', 'feature_size',
            'batch_size', 'accumulation_steps', 'num_epochs', 'min_epochs',
            'd_model', 'n_heads', 'n_layers', 'ff_dim',
            'lora_rank', 'lora_alpha', 'per_buffer_size', 'online_update_freq'
        ]
        
        for param in positive_ints:
            value = getattr(self, param)
            if not isinstance(value, int) or value <= 0:
                errors.append(f"{param} must be a positive integer, got {value}")
        
        # Validate probability ranges [0, 1]
        probabilities = ['dropout', 'lora_dropout', 'gamma', 'gae_lambda']
        for param in probabilities:
            value = getattr(self, param)
            if not 0 <= value <= 1:
                errors.append(f"{param} must be in [0, 1], got {value}")
        
        # Validate learning rates (0, 1)
        for param in ['learning_rate', 'lora_learning_rate']:
            value = getattr(self, param)
            if not 0 < value < 1:
                errors.append(f"{param} must be in (0, 1), got {value}")
        
        # Validate splits sum to 1.0
        split_sum = self.train_split + self.val_split + self.test_split
        if not 0.99 <= split_sum <= 1.01:
            errors.append(f"Data splits must sum to 1.0, got {split_sum}")
        
        # Validate model constraints
        if self.d_model % self.n_heads != 0:
            errors.append(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        
        if self.patch_length > self.seq_length:
            errors.append(f"patch_length ({self.patch_length}) cannot exceed seq_length ({self.seq_length})")
        
        # Validate portfolio mode
        valid_modes = ['independent', 'correlated', 'portfolio_opt']
        if self.portfolio_mode not in valid_modes:
            errors.append(f"portfolio_mode must be one of {valid_modes}, got {self.portfolio_mode}")
        
        # Validate symbols
        if not self.symbols or len(self.symbols) == 0:
            errors.append("At least one symbol must be specified")
        
        # Raise all errors at once
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)
        
        logger.info("✓ Configuration validation passed")
    
    def _create_directories(self):
        """Create all required directories."""
        directories = [
            self.memory_dir,
            f"{self.memory_dir}/checkpoints",
            f"{self.memory_dir}/checkpoints/regime_specific",
            f"{self.memory_dir}/checkpoints/best_per_asset",
            f"{self.memory_dir}/reports",
            f"{self.memory_dir}/logs",
            f"{self.memory_dir}/visualizations",
            f"{self.memory_dir}/assets",
            f"{self.memory_dir}/portfolio",
            f"{self.memory_dir}/experiences",
            'data/raw',
            'data/processed'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✓ Created directory structure in {self.memory_dir}")
    
    def _log_configuration(self):
        """Log configuration summary."""
        logger.info("="*60)
        logger.info("ATLAS V2 CONFIGURATION")
        logger.info("="*60)
        logger.info(f"Assets: {', '.join(self.symbols)}")
        logger.info(f"Model: {self.d_model}d, {self.n_layers} layers, {self.n_heads} heads")
        logger.info(f"Training: {self.num_epochs} epochs, batch_size={self.batch_size}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Memory: {self.memory_dir}")
        logger.info(f"Auto-resume: {self.auto_resume}")
        logger.info(f"Auto-optimize: {self.auto_optimize}")
        logger.info("="*60)
    
    def save(self, path: str = None):
        """
        Save configuration to JSON file.
        
        Args:
            path: File path (default: memory_dir/config.json)
        """
        if path is None:
            path = f"{self.memory_dir}/config.json"
        
        config_dict = asdict(self)
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"✓ Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ATLASConfig':
        """
        Load configuration from JSON file.
        
        Args:
            path: File path to config.json
            
        Returns:
            ATLASConfig instance
        """
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        logger.info(f"✓ Configuration loaded from {path}")
        return cls(**config_dict)
    
    def update(self, **kwargs):
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                old_value = getattr(self, key)
                setattr(self, key, value)
                logger.info(f"Updated {key}: {old_value} → {value}")
            else:
                logger.warning(f"Unknown parameter: {key}")
        
        # Re-validate after update
        self._validate_parameters()
    
    def get_n_patches(self) -> int:
        """Calculate number of patches."""
        return (self.seq_length - self.patch_length) // self.stride + 1
    
    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size with accumulation."""
        return self.batch_size * self.accumulation_steps
    
    def get_total_params(self) -> int:
        """Estimate total model parameters."""
        # Rough estimation
        patch_embed = self.patch_length * self.feature_size * self.d_model
        transformer = self.n_layers * (
            4 * self.d_model * self.d_model +  # Q, K, V, O projections
            2 * self.d_model * self.ff_dim      # FFN
        )
        lora = self.n_layers * 6 * self.d_model * self.lora_rank  # Q, K, V LoRA for each layer
        heads = len(self.symbols) * (self.d_model * 512 + 512 * 3)  # Actor/Critic per asset
        
        total = patch_embed + transformer + lora + heads
        return total
    
    def get_lora_params(self) -> int:
        """Estimate LoRA parameters."""
        return self.n_layers * 6 * self.d_model * self.lora_rank
    
    def summary(self) -> str:
        """Generate configuration summary."""
        summary = []
        summary.append("="*60)
        summary.append("ATLAS V2 CONFIGURATION SUMMARY")
        summary.append("="*60)
        
        summary.append("\n📊 DATA:")
        summary.append(f"  Assets: {len(self.symbols)} ({', '.join(self.symbols)})")
        summary.append(f"  Sequence Length: {self.seq_length}")
        summary.append(f"  Features: {self.feature_size}")
        summary.append(f"  Patches: {self.get_n_patches()}")
        
        summary.append("\n🧠 MODEL:")
        summary.append(f"  Architecture: PatchTST + LoRA")
        summary.append(f"  Dimension: {self.d_model}")
        summary.append(f"  Layers: {self.n_layers}")
        summary.append(f"  Heads: {self.n_heads}")
        summary.append(f"  Est. Total Params: {self.get_total_params():,}")
        summary.append(f"  Est. LoRA Params: {self.get_lora_params():,} ({100*self.get_lora_params()/self.get_total_params():.1f}%)")
        
        summary.append("\n🎯 TRAINING:")
        summary.append(f"  Epochs: {self.num_epochs}")
        summary.append(f"  Batch Size: {self.batch_size}")
        summary.append(f"  Effective Batch: {self.get_effective_batch_size()}")
        summary.append(f"  Learning Rate: {self.learning_rate}")
        summary.append(f"  LoRA LR: {self.lora_learning_rate}")
        
        summary.append("\n💾 SYSTEM:")
        summary.append(f"  Device: {self.device}")
        summary.append(f"  Mixed Precision: {self.use_amp}")
        summary.append(f"  Gradient Checkpointing: {self.gradient_checkpointing}")
        summary.append(f"  Memory Dir: {self.memory_dir}")
        
        summary.append("\n🔄 FEATURES:")
        summary.append(f"  Auto-Resume: {self.auto_resume}")
        summary.append(f"  Auto-Optimize: {self.auto_optimize}")
        summary.append(f"  Regime Detection: {self.enable_regime_detection}")
        summary.append(f"  Cross-Asset Attention: {self.cross_asset_attention}")
        summary.append(f"  Portfolio Mode: {self.portfolio_mode}")
        
        summary.append("\n" + "="*60)
        
        return "\n".join(summary)
    
    def __repr__(self) -> str:
        return self.summary()


# ==========================================
# CONFIGURATION PRESETS
# ==========================================

def get_default_config() -> ATLASConfig:
    """Get default configuration."""
    return ATLASConfig()


def get_minimal_config() -> ATLASConfig:
    """Get minimal configuration for testing."""
    return ATLASConfig(
        symbols=['EURUSD'],
        seq_length=128,
        d_model=64,
        n_layers=2,
        n_heads=4,
        num_epochs=10,
        batch_size=32
    )


def get_hft_config() -> ATLASConfig:
    """Get configuration optimized for high-frequency trading."""
    return ATLASConfig(
        symbols=['EURUSD', 'GBPUSD', 'USDJPY'],
        seq_length=256,
        online_update_freq=100,
        learning_rate=5e-5,
        batch_size=128,
        portfolio_mode='correlated',
        max_leverage=10.0
    )


def get_portfolio_config() -> ATLASConfig:
    """Get configuration for portfolio management."""
    return ATLASConfig(
        symbols=['EURUSD', 'GBPUSD', 'USDJPY', 'BTCUSD', 'ETHUSD', 'SPY', 'QQQ'],
        seq_length=512,
        num_epochs=200,
        portfolio_mode='portfolio_opt',
        cross_asset_attention=True,
        enable_regime_detection=True
    )


def get_research_config() -> ATLASConfig:
    """Get configuration for research and experimentation."""
    return ATLASConfig(
        symbols=['EURUSD', 'BTCUSD'],
        seq_length=512,
        d_model=512,
        n_layers=12,
        n_heads=16,
        num_epochs=500,
        batch_size=32,
        auto_optimize=False,
        gradient_checkpointing=True
    )


# ==========================================
# MODULE EXPORTS
# ==========================================

__all__ = [
    'ATLASConfig',
    'get_default_config',
    'get_minimal_config',
    'get_hft_config',
    'get_portfolio_config',
    'get_research_config'
]


if __name__ == '__main__':
    # Test configuration
    config = get_default_config()
    print(config.summary())
    
    # Test save/load
    config.save('test_config.json')
    loaded_config = ATLASConfig.load('test_config.json')
    print("\n✓ Configuration save/load test passed")
    
    # Test validation
    try:
        bad_config = ATLASConfig(d_model=100, n_heads=7)  # Not divisible
    except ValueError as e:
        print(f"\n✓ Validation test passed: {e}")
