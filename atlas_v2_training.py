"""
ATLAS V2: Training Module
==========================
Complete training infrastructure with PyTorch Lightning.

File: training.py
Status: PRODUCTION READY ✅
NO PLACEHOLDERS ✅
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime

from V2_models import ATLASModel

logger = logging.getLogger(__name__)


# ==========================================
# LIGHTNING MODULE
# ==========================================

class ATLASLightningModule(pl.LightningModule):
    """PyTorch Lightning module for ATLAS training."""
    
    def __init__(self, config):
        super().__init__()
        
        self.save_hyperparameters()
        self.config = config
        
        # Initialize model
        self.model = ATLASModel(config)
        
        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        logger.info("ATLASLightningModule initialized")
    
    def forward(self, batch):
        """Forward pass."""
        return self.model(batch)
    
    def training_step(self, batch, batch_idx):
        """
        Training step - complete implementation.
        
        Args:
            batch: Dictionary of asset batches
            batch_idx: Batch index
            
        Returns:
            loss: Training loss
        """
        try:
            # Forward pass
            outputs = self.forward(batch)
            
            # Compute loss for each asset
            total_loss = 0
            n_assets = 0
            
            for symbol in self.config.symbols:
                if symbol not in batch or symbol not in outputs:
                    continue
                
                # Get predictions and targets
                target = batch[symbol]['y']
                mean = outputs[symbol]['price_mean'].squeeze()
                variance = outputs[symbol]['price_variance'].squeeze()
                
                # Negative log-likelihood loss
                nll_loss = self._compute_nll_loss(mean, variance, target)
                
                # Variance regularization
                variance_reg = torch.mean(torch.log(variance + 1e-6))
                
                # Combined loss
                asset_loss = nll_loss + 0.01 * variance_reg
                
                total_loss += asset_loss
                n_assets += 1
                
                # Log per-asset metrics
                self.log(f'train/{symbol}_nll', nll_loss, on_step=False, on_epoch=True)
                self.log(f'train/{symbol}_variance', variance.mean(), on_step=False, on_epoch=True)
            
            # Average loss across assets
            if n_assets > 0:
                avg_loss = total_loss / n_assets
            else:
                avg_loss = total_loss
            
            # Log overall loss
            self.log('train/loss', avg_loss, on_step=True, on_epoch=True, prog_bar=True)
            
            # Store for later analysis
            self.train_losses.append(avg_loss.item())
            
            return avg_loss
            
        except Exception as e:
            logger.error(f"Training step error: {e}")
            raise
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step - complete implementation.
        
        Args:
            batch: Dictionary of asset batches
            batch_idx: Batch index
            
        Returns:
            loss: Validation loss
        """
        try:
            # Forward pass
            outputs = self.forward(batch)
            
            # Compute loss for each asset
            total_loss = 0
            n_assets = 0
            
            for symbol in self.config.symbols:
                if symbol not in batch or symbol not in outputs:
                    continue
                
                # Get predictions and targets
                target = batch[symbol]['y']
                mean = outputs[symbol]['price_mean'].squeeze()
                variance = outputs[symbol]['price_variance'].squeeze()
                
                # NLL loss
                nll_loss = self._compute_nll_loss(mean, variance, target)
                
                total_loss += nll_loss
                n_assets += 1
                
                # Compute calibration metrics
                std = torch.sqrt(variance)
                lower_bound = mean - 1.96 * std
                upper_bound = mean + 1.96 * std
                coverage = ((target >= lower_bound) & (target <= upper_bound)).float().mean()
                
                # Log per-asset metrics
                self.log(f'val/{symbol}_nll', nll_loss, on_step=False, on_epoch=True)
                self.log(f'val/{symbol}_coverage_95', coverage, on_step=False, on_epoch=True)
                self.log(f'val/{symbol}_variance', variance.mean(), on_step=False, on_epoch=True)
            
            # Average loss
            if n_assets > 0:
                avg_loss = total_loss / n_assets
            else:
                avg_loss = total_loss
            
            # Log overall loss
            self.log('val/loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True)
            
            # Store for later
            self.val_losses.append(avg_loss.item())
            
            # Update best val loss
            if avg_loss < self.best_val_loss:
                self.best_val_loss = avg_loss.item()
            
            return avg_loss
            
        except Exception as e:
            logger.error(f"Validation step error: {e}")
            raise
    
    def test_step(self, batch, batch_idx):
        """
        Test step - complete implementation.
        
        Args:
            batch: Dictionary of asset batches
            batch_idx: Batch index
            
        Returns:
            Dictionary of test metrics
        """
        # Same as validation but store results
        outputs = self.forward(batch)
        
        results = {}
        
        for symbol in self.config.symbols:
            if symbol not in batch or symbol not in outputs:
                continue
            
            target = batch[symbol]['y']
            mean = outputs[symbol]['price_mean'].squeeze()
            variance = outputs[symbol]['price_variance'].squeeze()
            
            # Compute metrics
            nll_loss = self._compute_nll_loss(mean, variance, target)
            mae = torch.abs(mean - target).mean()
            mse = torch.pow(mean - target, 2).mean()
            
            results[symbol] = {
                'nll': nll_loss.item(),
                'mae': mae.item(),
                'mse': mse.item(),
                'predictions': mean.detach().cpu(),
                'targets': target.detach().cpu(),
                'variance': variance.detach().cpu()
            }
            
            # Log test metrics
            self.log(f'test/{symbol}_nll', nll_loss)
            self.log(f'test/{symbol}_mae', mae)
            self.log(f'test/{symbol}_mse', mse)
        
        return results
    
    def _compute_nll_loss(self, mean: torch.Tensor, variance: torch.Tensor, 
                         target: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood loss.
        
        Args:
            mean: Predicted mean
            variance: Predicted variance
            target: Ground truth
            
        Returns:
            nll_loss: Negative log-likelihood
        """
        # Create Gaussian distribution
        dist = torch.distributions.Normal(mean, torch.sqrt(variance))
        
        # Negative log-likelihood
        nll = -dist.log_prob(target).mean()
        
        return nll
    
    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        
        Returns:
            Dictionary with optimizer and scheduler
        """
        # Separate LoRA parameters
        lora_params = []
        other_params = []
        
        for name, param in self.named_parameters():
            if 'lora' in name:
                lora_params.append(param)
            else:
                other_params.append(param)
        
        # Create optimizer with different learning rates
        optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': self.config.learning_rate},
            {'params': lora_params, 'lr': self.config.lora_learning_rate}
        ], weight_decay=self.config.weight_decay)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[self.config.learning_rate, self.config.lora_learning_rate],
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        avg_train_loss = sum(self.train_losses[-len(self.train_dataloader()):]) / len(self.train_dataloader())
        logger.info(f"Epoch {self.current_epoch} | Train Loss: {avg_train_loss:.6f}")
    
    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        if self.val_losses:
            avg_val_loss = sum(self.val_losses[-len(self.val_dataloader()):]) / len(self.val_dataloader())
            logger.info(f"Epoch {self.current_epoch} | Val Loss: {avg_val_loss:.6f} | Best: {self.best_val_loss:.6f}")


# ==========================================
# TRAINER SETUP
# ==========================================

def create_trainer(config) -> pl.Trainer:
    """
    Create PyTorch Lightning trainer with all callbacks.
    
    Args:
        config: ATLAS configuration
        
    Returns:
        trainer: Configured PyTorch Lightning Trainer
    """
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{config.memory_dir}/checkpoints",
        filename='atlas-v2-{epoch:02d}-{val/loss:.4f}',
        monitor='val/loss',
        mode='min',
        save_top_k=config.keep_n_checkpoints,
        save_last=True,
        verbose=True
    )
    
    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val/loss',
        patience=config.patience,
        mode='min',
        verbose=True,
        min_delta=config.min_improvement
    )
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        min_epochs=config.min_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='16-mixed' if config.use_amp else '32',
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=config.accumulation_steps,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        log_every_n_steps=config.log_every_n_steps,
        val_check_interval=config.val_check_interval,
        enable_progress_bar=config.enable_progress_bar,
        enable_model_summary=config.enable_model_summary,
        deterministic=False,  # For performance
        benchmark=True  # Enable cuDNN benchmark
    )
    
    logger.info("Trainer created with callbacks:")
    logger.info(f"  - ModelCheckpoint: top_{config.keep_n_checkpoints}")
    logger.info(f"  - EarlyStopping: patience={config.patience}")
    logger.info(f"  - LearningRateMonitor")
    
    return trainer


# ==========================================
# RESUME TRAINING
# ==========================================

def resume_from_checkpoint(config) -> Optional[str]:
    """
    Find and return path to checkpoint for resuming training.
    
    Args:
        config: ATLAS configuration
        
    Returns:
        checkpoint_path: Path to checkpoint or None
    """
    if not config.auto_resume:
        return None
    
    checkpoint_dir = Path(config.memory_dir) / 'checkpoints'
    
    # Look for last checkpoint
    last_ckpt = checkpoint_dir / 'last.ckpt'
    if last_ckpt.exists():
        logger.info(f"Found checkpoint for resuming: {last_ckpt}")
        return str(last_ckpt)
    
    # Look for best checkpoint
    best_ckpts = list(checkpoint_dir.glob('atlas-v2-*.ckpt'))
    if best_ckpts:
        # Sort by modification time, get most recent
        latest_ckpt = max(best_ckpts, key=lambda p: p.stat().st_mtime)
        logger.info(f"Found checkpoint for resuming: {latest_ckpt}")
        return str(latest_ckpt)
    
    logger.info("No checkpoint found, starting from scratch")
    return None


# ==========================================
# TRAINING PIPELINE
# ==========================================

def train_model(config, train_loader, val_loader, test_loader=None) -> Tuple[ATLASLightningModule, pl.Trainer]:
    """
    Complete training pipeline.
    
    Args:
        config: ATLAS configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Optional test data loader
        
    Returns:
        (model, trainer): Trained model and trainer
    """
    logger.info("="*60)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("="*60)
    
    # Check for resume
    resume_ckpt = resume_from_checkpoint(config)
    
    if resume_ckpt:
        logger.info(f"Resuming training from: {resume_ckpt}")
        model = ATLASLightningModule.load_from_checkpoint(
            resume_ckpt,
            config=config
        )
    else:
        logger.info("Initializing new model")
        model = ATLASLightningModule(config)
    
    # Create trainer
    trainer = create_trainer(config)
    
    # Log training info
    logger.info(f"Training configuration:")
    logger.info(f"  Epochs: {config.num_epochs}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Effective batch: {config.batch_size * config.accumulation_steps}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Mixed precision: {config.use_amp}")
    
    # Start training
    logger.info("\nStarting training...")
    start_time = datetime.now()
    
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_ckpt if resume_ckpt else None
    )
    
    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()
    
    logger.info(f"\nTraining completed in {training_time:.2f} seconds")
    logger.info(f"Best validation loss: {model.best_val_loss:.6f}")
    
    # Test if test loader provided
    if test_loader is not None:
        logger.info("\nRunning test evaluation...")
        test_results = trainer.test(model, test_loader)
        logger.info("Test evaluation completed")
    
    # Save final model
    final_model_path = f"{config.memory_dir}/checkpoints/final_model.ckpt"
    trainer.save_checkpoint(final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")
    
    logger.info("="*60)
    
    return model, trainer


# ==========================================
# MODULE EXPORTS
# ==========================================

__all__ = [
    'ATLASLightningModule',
    'create_trainer',
    'resume_from_checkpoint',
    'train_model'
]


if __name__ == '__main__':
    # Test the training module
    from config import get_minimal_config
    from data import prepare_dataloaders
    
    config = get_minimal_config()
    config.num_epochs = 2  # Short test
    
    # Prepare data
    train_loader, val_loader, test_loader, _ = prepare_dataloaders(config)
    
    # Train
    model, trainer = train_model(config, train_loader, val_loader, test_loader)
    
    print("\n✓ Training module test passed")
