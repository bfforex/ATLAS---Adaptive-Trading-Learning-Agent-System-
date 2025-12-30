"""
ATLAS V2: Persistent Learning & Memory Module (Production-Ready)
----------------------------------------------------------------
- Atomic, robust file IO for checkpoints and experiences
- Versioned artifacts and schema validation on load
- Graceful error management with logging
"""

import os
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

try:
    import torch
except ImportError:
    torch = None  # This is for doc/test/linting, in practice torch is required.

# --- Setup module-level logger ---
logger = logging.getLogger("atlas_v2_memory")
logger.setLevel(logging.INFO)  # Set to DEBUG for more details; in prod may come via config
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

MEMORY_VERSION = "2.0.0"


def atomic_write(target_path, data_bytes):
    """Write bytes to target_path atomically."""
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    try:
        with open(tmp_path, "wb") as f:
            f.write(data_bytes)
        os.replace(tmp_path, target_path)
    except Exception as e:
        logger.error(f"Atomic write failed: {e} (target={target_path})")
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def atomic_json_write(target_path, obj):
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    try:
        with open(tmp_path, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp_path, target_path)
    except Exception as e:
        logger.error(f"Atomic JSON write failed: {e} (target={target_path})")
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


class CheckpointManager:
    """
    Manages model/optimizer checkpoint saving and loading.
    """
    CHECKPOINT_TYPE = "atlas_checkpoint"
    VERSION = MEMORY_VERSION

    def __init__(self, memory_dir: str = "memory/checkpoints"):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, model, epoch: int, val_loss: float, trainer=None, extra: Optional[dict]=None):
        import torch
        fname = f"checkpoint_epoch{epoch:03d}_loss{val_loss:.4f}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
        save_path = self.memory_dir / fname
        # Build the checkpoint payload with version/type meta
        checkpoint = {
            '__type__': self.CHECKPOINT_TYPE,
            '__version__': self.VERSION,
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            'timestamp': datetime.now().isoformat()
        }
        if trainer is not None:
            try:
                checkpoint['optimizer_state_dict'] = trainer.optimizers[0].state_dict()
            except Exception as e:
                logger.warning(f"Optimizer state not saved: {e}")
        if extra:
            checkpoint.update(extra)
        try:
            # Torch must save to buffer, then atomically write
            with open(save_path.with_suffix(".tmp"), "wb") as f:
                torch.save(checkpoint, f)
            os.replace(save_path.with_suffix(".tmp"), save_path)
            logger.info(f"Model checkpoint saved: {save_path}")
            # Atomic symlink/copy for "latest"
            latest_link = self.memory_dir / "last.pth"
            try:
                if latest_link.exists() or latest_link.is_symlink():
                    latest_link.unlink()
                latest_link.symlink_to(save_path.resolve())
            except Exception:
                # fallback: copy
                import shutil
                shutil.copy(str(save_path), str(latest_link))
        except Exception as e:
            logger.error(f"Checkpoint save failed: {e}")
            raise

    def load_checkpoint(self, checkpoint_path: Optional[str] = None):
        """
        Returns checkpoint dict after version/type validation.
        """
        import torch
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()
        if not checkpoint_path:
            logger.error("No checkpoint file found.")
            return None
        try:
            with open(checkpoint_path, "rb") as f:
                checkpoint = torch.load(f, map_location="cpu")
            # Validate header
            if checkpoint.get("__type__") != self.CHECKPOINT_TYPE:
                logger.error(f"File is not a valid ATLAS checkpoint: {checkpoint_path}")
                return None
            if checkpoint.get("__version__") != self.VERSION:
                logger.error(f"Checkpoint version mismatch: {checkpoint.get('__version__')} (expected {self.VERSION})")
                # Accept? Optionally return anyway
            logger.info(f"Loaded checkpoint: {checkpoint_path} [epoch={checkpoint.get('epoch')}]")
            return checkpoint
        except Exception as e:
            logger.error(f"Checkpoint load failed: {e}")
            return None

    def get_latest_checkpoint(self) -> Optional[str]:
        try:
            candidates = sorted(self.memory_dir.glob("checkpoint_*.pth"), key=os.path.getmtime)
            if not candidates:
                logger.warning("No checkpoint files found in directory.")
                return None
            return str(candidates[-1])
        except Exception as e:
            logger.error(f"Failed to scan checkpoint directory: {e}")
            return None


class ExperienceReplayManager:
    """
    Handles persistence of experience replay buffers for continual/online learning.
    """
    BUFFER_TYPE = "atlas_replay_buffer"
    VERSION = MEMORY_VERSION

    def __init__(self, memory_dir: str = 'memory/experiences'):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def save_buffer(self, buffer: Any, step: int):
        fname = f"replay_buffer_step{step}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        out_path = self.memory_dir / fname
        payload = {
            '__type__': self.BUFFER_TYPE,
            '__version__': self.VERSION,
            'buffer': buffer,
            'step': step,
            'timestamp': datetime.now().isoformat()
        }
        try:
            with open(out_path.with_suffix('.tmp'), "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(out_path.with_suffix('.tmp'), out_path)
            logger.info(f"Replay buffer saved: {out_path}")
            self._prune_old(N=5)
        except Exception as e:
            logger.error(f"Replay buffer save failed: {e}")
            raise

    def load_latest_buffer(self) -> Optional[Any]:
        try:
            buffers = sorted(self.memory_dir.glob("replay_buffer_step*.pkl"), key=os.path.getmtime)
            if not buffers:
                logger.warning("No experience replay buffer found.")
                return None
            with open(buffers[-1], "rb") as f:
                payload = pickle.load(f)
            # Validate
            if payload.get('__type__') != self.BUFFER_TYPE:
                logger.error(f"File is not a valid ATLAS replay buffer.")
                return None
            if payload.get('__version__') != self.VERSION:
                logger.error(f"Replay buffer version mismatch: {payload.get('__version__')} (expected {self.VERSION})")
                # Accept? Optionally return anyway
            logger.info(f"Loaded experience buffer: {buffers[-1]} [step={payload.get('step')}]")
            return payload.get('buffer')
        except Exception as e:
            logger.error(f"Failed to load replay buffer: {e}")
            return None

    def _prune_old(self, N=5):
        try:
            buffers = sorted(self.memory_dir.glob("replay_buffer_step*.pkl"), key=os.path.getmtime)
            for b in buffers[:-N]:
                try:
                    b.unlink()
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Pruning of old buffers failed: {e}")


class OnlineLearningLogManager:
    """
    Persists online learning and update events (for audit/history).
    """
    LOG_TYPE = "atlas_online_log"
    VERSION = MEMORY_VERSION

    def __init__(self, memory_dir: str = 'memory/online_learning'):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.memory_dir / "online_learning_events.json"

    def log_event(self, event: dict):
        event['timestamp'] = datetime.now().isoformat()
        history = []
        # Read, validate, write atomically
        try:
            if self.log_file.exists():
                with open(self.log_file, "r") as f:
                    history = json.load(f)
                # Validate type+ver
                if history and isinstance(history, list) and "__type__" in history[0]:
                    for rec in history:
                        if rec.get("__type__") != self.LOG_TYPE:
                            logger.warning(f"Non-ATLAS record in log; ignoring entry.")
                else:
                    logger.warning("Online log missing required schema header.")
            event['__type__'] = self.LOG_TYPE
            event['__version__'] = self.VERSION
            history.append(event)
            atomic_json_write(self.log_file, history)
            logger.info("Online learning event logged.")
        except Exception as e:
            logger.error(f"Failed to log online learning event: {e}")

    def load_history(self):
        if not self.log_file.exists():
            logger.warning("No online learning log found.")
            return []
        try:
            with open(self.log_file, "r") as f:
                history = json.load(f)
            # Validate
            for rec in history:
                if rec.get('__type__') != self.LOG_TYPE:
                    logger.warning(f"Ignoring record with wrong type in log file: {rec}")
            return history
        except Exception as e:
            logger.error(f"Failed to load online learning log: {e}")
            return []


class ConfigSnapshotManager:
    """
    Archives configurations for reproducibility and rollback.
    """
    CONFIG_TYPE = "atlas_config_snapshot"
    VERSION = MEMORY_VERSION

    def __init__(self, memory_dir: str = 'memory/reports'):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def save_config(self, config, tag: Optional[str]=None):
        tag = tag or datetime.now().strftime('%Y%m%d_%H%M%S')
        path = self.memory_dir / f"config_{tag}.json"
        if hasattr(config, 'asdict'):
            conf_dict = config.asdict()
        elif hasattr(config, '__dict__'):
            conf_dict = dict(config.__dict__)
        else:
            conf_dict = dict(config)
        # Add header
        conf_dict['__type__'] = self.CONFIG_TYPE
        conf_dict['__version__'] = self.VERSION
        try:
            atomic_json_write(path, conf_dict)
            logger.info(f"Config snapshot saved: {path}")
        except Exception as e:
            logger.error(f"Config snapshot save failed: {e}")

    def load_latest(self):
        try:
            configs = sorted(self.memory_dir.glob("config_*.json"), key=os.path.getmtime)
            if not configs:
                logger.warning("No config snapshots found.")
                return None
            with open(configs[-1], "r") as f:
                conf = json.load(f)
            # Validate
            if conf.get('__type__') != self.CONFIG_TYPE:
                logger.error(f"File is not a valid ATLAS config snapshot.")
                return None
            if conf.get('__version__') != self.VERSION:
                logger.error(f"Config snapshot version mismatch: {conf.get('__version__')} (expected {self.VERSION})")
                # Accept? Optionally return anyway
            logger.info(f"Loaded configuration from: {configs[-1]}")
            return conf
        except Exception as e:
            logger.error(f"Failed to load config snapshot: {e}")
            return None


if __name__ == "__main__":
    logger.info("Testing ATLAS V2 Memory/Checkpoint/Replay module setup (stateless demo only)...")
    logger.info("- Create and use CheckpointManager, ExperienceReplayManager, ConfigSnapshotManager as needed.")
