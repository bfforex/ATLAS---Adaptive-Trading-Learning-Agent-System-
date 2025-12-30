"""
ATLAS V2: Automated Improvement System
======================================
Encapsulates all logic for automated performance tracking,
hyperparameter adjustment, degradation handling, and actionable
recommendations for model improvement.

To use:
    from improvement import AutomatedImprovementSystem
    imp_sys = AutomatedImprovementSystem(config, memory_manager=None)
    actions = imp_sys.evaluate_performance(epoch, metrics)
    adjustments = imp_sys.adjust_hyperparameters(model, actions)
    recs = imp_sys.generate_improvement_recommendations()
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List


class AutomatedImprovementSystem:
    """
    Monitors performance and automatically adjusts training to ensure improvement.

    Features:
      - Performance tracking across sessions
      - Automatic hyperparameter adjustment
      - Learning rate scheduling based on performance
      - Early stopping on degradation
      - Self-healing on divergence
      - Adaptive exploration
    """

    def __init__(self, config, memory_manager=None):
        self.config = config
        self.memory_manager = memory_manager
        # Internal tracking
        self.performance_history = []
        self.best_performance = {
            'sharpe': -999, 'pnl': -999, 'win_rate': 0, 'epoch': 0
        }
        self.hyperparam_history = []
        self.degradation_count = 0
        self.max_degradation_tolerance = getattr(config, "degradation_tolerance", 3)
        self.adjustments_made = []

    def evaluate_performance(self, epoch: int, metrics: Dict) -> Dict:
        """
        Comprehensive performance evaluation returning dict of recommended actions.
        """
        actions = {
            'continue_training': True,
            'adjust_lr': False,
            'increase_exploration': False,
            'decrease_exploration': False,
            'trigger_fine_tune': False,
            'save_checkpoint': False,
            'early_stop': False,
            'reason': []
        }

        current_sharpe = metrics['portfolio']['sharpe']
        current_pnl = metrics['portfolio']['total_pnl']
        current_win_rate = np.mean([m['win_rate'] for m in metrics['assets'].values()])

        # Check for improvement
        if current_sharpe > self.best_performance['sharpe']:
            self.best_performance.update({
                'sharpe': current_sharpe,
                'pnl': current_pnl,
                'win_rate': current_win_rate,
                'epoch': epoch
            })
            actions['save_checkpoint'] = True
            actions['reason'].append(f"New best Sharpe: {current_sharpe:.3f}")
            self.degradation_count = 0
        else:
            self.degradation_count += 1

        # Degradation logic
        if current_sharpe < self.best_performance['sharpe'] * 0.7:
            actions['reason'].append(f"Performance degraded: {current_sharpe:.3f} < {self.best_performance['sharpe']*0.7:.3f}")
            if self.degradation_count >= self.max_degradation_tolerance:
                if getattr(self.config, 'early_stop_on_degradation', True):
                    actions['early_stop'] = True
                    actions['reason'].append("Early stop: Consistent degradation detected")
                else:
                    actions['trigger_fine_tune'] = True
                    actions['reason'].append("Triggering recovery fine-tuning")

        # Below threshold
        if current_sharpe < getattr(self.config, 'performance_threshold', 1.5):
            actions['adjust_lr'] = True
            actions['increase_exploration'] = True
            actions['reason'].append(f"Below threshold: {current_sharpe:.3f} < {self.config.performance_threshold}")

        # Stagnation detection (minimal improvement)
        if len(self.performance_history) >= 10:
            recent_sharpes = [p['sharpe'] for p in self.performance_history[-10:]]
            if np.std(recent_sharpes) < 0.05:
                actions['increase_exploration'] = True
                actions['reason'].append("Stagnation detected: Increasing exploration")

        # Excessive exploration (low win rate)
        if current_win_rate < 0.45:
            actions['decrease_exploration'] = True
            actions['reason'].append(f"Low win rate: {current_win_rate:.2%}")

        # Record performance
        self.performance_history.append({
            'epoch': epoch,
            'sharpe': current_sharpe,
            'pnl': current_pnl,
            'win_rate': current_win_rate,
            'timestamp': datetime.now().isoformat()
        })

        return actions

    def adjust_hyperparameters(self, model, actions: Dict) -> Dict:
        """
        Automatically adjust hyperparameters based on actions.
        Returns dict of changes made.
        """
        adjustments = {}

        # Learning rate adjustment
        if actions.get('adjust_lr'):
            old_lr = self.config.learning_rate
            self.config.learning_rate *= 0.5  # Reduce by half
            adjustments['learning_rate'] = {
                'old': old_lr,
                'new': self.config.learning_rate,
                'reason': 'Performance below threshold'
            }

        # Increase exploration (entropy coefficient)
        if actions.get('increase_exploration'):
            old_entropy = self.config.entropy_coef
            self.config.entropy_coef = min(0.1, self.config.entropy_coef * 1.5)
            adjustments['entropy_coef'] = {
                'old': old_entropy,
                'new': self.config.entropy_coef,
                'reason': 'Increasing exploration'
            }

        # Decrease exploration
        if actions.get('decrease_exploration'):
            old_entropy = self.config.entropy_coef
            self.config.entropy_coef = max(0.001, self.config.entropy_coef * 0.7)
            adjustments['entropy_coef'] = {
                'old': old_entropy,
                'new': self.config.entropy_coef,
                'reason': 'Decreasing exploration'
            }

        # Record and persist any changes
        if adjustments:
            self.adjustments_made.append({
                'epoch': len(self.performance_history),
                'adjustments': adjustments,
                'timestamp': datetime.now().isoformat()
            })
            adjustment_file = Path(self.config.memory_dir) / 'hyperparameter_adjustments.json'
            with open(adjustment_file, 'w') as f:
                json.dump(self.adjustments_made, f, indent=2)

        return adjustments

    def generate_improvement_recommendations(self) -> List[str]:
        """Generate human-readable recommendations based on tracked performance."""
        recommendations = []

        if len(self.performance_history) < 5:
            recommendations.append("⏳ Insufficient data for recommendations. Continue training.")
            return recommendations

        recent_performance = self.performance_history[-5:]
        avg_sharpe = np.mean([p['sharpe'] for p in recent_performance])
        trend = np.polyfit(range(5), [p['sharpe'] for p in recent_performance], 1)[0]

        if avg_sharpe < 1.0:
            recommendations.append("⚠️ LOW PERFORMANCE: Consider reviewing data quality and feature engineering")
        if trend < -0.05:
            recommendations.append("⚠️ DECLINING TREND: Performance is degrading. Consider:")
            recommendations.append("   - Reducing learning rate")
            recommendations.append("   - Increasing regularization")
            recommendations.append("   - Loading earlier checkpoint")
        if avg_sharpe > 2.0 and trend > 0.05:
            recommendations.append("✅ EXCELLENT: Agent is performing well and improving")
        if self.degradation_count > 0:
            recommendations.append(f"⚠️ Degradation count: {self.degradation_count}/{self.max_degradation_tolerance}")

        return recommendations

# If you intend to support CLI/test:
if __name__ == '__main__':
    print("AutomatedImprovementSystem module test: (no stateful demonstration)")
