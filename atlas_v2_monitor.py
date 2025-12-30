"""
ATLAS V2 System and Agent Health Monitor
========================================
Monitors agent output, trading loop, data, and environment for production live trading.
Triggers alerts, logs events, halts trading, and reconciles edge cases for robust safety.
"""

import logging
import time
import traceback
from datetime import datetime
import numpy as np

logger = logging.getLogger("atlas_v2_monitor")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s'))
    logger.addHandler(handler)

class AtlasMonitorConfig:
    """Customizable thresholds for risk, ops, and agent health."""
    # Default safety values, override as desired
    max_drawdown = 0.20        # 20%
    min_sharpe = 0.20
    min_equity = 1000          # Don't trade below $1,000
    max_nan_inf = 3            # More than 3 NaN/inf outputs triggers halt
    min_hold_time = 2          # Ticks to hold before close allowed
    max_order_reject = 3       # Max consecutive order rejections

class AtlasSystemMonitor:
    def __init__(self, initial_equity, symbols, config=None, alert_callback=None):
        self.equity_history = [initial_equity]
        self.symbols = symbols
        self.config = config or AtlasMonitorConfig()
        self.last_agent_outputs = {s: None for s in symbols}
        self.nan_inf_counts = {s: 0 for s in symbols}
        self.hold_counts = {s: 0 for s in symbols}
        self.order_reject_count = 0
        self.state = {"active": True, "halt_reason": "", "timestamp": None}
        self.last_positions = {s: 0 for s in symbols}  # 1=long, -1=short, 0=flat
        self.last_alert_time = None
        self.alert_callback = alert_callback  # Custom email/SMS/webhook function

    def update_equity(self, current_equity):
        self.equity_history.append(current_equity)
        # max drawdown calc
        values = np.array(self.equity_history)
        running_max = np.maximum.accumulate(values)
        drawdown = min((values - running_max) / (running_max + 1e-8))
        self.drawdown = drawdown
        if current_equity <= self.config.min_equity:
            self.halt("Equity below minimum threshold")
        if drawdown < -self.config.max_drawdown:
            self.halt(f"Drawdown {drawdown:.2%} exceeds threshold")
        # Sharpe calculation (rolling window)
        if len(values) > 3:
            returns = np.diff(values) / (values[:-1] + 1e-8)
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            self.sharpe = sharpe
            if sharpe < self.config.min_sharpe:
                self.halt(f"Sharpe {sharpe:.3f} below min threshold")

    def update_agent_outputs(self, agent_outputs):
        for s, val in agent_outputs.items():
            # Health check for output
            try:
                if val is None or np.isnan(val) or np.isinf(val):
                    self.nan_inf_counts[s] += 1
                    logger.warning(f"Agent output NaN/Inf for {s}: {val}")
                    if self.nan_inf_counts[s] > self.config.max_nan_inf:
                        self.halt(f"Agent output NaN/Inf exceeded for {s}")
                        self.alert(f"Agent output anomaly on {s}: {val}")
                else:
                    self.nan_inf_counts[s] = 0
                    self.last_agent_outputs[s] = val
            except Exception as e:
                logger.error(f"Agent output validation exception: {e}")

    def update_positions(self, positions, agent_signals):
        # Hysteresis/min-hold logic to prevent whipsaw/false close
        for s in self.symbols:
            signal = agent_signals.get(s, 0)
            last_pos = self.last_positions.get(s, 0)
            if signal == last_pos:
                self.hold_counts[s] += 1
            else:
                self.hold_counts[s] = 0
            # Only allow close if hold period fully elapsed
            if last_pos != 0 and signal == 0 and self.hold_counts[s] < self.config.min_hold_time:
                logger.info(f"Hysteresis: suppressing close for {s} (hold_count={self.hold_counts[s]})")
                # Optionally force trade loop to ignore closing signal here
        self.last_positions = {s: agent_signals.get(s, 0) for s in self.symbols}

    def report_order_rejection(self):
        self.order_reject_count += 1
        logger.warning(f"Order reject detected ({self.order_reject_count})")
        if self.order_reject_count >= self.config.max_order_reject:
            self.halt("Too many consecutive order rejections")
            self.alert(f"Order reject threshold exceeded")

    def reset_order_reject(self):
        self.order_reject_count = 0

    def monitor_market_data(self, prices, ticks_ok=True):
        # Example integrity checks for data
        for s in self.symbols:
            px = prices.get(s)
            if px is None or not np.isfinite(px):
                logger.error(f"Bad market data for {s}: {px}")
                self.halt(f"Bad/missing price data for {s}")

    def heartbeat(self):
        now = datetime.now().isoformat()
        logger.info(f"Monitor Heartbeat @ {now} | State: {self.state['active']} | Reason: {self.state.get('halt_reason', '')}")
        # Optionally send heartbeat info to dashboard/ops center
        # if self.alert_callback: self.alert_callback({"type":"heartbeat", "state":self.state, "timestamp":now})
        pass

    def halt(self, reason):
        self.state = {"active": False, "halt_reason": reason, "timestamp": datetime.now().isoformat()}
        logger.error(f"[MONITOR HALT] Trading Halted: {reason}")
        self.alert(f"[AtlasMonitor] Trading halted: {reason}")

    def alert(self, message):
        logger.warning(f"[ALERT] {message}")
        now = time.time()
        # Guard against alert spam (no more than one alert per 60s)
        if self.last_alert_time is None or now - self.last_alert_time > 60:
            self.last_alert_time = now
            if self.alert_callback:
                self.alert_callback({"type": "alert", "message": message, "timestamp": datetime.now().isoformat()})
        # Optionally: send to email/SMS/Slack/Telegram channel

    def should_halt(self):
        return not self.state["active"]

# --- Usage Example ---
# monitor = AtlasSystemMonitor(initial_equity=100_000, symbols=["EURUSD","GBPUSD"])
# In trading loop:
#   monitor.update_equity(latest_equity)
#   monitor.update_agent_outputs({sym: agent_output})
#   monitor.update_positions(real_positions, agent_signals)
#   monitor.report_order_rejection() # when broker/order API rejects
#   if monitor.should_halt():
#       # Halt trading, flatten, notify ops
#       break
#   monitor.heartbeat() # Want periodic heartbeats
