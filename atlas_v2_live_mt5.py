"""
ATLAS V2: XMGlobal MT5 Live Trading Module (Production Ready)
=============================================================
Adds agent/ops health monitoring, performance tracking, and real-time safety features.
"""

import MetaTrader5 as mt5
import logging
import time
from datetime import datetime
from pathlib import Path
import sys
import numpy as np

logger = logging.getLogger("atlas_v2_live_mt5")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s'))
    logger.addHandler(handler)

class AgentPerformanceMonitor:
    """Tracks live performance and safety metrics."""
    def __init__(self, initial_equity, max_drawdown=0.2, min_sharpe=0.2, window=100):
        self.equity_history = [initial_equity]
        self.max_drawdown = max_drawdown
        self.min_sharpe = min_sharpe
        self.window = window
        self.signal_counters = {"nan": 0, "inf": 0, "ok": 0}
        self.stop_trading = False

    def update(self, current_equity):
        self.equity_history.append(current_equity)
        if len(self.equity_history) > self.window:
            self.equity_history = self.equity_history[-self.window:]
        # Calculate running performance
        returns = np.diff(self.equity_history) / (np.array(self.equity_history[:-1]) + 1e-8)
        if len(returns) > 1:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            max_dd = self._compute_drawdown(self.equity_history)
            logger.info(f"[PERF] Sharpe: {sharpe:.3f} | Max DD: {max_dd:.2%}")
            if sharpe < self.min_sharpe or max_dd <= -self.max_drawdown:
                logger.warning("[SAFETY] Triggered stop: performance degraded (Sharpe < threshold or Max DD exceeded)")
                self.stop_trading = True
            else:
                self.stop_trading = False

    def _compute_drawdown(self, equity_curve):
        values = np.array(equity_curve)
        running_max = np.maximum.accumulate(values)
        drawdowns = (values - running_max) / (running_max + 1e-8)
        return float(drawdowns[-1])

    def agent_output_check(self, val):
        """Catch bad output from model."""
        if np.isnan(val):
            self.signal_counters["nan"] += 1
            logger.error("Agent output is NaN!")
            if self.signal_counters["nan"] > 3:
                self.stop_trading = True
                logger.error("[SAFETY] Too many NaN signals! Trading stopped.")
        elif np.isinf(val):
            self.signal_counters["inf"] += 1
            logger.error("Agent output is Inf!")
        else:
            self.signal_counters["ok"] += 1

class MT5Connector:
    ... # Same as previous, for brevity paste original or above

class LiveTradingMT5Loop:
    def __init__(self, agent, config, connector: MT5Connector, run_dir=None, interval_sec: int = 60):
        self.agent = agent
        self.config = config
        self.connector = connector
        self.symbols = connector.symbols
        self.run_dir = Path(run_dir or (Path("mt5_live_logs") / datetime.now().strftime("%Y%m%d_%H%M%S")))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.interval_sec = interval_sec
        self.trade_log = []
        self.heartbeat_log = []
        self.monitor = None

    def run(self, n_steps: int = None):
        self.connector.connect()
        acc = self.connector.get_account_info()
        equity = acc.get("equity", 100000)
        self.monitor = AgentPerformanceMonitor(equity, max_drawdown=0.2, min_sharpe=0.2)
        try:
            step = 0
            prev_equity = equity
            while n_steps is None or step < n_steps:
                logger.info(f"Live step {step} | {datetime.now().isoformat()}")
                keep_going = self._tick()
                acc = self.connector.get_account_info()
                new_equity = acc.get("equity", prev_equity)
                self.monitor.update(new_equity)
                self._heartbeat(step, status="running", equity=new_equity)
                if self.monitor.stop_trading or not keep_going:
                    logger.warning(f"TRADING HALTED (step={step}) for safety or health reasons.")
                    self._heartbeat(step, status="halted", equity=new_equity)
                    break
                prev_equity = new_equity
                step += 1
                time.sleep(self.interval_sec)
        except Exception as e:
            logger.error(f"Exception in MT5 live loop: {e}", exc_info=1)
            self._heartbeat(step, status="crashed")
        finally:
            self.connector.shutdown()
            self.save_log()

    def _tick(self):
        # Returns False if fatal
        prices = self.connector.get_prices()
        batch = self._assemble_agent_batch(prices)
        try:
            agent_out = self.agent(batch)
        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=1)
            return False
        signals = self._interpret_model_output(agent_out, prices)
        positions = self.connector.get_positions()
        trade_actions = []
        for symbol in self.symbols:
            target = signals.get(symbol, 0)
            out_val = agent_out.get(symbol, {}).get("price_mean", prices[symbol])
            self.monitor.agent_output_check(out_val)
            pos = positions.get(symbol)
            # (Same logic as before, closing conflicting positions, etc)
            if pos:
                pos_side = 1 if pos['type'] == 0 else -1
                if pos_side != target and target != 0:
                    self.connector.close_position(symbol)
            if not pos and target != 0:
                result = self.connector.place_order(symbol, target)
                trade_actions.append({
                    "time": datetime.now().isoformat(),
                    "symbol": symbol,
                    "action": "BUY" if target == 1 else "SELL",
                    "price": prices[symbol],
                    "order_result": str(result),
                })
        if trade_actions:
            self.trade_log.extend(trade_actions)
        # Health check hook: If agent outputs persistently NaN, return False to break loop
        if self.monitor.stop_trading:
            return False
        return True

    def _assemble_agent_batch(self, prices):
        batch = {}
        for symbol in self.symbols:
            x = np.array([[prices[symbol]] * self.config.feature_size])
            try:
                import torch
                batch[symbol] = {'x': torch.FloatTensor(x).to(self.config.device)}
            except ImportError:
                batch[symbol] = {'x': x}
        return batch

    def _interpret_model_output(self, agent_out, prices):
        signals = {}
        for symbol in self.symbols:
            out = agent_out.get(symbol, {})
            pred = out.get("price_mean", prices[symbol])
            refpx = prices[symbol]
            try:
                pf = float(pred)
            except Exception:
                pf = refpx
            if pf > refpx * 1.001:
                signals[symbol] = 1
            elif pf < refpx * 0.999:
                signals[symbol] = -1
            else:
                signals[symbol] = 0
        return signals

    def _heartbeat(self, step, status="running", equity=None):
        msg = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "status": status,
            "equity": equity,
        }
        logger.info(f"HEARTBEAT: {msg}")
        self.heartbeat_log.append(msg)
        # Optionally: Send to external dashboard, email, or webhook here
        # self._send_alert(msg)

    def save_log(self):
        path = self.run_dir / "trade_log.csv"
        import csv
        if self.trade_log:
            with open(path, "w", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.trade_log[0].keys())
                writer.writeheader()
                writer.writerows(self.trade_log)
            logger.info(f"Trade log saved: {path}")
        heartbeat_path = self.run_dir / "heartbeat_log.json"
        import json
        with open(heartbeat_path, "w") as f:
            json.dump(self.heartbeat_log, f, indent=2)
        logger.info(f"Heartbeat log saved: {heartbeat_path}")

    # def _send_alert(self, msg):
    #     """Optionally implement webhook/email alerting here!"""
    #     pass

# Use as before, passing config/login, and optionally plug in alert hooks and extra risk logic.
