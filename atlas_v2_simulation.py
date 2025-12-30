"""
ATLAS V2: Advanced Simulation & Live Trading Module
==================================================
Supports position sizing, leverage, fees, risk limits, multi-market, and
persistent logs for post-trade analysis/dashboarding.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Callable
from datetime import datetime
import numpy as np
import logging
import csv
import json

logger = logging.getLogger("atlas_v2_simulation")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class MarketDataProvider:
    def __init__(self, symbols, data: Dict[str, np.ndarray]):
        self.symbols = symbols
        self.data = data
        self.length = min(len(arr) for arr in data.values())

    def __len__(self):
        return self.length

    def get_step(self, idx: int) -> Dict[str, Dict[str, Any]]:
        state = {}
        for sym in self.symbols:
            arr = self.data[sym]
            state[sym] = {
                'features': arr[idx],
                'price': arr[idx, 0]
            }
        return state

class LiveTradingSimulator:
    def __init__(
        self,
        agent,
        config,
        market_data_provider: MarketDataProvider,
        results_dir: str = 'simulation_logs',
        name: Optional[str] = None,
    ):
        self.agent = agent
        self.config = config
        self.market = market_data_provider
        self.name = name or f'sim_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.results_dir = Path(results_dir) / self.name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.symbols = self.config.symbols
        self.init_state()

    def init_state(self):
        self.balance = getattr(self.config, "initial_balance", 100_000.0)
        self.initial_balance = self.balance
        self.portfolio = {s: {'position': 0.0, 'entry': 0.0, 'pnl': 0.0, 'leverage': 1.0, 'size': 0.0} for s in self.symbols}
        self.equity_curve = [self.balance]
        self.trades_history = []
        self.risk_history = []
        self.step = 0
        self.stopped = False

    def run(self, n_steps: Optional[int] = None):
        logger.info(f"Running trading simulation: {self.name}")
        n_steps = n_steps or len(self.market)
        for step in range(n_steps):
            if self.stopped:
                break
            market_state = self.market.get_step(step)
            batch = self._assemble_agent_batch(market_state)
            try:
                with np.errstate(all='ignore'):
                    agent_out = self.agent(batch)
            except Exception as e:
                logger.error(f"Agent call failed at step {step}: {e}")
                break
            signals, sizes, leverages = self._generate_signals_and_positions(market_state, agent_out)
            self._execute_trades(signals, sizes, leverages, market_state)
            self._update_performance(market_state)
            self._log_step(step, market_state, signals, sizes, leverages)
            self._risk_check(step)
            self.step += 1
        self._final_report()
        self._persist_rich_logs()

    def _assemble_agent_batch(self, market_state):
        batch = {}
        for symbol in self.symbols:
            x = np.expand_dims(market_state[symbol]['features'], 0)
            try:
                import torch
                batch[symbol] = {'x': torch.FloatTensor(x).to(self.config.device)}
            except ImportError:
                batch[symbol] = {'x': x}
        return batch

    def _generate_signals_and_positions(self, market_state, agent_out):
        # Agent output could deliver not just price_mean but (signal, size, leverage)!
        signals = {}
        sizes = {}
        leverages = {}
        for symbol in self.symbols:
            out = agent_out.get(symbol, {})
            pred = out.get("price_mean")
            if hasattr(pred, "detach"):
                pred = pred.cpu().numpy().item()
            else:
                pred = np.array(pred).item()
            price = market_state[symbol]['price']
            signal = 1 if pred > price * 1.001 else -1 if pred < price * 0.999 else 0
            signals[symbol] = signal

            # Example: Constant fraction of equity (customize as needed)
            default_frac = getattr(self.config, 'trade_fraction', 0.10)
            max_leverage = getattr(self.config, 'max_leverage', 3.0)
            size = default_frac * self.balance
            leverage = max_leverage if hasattr(self.config, 'use_max_leverage') and self.config.use_max_leverage else 1.0
            sizes[symbol] = size
            leverages[symbol] = leverage
        return signals, sizes, leverages

    def _execute_trades(self, signals, sizes, leverages, market_state):
        fees_rate = getattr(self.config, 'fee_rate', 0.0005)
        for symbol in self.symbols:
            signal = signals[symbol]
            position = self.portfolio[symbol]['position']
            entry_price = self.portfolio[symbol]['entry']
            price = market_state[symbol]['price']
            size = sizes[symbol]
            leverage = leverages[symbol]
            pnl = 0.0
            trade = None

            if signal != 0 and np.abs(size) > 0:
                # Close opposite, open new
                if (signal == 1 and position < 0) or (signal == -1 and position > 0):
                    gained = (price - entry_price) * np.abs(self.portfolio[symbol]['size']) * self.portfolio[symbol]['leverage'] * np.sign(position)
                    fee = np.abs(self.portfolio[symbol]['size']) * fees_rate
                    pnl += gained - fee
                    self.balance += pnl
                    self.portfolio[symbol]['pnl'] += pnl
                    self.portfolio[symbol]['position'] = 0
                    self.portfolio[symbol]['size'] = 0
                    self.portfolio[symbol]['leverage'] = 1.0
                    trade = dict(
                        symbol=symbol, action="LIQUIDATE", price=price, pnl=gained, size=self.portfolio[symbol]['size'],
                        leverage=self.portfolio[symbol]['leverage'], fee=fee, step=self.step, time=datetime.now().isoformat(),
                    )
                # Open new position in direction of signal
                self.portfolio[symbol]['position'] = signal
                self.portfolio[symbol]['entry'] = price
                self.portfolio[symbol]['size'] = size
                self.portfolio[symbol]['leverage'] = leverage
                opening_fee = size * fees_rate
                self.balance -= opening_fee
                trade = dict(
                    symbol=symbol, action="BUY" if signal == 1 else "SELL", price=price, pnl=0,
                    size=size, leverage=leverage, fee=opening_fee, step=self.step, time=datetime.now().isoformat(),
                )
            if trade is not None:
                self.trades_history.append(trade)
                logger.debug(f"Trade: {trade}")

    def _update_performance(self, market_state):
        equity = self.balance
        for symbol in self.symbols:
            pos = self.portfolio[symbol]
            price = market_state[symbol]['price']
            if pos['position'] != 0 and pos['size'] > 0:
                open_pnl = (price - pos['entry']) * pos['size'] * pos['leverage'] * np.sign(pos['position'])
                equity += open_pnl
        self.equity_curve.append(equity)

        # Log rich risk metrics
        ret = (self.equity_curve[-1] - self.equity_curve[-2]) / (self.equity_curve[-2] + 1e-12) if len(self.equity_curve)>1 else 0
        self.risk_history.append({
            "step": self.step,
            "equity": equity,
            "return": ret,
            "drawdown": self._compute_drawdown()
        })

    def _risk_check(self, step):
        # Stop out criteria
        equity = self.equity_curve[-1]
        max_drawdown = getattr(self.config, 'max_drawdown', 0.2)
        if self._compute_drawdown() < -abs(max_drawdown):
            logger.warning(f"Max drawdown reached at step {step}, stopping out!")
            self._liquidate_all(step)
            self.stopped = True

    def _compute_drawdown(self):
        values = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(values)
        drawdowns = (values - running_max) / (running_max + 1e-12)
        return float(drawdowns[-1])

    def _liquidate_all(self, step):
        # Force close all positions
        for symbol in self.symbols:
            pos = self.portfolio[symbol]
            if pos['position'] != 0 and pos['size'] > 0:
                # Settle at last price
                price = self.market.get_step(step)[symbol]['price']
                gained = (price - pos['entry']) * pos['size'] * pos['leverage'] * np.sign(pos['position'])
                fee = np.abs(pos['size']) * getattr(self.config, 'fee_rate', 0.0005)
                self.balance += gained - fee
                self.portfolio[symbol] = {'position': 0, 'entry': 0, 'pnl': pos['pnl']+gained-fee, 'leverage': 1.0, 'size': 0.0}
                liq_trade = dict(
                    symbol=symbol, action="FORCE_LIQUIDATE", price=price, pnl=gained,
                    size=pos['size'], leverage=pos['leverage'],
                    fee=fee, step=step, time=datetime.now().isoformat()
                )
                self.trades_history.append(liq_trade)

    def _log_step(self, step, market_state, signals, sizes, leverages):
        # Custom per-step logging (could persist incremental logs if desired)
        pass

    def _final_report(self):
        logger.info(f"==== SIMULATION COMPLETE ({self.name}) ====")
        final_equity = self.equity_curve[-1]
        total_return = (final_equity - self.initial_balance) / self.initial_balance * 100
        logger.info(f"Initial Balance: ${self.initial_balance:,.2f}")
        logger.info(f"Final Equity:    ${final_equity:,.2f}")
        logger.info(f"Total Return:    {total_return:+.2f}%")
        logger.info(f"Total Trades:    {len(self.trades_history)}")

        returns = np.diff(self.equity_curve) / (np.array(self.equity_curve)[:-1] + 1e-12)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-12) * np.sqrt(252) if len(returns)>1 else 0.0
        if len(returns) > 1:
            cumulative = (1 + returns).cumprod()
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_dd = np.min(drawdown)
        else:
            max_dd = 0.0
        logger.info(f"Sharpe Ratio:    {sharpe:.3f}")
        logger.info(f"Max Drawdown:    {max_dd:.2%}")

    def _persist_rich_logs(self):
        # TRADE LOG (CSV)
        trade_path = self.results_dir / "trades.csv"
        if self.trades_history:
            with open(trade_path, "w", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.trades_history[0].keys())
                writer.writeheader()
                writer.writerows(self.trades_history)
        # EQUITY CURVE (CSV)
        equity_path = self.results_dir / "equity_curve.csv"
        with open(equity_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["step", "equity"])
            for i, eq in enumerate(self.equity_curve):
                writer.writerow([i, eq])
        # RISK HISTORY (JSON)
        risk_path = self.results_dir / "risk_history.json"
        with open(risk_path, "w") as f:
            json.dump(self.risk_history, f, indent=2)
        logger.info(f"Simulation logs written to {self.results_dir}")

# Example for "multi-market simulation": run parallel instances or extend MarketDataProvider
# For a practical multi-agent/portfolio scenario, you can:
#    - Instantiate separate LiveTradingSimulator objects (one per market/agent/config)
#    - OR extend MarketDataProvider to handle multiplexed data and adapt all .get_step and .run() logic.

if __name__ == "__main__":
    logger.info("Demo: ATLAS V2 Advanced LiveTradingSimulator (pluggable agent, data, with full logs)")
