"""
ATLAS V2: Backtesting Module
============================
Efficient, reproducible, event-safe backtesting engine.
"""
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
from datetime import datetime
import logging
import json
import csv

logger = logging.getLogger("atlas_v2_backtest")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s'))
    logger.addHandler(handler)

class Backtester:
    def __init__(
        self,
        agent,
        config,
        market_data_provider,
        result_dir='backtest_logs',
        name: Optional[str]=None,
        slippage: float=0.0,
        commission: float=0.0,
        allow_shorting: bool=True
    ):
        self.agent = agent
        self.config = config
        self.market = market_data_provider
        self.slippage = slippage
        self.commission = commission
        self.allow_shorting = allow_shorting
        self.name = name or f'backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.result_dir = Path(result_dir) / self.name
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.init_state()

    def init_state(self):
        self.balance = getattr(self.config, "initial_balance", 100_000.0)
        self.initial_balance = self.balance
        self.portfolio = {s: {'position': 0.0, 'entry': 0.0, 'pnl': 0.0, 'size': 0.0} for s in self.config.symbols}
        self.equity_curve = [self.balance]
        self.trades_history = []
        self.risk_history = []
        self.step = 0
        self.stopped = False

    def run(self, start=0, end=None):
        if end is None: end = len(self.market)
        logger.info(f"Running backtest: {self.name} from {start} to {end}")
        for step in range(start, end):
            market_state = self.market.get_step(step)
            batch = self._assemble_agent_batch(market_state, step)
            try:
                agent_out = self.agent(batch)
            except Exception as e:
                logger.error(f"Agent call failed at backtest step {step}: {e}")
                break
            signals, sizes = self._generate_signals_and_positions(market_state, agent_out)
            self._execute_trades(signals, sizes, market_state)
            self._update_performance(market_state)
            self.step += 1
        self._final_report()
        self._persist_logs()

    def _assemble_agent_batch(self, market_state, step):
        # Serve agent only present and past data (no lookahead)
        batch = {}
        for symbol in self.config.symbols:
            features_arr = []
            for i in range(max(0, step - self.config.seq_length + 1), step + 1):
                features_arr.append(market_state[symbol]['features'] if i == step else self.market.get_step(i)[symbol]['features'])
            x = np.stack(features_arr, axis=0)
            try:
                import torch
                batch[symbol] = {'x': torch.FloatTensor(x).unsqueeze(0).to(self.config.device)}
            except ImportError:
                batch[symbol] = {'x': x[np.newaxis, ...]}
        return batch

    def _generate_signals_and_positions(self, market_state, agent_out):
        signals = {}
        sizes = {}
        for symbol in self.config.symbols:
            out = agent_out.get(symbol, {})
            pred = out.get("price_mean")
            if hasattr(pred, "detach"):
                pred = pred.cpu().numpy().item()
            else:
                pred = np.array(pred).item()
            price = market_state[symbol]['price']
            signal = 1 if pred > price * 1.001 else -1 if (self.allow_shorting and pred < price * 0.999) else 0
            signals[symbol] = signal
            sizes[symbol] = getattr(self.config, 'trade_fraction', 0.10) * self.balance if signal != 0 else 0
        return signals, sizes

    def _execute_trades(self, signals, sizes, market_state):
        for symbol in self.config.symbols:
            signal = signals[symbol]
            position = self.portfolio[symbol]['position']
            entry_price = self.portfolio[symbol]['entry']
            price = market_state[symbol]['price']
            size = sizes[symbol]
            commission = abs(size) * self.commission
            slip = abs(price * self.slippage)
            trade_price = price + slip if signal == 1 else price - slip if signal == -1 else price
            pnl = 0.0
            trade = None
            if signal != 0:
                if (signal == 1 and position < 0) or (signal == -1 and position > 0):
                    gained = (trade_price - entry_price) * abs(self.portfolio[symbol]['size']) * np.sign(position)
                    pnl += gained - commission
                    self.balance += pnl
                    self.portfolio[symbol]['pnl'] += pnl
                    self.portfolio[symbol]['position'] = 0
                    self.portfolio[symbol]['size'] = 0
                    trade = dict(
                        symbol=symbol, action="LIQUIDATE", price=trade_price, pnl=gained,
                        size=self.portfolio[symbol]['size'], fee=commission, step=self.step,
                        time=datetime.now().isoformat(),
                    )
                self.portfolio[symbol]['position'] = signal
                self.portfolio[symbol]['entry'] = trade_price
                self.portfolio[symbol]['size'] = size
                self.balance -= commission
                trade = dict(
                    symbol=symbol, action="BUY" if signal == 1 else "SELL", price=trade_price, pnl=0,
                    size=size, fee=commission, step=self.step, time=datetime.now().isoformat(),
                )
            if trade is not None:
                self.trades_history.append(trade)

    def _update_performance(self, market_state):
        equity = self.balance
        for symbol in self.config.symbols:
            pos = self.portfolio[symbol]
            price = market_state[symbol]['price']
            if pos['position'] != 0 and pos['size'] > 0:
                open_pnl = (price - pos['entry']) * pos['size'] * np.sign(pos['position'])
                equity += open_pnl
        self.equity_curve.append(equity)

    def _final_report(self):
        logger.info(f"==== BACKTEST COMPLETE ({self.name}) ====")
        final_equity = self.equity_curve[-1]
        total_return = (final_equity - self.initial_balance) / self.initial_balance * 100
        logger.info(f"Initial Balance: ${self.initial_balance:,.2f}")
        logger.info(f"Final Equity:    ${final_equity:,.2f}")
        logger.info(f"Total Return:    {total_return:+.2f}%")
        logger.info(f"Total Trades:    {len(self.trades_history)}")

    def _persist_logs(self):
        # Save trades and equity to disk
        trade_path = self.result_dir / "trades.csv"
        if self.trades_history:
            with open(trade_path, "w", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.trades_history[0].keys())
                writer.writeheader()
                writer.writerows(self.trades_history)
        equity_path = self.result_dir / "equity_curve.csv"
        with open(equity_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["step", "equity"])
            for i, eq in enumerate(self.equity_curve):
                writer.writerow([i, eq])
        logger.info(f"Backtest logs written to {self.result_dir}")
