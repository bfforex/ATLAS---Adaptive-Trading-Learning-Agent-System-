"""
ATLAS V2: XMGlobal MT5 Live Trading Module with System Monitor Integration
=========================================================================
Production live trading loop using MT5Connector and AtlasSystemMonitor.
"""

import MetaTrader5 as mt5
import logging
import time
from datetime import datetime
from pathlib import Path
import sys
import numpy as np

from atlas_v2_monitor import AtlasSystemMonitor, AtlasMonitorConfig  # <-- import monitor!

logger = logging.getLogger("atlas_v2_live_mt5")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s'))
    logger.addHandler(handler)

class MT5Connector:
    def __init__(self, login: int, password: str, server: str, symbols, lot_size=0.1):
        self.login = login
        self.password = password
        self.server = server
        self.symbols = symbols
        self.lot_size = lot_size
        self.connected = False

    def connect(self):
        if mt5.initialize(login=self.login, password=self.password, server=self.server):
            logger.info("MT5 initialized and logged in.")
            self.connected = True
        else:
            logger.error(f"Initialization failed: {mt5.last_error()}")
            sys.exit(1)
        for symbol in self.symbols:
            if not mt5.symbol_select(symbol, True):
                logger.warning(f"Couldn't activate symbol {symbol}. Check MT5 market watch.")

    def shutdown(self):
        logger.info("Deinitializing MT5.")
        mt5.shutdown()

    def get_prices(self):
        prices = {}
        for symbol in self.symbols:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                prices[symbol] = tick.ask
            else:
                logger.warning(f"No tick for {symbol}")
        return prices

    def get_account_info(self):
        acc = mt5.account_info()
        if acc:
            return acc._asdict()
        logger.error("Could not retrieve account info.")
        return {}

    def get_positions(self):
        pos_list = mt5.positions_get()
        if pos_list is None:
            logger.error("Error retrieving positions")
            return {}
        positions = {}
        for pos in pos_list:
            positions[pos.symbol] = {
                'ticket': pos.ticket,
                'type': pos.type,  # 0=buy, 1=sell
                'volume': pos.volume,
                'price_open': pos.price_open
            }
        return positions

    def close_position(self, symbol):
        pos = self.get_positions().get(symbol)
        if not pos:
            return
        close_type = 1 if pos['type'] == 0 else 0
        price = mt5.symbol_info_tick(symbol).bid if close_type == 1 else mt5.symbol_info_tick(symbol).ask
        order_dict = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos['volume'],
            "type": close_type,
            "position": pos['ticket'],
            "price": price,
            "deviation": 10,
            "magic": 42,
            "comment": "AtlasV2 close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(order_dict)
        logger.info(f"Closed position for {symbol}, result: {result}")

    def place_order(self, symbol, side: int):
        order_type = mt5.ORDER_TYPE_BUY if side == 1 else mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).ask if side == 1 else mt5.symbol_info_tick(symbol).bid
        order_dict = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": self.lot_size,
            "type": order_type,
            "price": price,
            "deviation": 10,
            "magic": 42,
            "comment": "AtlasV2 live",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(order_dict)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Order placed: {symbol} {'BUY' if side == 1 else 'SELL'} {self.lot_size} lot")
        else:
            logger.error(f"Order failed ({result.retcode}): {result}")
        return result

class LiveTradingMT5Loop:
    def __init__(self, agent, config, connector: MT5Connector, run_dir=None, interval_sec: int = 60, monitor_config=None, alert_callback=None):
        self.agent = agent
        self.config = config
        self.connector = connector
        self.symbols = connector.symbols
        self.run_dir = Path(run_dir or (Path("mt5_live_logs") / datetime.now().strftime("%Y%m%d_%H%M%S")))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.interval_sec = interval_sec
        self.trade_log = []
        self.heartbeat_log = []
        # --- Monitor setup ---
        acc_info = self.connector.get_account_info()
        initial_equity = acc_info.get("equity", 100000)
        self.monitor = AtlasSystemMonitor(
            initial_equity=initial_equity,
            symbols=self.symbols,
            config=monitor_config,
            alert_callback=alert_callback,
        )

    def run(self, n_steps: int = None):
        self.connector.connect()
        try:
            step = 0
            prev_equity = self.connector.get_account_info().get("equity", 100000)
            while n_steps is None or step < n_steps:
                logger.info(f"Live step {step} | {datetime.now().isoformat()}")
                keep_going = self._tick(step)
                acc_info = self.connector.get_account_info()
                new_equity = acc_info.get("equity", prev_equity)
                self.monitor.update_equity(new_equity)
                self.monitor.heartbeat()
                if self.monitor.should_halt() or not keep_going:
                    logger.warning(f"TRADING HALTED (step={step}) for monitor safety or agent health.")
                    self._flatten_all_positions()
                    self.monitor.heartbeat()
                    break
                prev_equity = new_equity
                step += 1
                time.sleep(self.interval_sec)
        except Exception as e:
            logger.error(f"Exception in MT5 live loop: {e}", exc_info=1)
            self.monitor.halt(f"Exception in live loop: {e}")
            self.monitor.heartbeat()
        finally:
            self.connector.shutdown()
            self.save_log()

    def _tick(self, step):
        prices = self.connector.get_prices()
        # --- Monitor price integrity ---
        self.monitor.monitor_market_data(prices)
        batch = self._assemble_agent_batch(prices)
        try:
            agent_out = self.agent(batch)
        except Exception as e:
            logger.error(f"Agent error: {e}", exc_info=1)
            self.monitor.halt(f"Agent error: {e}")
            return False
        signals = self._interpret_model_output(agent_out, prices)
        positions = self.connector.get_positions()
        # --- Monitor agent outputs & positions ---
        # agent_out: dict {symbol: {'price_mean': ...}}
        agent_outputs = {sym: agent_out.get(sym, {}).get("price_mean") for sym in self.symbols}
        self.monitor.update_agent_outputs(agent_outputs)
        self.monitor.update_positions(positions, signals)
        trade_actions = []
        for symbol in self.symbols:
            target = signals.get(symbol, 0)
            out_val = agent_outputs.get(symbol, prices[symbol])
            pos = positions.get(symbol)
            min_hold_time = getattr(self.monitor.config, "min_hold_time", 2)
            hold_count = self.monitor.hold_counts.get(symbol, 0)
            if pos:
                pos_side = 1 if pos['type'] == 0 else -1
                if pos_side != target and target != 0:
                    if hold_count >= min_hold_time:
                        self.connector.close_position(symbol)
                    else:
                        logger.info(f"[Monitor] Suppressed close for {symbol} (hold_count={hold_count})")
            if not pos and target != 0:
                result = self.connector.place_order(symbol, target)
                if getattr(result, "retcode", None) != mt5.TRADE_RETCODE_DONE:
                    self.monitor.report_order_rejection()
                else:
                    self.monitor.reset_order_reject()
                trade_actions.append({
                    "time": datetime.now().isoformat(),
                    "symbol": symbol,
                    "action": "BUY" if target == 1 else "SELL",
                    "price": prices[symbol],
                    "order_result": str(result),
                })
        if trade_actions:
            self.trade_log.extend(trade_actions)
        # Final health check
        if self.monitor.should_halt():
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
            try:
                pf = float(pred)
            except Exception:
                pf = prices[symbol]
            if pf > prices[symbol] * 1.001:
                signals[symbol] = 1
            elif pf < prices[symbol] * 0.999:
                signals[symbol] = -1
            else:
                signals[symbol] = 0
        return signals

    def _flatten_all_positions(self):
        logger.warning("[Monitor] Flattening all positions for safety.")
        pos_list = self.connector.get_positions()
        for symbol, pos in pos_list.items():
            self.connector.close_position(symbol)

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
            json.dump(self.monitor.state, f, indent=2)
        logger.info(f"Monitor state log saved: {heartbeat_path}")

# Usage in main.py:
# connector = MT5Connector(login, password, server, symbols, lot_size)
# monitor_config = AtlasMonitorConfig() # Set custom thresholds if desired
# live_loop = LiveTradingMT5Loop(agent, config, connector, run_dir, interval_sec, monitor_config, alert_callback)
# live_loop.run(n_steps)
