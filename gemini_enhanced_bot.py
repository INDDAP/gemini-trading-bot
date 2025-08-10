# Final Bot Code - Web App Version for AWS using Binance API

import asyncio
import os
import threading
from flask import Flask
import aiohttp
import json
import time
import hmac
import base64
import pandas as pd
import pandas_ta as ta

# --- Flask Web App Setup ---
app = Flask(__name__)

@app.route('/')
def home():
    """This is the webpage the uptime service will visit."""
    return "Bot is alive and running!"

def run_flask_app():
    """Runs the Flask app on host 0.0.0.0."""
    app.run(host='0.0.0.0', port=8080)


# --- Bot Logic ---
async def run_bot_logic():
    """Contains the entire trading bot logic."""
    
    # ----- Configuration (Reads from ~/.profile on your server) -----
    BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY")
    BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET")
    IS_SANDBOX_str = os.environ.get("IS_SANDBOX", "true").lower()
    IS_SANDBOX = IS_SANDBOX_str in ["true", "1"]
    
    SYMBOL = os.environ.get("SYMBOL", "BTCUSDT")
    TIMEFRAME = os.environ.get("TIMEFRAME", "5m")
    START_BALANCE = float(os.environ.get("START_BALANCE", "1000.0"))
    RISK_PER_TRADE_PERCENT = float(os.environ.get("RISK_PER_TRADE_PERCENT", "1.0"))

    # ----- Strategy Parameters -----
    EMA_FAST = 12
    EMA_SLOW = 26
    ADX_THRESHOLD = 25
    ATR_MULTIPLIER = 2.0
    TAKE_PROFIT_RATIO = 1.5

    # ----- Binance API Connector -----
    class BinanceAPI:
        def __init__(self, api_key, api_secret, is_sandbox=True):
            self.base_url = "https://testnet.binance.vision/api/v3" if is_sandbox else "https://api.binance.com/api/v3"
            self.api_key = api_key
            self.api_secret = api_secret.encode('utf-8') if api_secret else None
            self.session = None

        async def _get_session(self):
            if self.session is None or self.session.closed:
                self.session = aiohttp.ClientSession()
            return self.session

        async def get_candles(self, symbol, timeframe):
            endpoint = "/klines"
            url = self.base_url + endpoint
            params = {'symbol': symbol, 'interval': timeframe, 'limit': 100}

            try:
                session = await self._get_session()
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    if not isinstance(data, list) or len(data) == 0:
                        print(f"❌ Error fetching candles, received: {data}")
                        return None
                    
                    df = pd.DataFrame(data, columns=[
                        'time', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                    ])
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col])
                    df['time'] = pd.to_datetime(df['time'], unit='ms')
                    df.set_index('time', inplace=True)
                    return df
            except Exception as e:
                print(f"❌ Error fetching candles: {e}")
                return None

        async def close(self):
            if self.session:
                await self.session.close()

    # ----- Strategy & Indicators -----
    def add_indicators(df):
        df.ta.ema(length=EMA_FAST, append=True)
        df.ta.ema(length=EMA_SLOW, append=True)
        df.ta.adx(append=True)
        df.ta.atr(append=True)
        return df

    def check_buy_signal(df):
        if len(df) < EMA_SLOW:
            return False
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        ema_cross_up = prev[f'EMA_{EMA_FAST}'] < prev[f'EMA_{EMA_SLOW}'] and latest[f'EMA_{FAST}'] > latest[f'EMA_{SLOW}']
        strong_trend = latest[f'ADX_{14}'] > ADX_THRESHOLD
        if ema_cross_up and strong_trend:
            print(f"\n📈 Buy Signal Found! EMA Cross: {ema_cross_up}, ADX: {latest[f'ADX_{14}']:.2f}")
            return True
        return False

    # ----- Trade Manager -----
    class TradeManager:
        def __init__(self, initial_balance, api):
            self.balance = initial_balance
            self.api = api
            self.position_open = False
            self.entry_price = 0.0
            self.stop_loss_price = 0.0
            self.take_profit_price = 0.0
            self.qty = 0.0

        def calculate_position_size(self, entry_price, stop_loss_price):
            risk_amount_per_trade = self.balance * (RISK_PER_TRADE_PERCENT / 100)
            risk_per_unit = entry_price - stop_loss_price
            if risk_per_unit <= 0: return 0
            qty = risk_amount_per_trade / risk_per_unit
            return round(qty, 6)

        async def open_position(self, df):
            latest = df.iloc[-1]
            entry_price = latest['close']
            atr = latest[f'ATRr_{14}']
            self.stop_loss_price = entry_price - (atr * ATR_MULTIPLIER)
            self.take_profit_price = entry_price + ((entry_price - self.stop_loss_price) * TAKE_PROFIT_RATIO)
            self.qty = self.calculate_position_size(entry_price, self.stop_loss_price)

            if self.qty <= 0 or (self.qty * entry_price > self.balance):
                print("\n⚠️ Insufficient balance or invalid quantity. Cannot open position.")
                return

            self.entry_price = entry_price
            self.position_open = True
            print(f"\n✅ Position Opened: Qty={self.qty:.6f} @ ${self.entry_price:.2f}")
            print(f"   - Stop Loss: ${self.stop_loss_price:.2f}")
            print(f"   - Take Profit: ${self.take_profit_price:.2f}")

        def check_exit(self, current_price):
            exit_reason = None
            if current_price <= self.stop_loss_price:
                exit_reason = "Stop-Loss"
            elif current_price >= self.take_profit_price:
                exit_reason = "Take-Profit"
            if exit_reason:
                profit = (current_price - self.entry_price) * self.qty
                self.balance += profit
                pnl = ((current_price - self.entry_price) / self.entry_price) * 100
                print(f"\n❌ Position Closed ({exit_reason}): Exited @ ${current_price:.2f}, PnL: ${profit:.2f} ({pnl:.2f}%)")
                print(f"   - New Balance: ${self.balance:.2f}")
                self._reset_position()
        
        def _reset_position(self):
            self.position_open = False
            self.entry_price = 0.0
            self.stop_loss_price = 0.0
            self.take_profit_price = 0.0
            self.qty = 0.0

    # ----- Main Bot Loop -----
    async def main_loop():
        api = BinanceAPI(BINANCE_API_KEY, BINANCE_API_SECRET, IS_SANDBOX)
        trade_manager = TradeManager(initial_balance=START_BALANCE, api=api)
        
        print(f"🚀 Bot logic started on {SYMBOL} ({TIMEFRAME}) with ${START_BALANCE:.2f} USD")
        print(f"   - Mode: {'Sandbox (Paper Trading)' if IS_SANDBOX else 'Live Trading'}")
        print("-" * 30)

        if not BINANCE_API_KEY or not BINANCE_API_SECRET:
            print("🛑 Critical Error: API keys not found. Shutting down.")
            return

        try:
            while True:
                candles_df = await api.get_candles(SYMBOL, TIMEFRAME)
                if candles_df is None or candles_df.empty:
                    print("\nCould not fetch candle data. Retrying in 60s...")
                    await asyncio.sleep(60)
                    continue

                candles_df = add_indicators(candles_df)
                current_price = candles_df.iloc[-1]['close']
                
                print(f"\r📊 Checking {time.strftime('%Y-%m-%d %H:%M:%S')}, Price: ${current_price:.2f}, ADX: {candles_df.iloc[-1][f'ADX_{14}']:.2f}  ", end="", flush=True)

                if not trade_manager.position_open:
                    if check_buy_signal(candles_df):
                        await trade_manager.open_position(candles_df)
                else:
                    trade_manager.check_exit(current_price)
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            print("\n🛑 Bot shutting down.")
        finally:
            await api.close()

    await main_loop()


# --- Main Entry Point ---
if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.daemon = True
    flask_thread.start()
    try:
        asyncio.run(run_bot_logic())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped manually.")
