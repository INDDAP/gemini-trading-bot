# gemini_enhanced_bot.py

import asyncio
import aiohttp
import json
import time
import hmac
import base64
import pandas as pd
import pandas_ta as ta
import os # Import the os library

# ----- Configuration (Set these in the Render Environment Variables) -----
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_API_SECRET = os.environ.get("GEMINI_API_SECRET")
# The IS_SANDBOX variable is read as a string. "true" (default) or "false".
IS_SANDBOX_str = os.environ.get("IS_SANDBOX", "true").lower()
IS_SANDBOX = IS_SANDBOX_str in ["true", "1"]

# You can also make these strategy parameters environment variables for easy tuning
SYMBOL = os.environ.get("SYMBOL", "BTCUSD")
TIMEFRAME = os.environ.get("TIMEFRAME", "5m")
START_BALANCE = float(os.environ.get("START_BALANCE", "1000.0"))
RISK_PER_TRADE_PERCENT = float(os.environ.get("RISK_PER_TRADE_PERCENT", "1.0"))

# ----- Strategy Parameters -----
EMA_FAST = 12
EMA_SLOW = 26
ADX_THRESHOLD = 25
ATR_MULTIPLIER = 2.0 # Stop-loss will be 2 * ATR below entry
TAKE_PROFIT_RATIO = 1.5 # Reward:Risk ratio of 1.5:1

# ----- Gemini API Connector -----
class GeminiAPI:
    def __init__(self, api_key, api_secret, is_sandbox=True):
        self.base_url = "https://api.sandbox.gemini.com" if is_sandbox else "https://api.gemini.com"
        self.api_key = api_key
        # Ensure secret is encoded, especially important if it's None initially
        self.api_secret = api_secret.encode('utf-8') if api_secret else None
        self.session = None

    async def _get_session(self):
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def _send_request(self, method, endpoint, payload=None):
        if not self.api_key or not self.api_secret:
            print("❌ Error: API Key or Secret is not set. Check your environment variables.")
            return None
            
        session = await self._get_session()
        url = self.base_url + endpoint
        
        if payload is None:
            payload = {}
        payload['request'] = endpoint
        payload['nonce'] = int(time.time() * 1000)
        
        encoded_payload = json.dumps(payload).encode('utf-8')
        b64 = base64.b64encode(encoded_payload)
        signature = hmac.new(self.api_secret, b64, 'sha384').hexdigest()

        headers = {
            'Content-Type': 'text/plain',
            'X-GEMINI-APIKEY': self.api_key,
            'X-GEMINI-PAYLOAD': b64.decode('utf-8'),
            'X-GEMINI-SIGNATURE': signature
        }

        try:
            async with session.post(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    print(f"❌ Error: {response.status} - {await response.text()}")
                    return None
        except aiohttp.ClientError as e:
            print(f"❌ Network Error: {e}")
            return None

    async def get_candles(self, symbol, timeframe):
        endpoint = f"/v2/candles/{symbol}/{timeframe}"
        url = self.base_url + endpoint
        try:
            session = await self._get_session()
            async with session.get(url) as response:
                data = await response.json()
                df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
                df['time'] = pd.to_datetime(df['time'], unit='ms')
                df.set_index('time', inplace=True)
                return df.iloc[::-1] # Reverse to have latest candle at the end
        except Exception as e:
            print(f"❌ Error fetching candles: {e}")
            return None

    async def place_order(self, symbol, amount, price, side):
        endpoint = "/v1/order/new"
        payload = {
            "symbol": symbol,
            "amount": str(amount),
            "price": str(price),
            "side": side,
            "type": "exchange limit",
        }
        print(f"➡️  Placing {side} order: {amount} {symbol} at ${price}")
        return await self._send_request('POST', endpoint, payload)

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

    # EMA Crossover: fast EMA crosses above slow EMA
    ema_cross_up = prev[f'EMA_{EMA_FAST}'] < prev[f'EMA_{EMA_SLOW}'] and latest[f'EMA_{EMA_FAST}'] > latest[f'EMA_{EMA_SLOW}']
    
    # Trend Strength: ADX is above threshold
    strong_trend = latest[f'ADX_{14}'] > ADX_THRESHOLD
    
    if ema_cross_up and strong_trend:
        print(f"📈 Buy Signal Found! EMA Cross: {ema_cross_up}, ADX: {latest[f'ADX_{14}']:.2f}")
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
        return round(qty, 6) # Round to a reasonable precision for BTC

    async def open_position(self, df):
        latest = df.iloc[-1]
        entry_price = latest['close']
        atr = latest[f'ATRr_{14}']
        
        self.stop_loss_price = entry_price - (atr * ATR_MULTIPLIER)
        self.take_profit_price = entry_price + ((entry_price - self.stop_loss_price) * TAKE_PROFIT_RATIO)
        self.qty = self.calculate_position_size(entry_price, self.stop_loss_price)

        if self.qty <= 0 or (self.qty * entry_price > self.balance):
            print("⚠️ Insufficient balance or invalid quantity. Cannot open position.")
            return

        # In a live deployment, you would uncomment this to place a real order
        # order_result = await self.api.place_order(SYMBOL, self.qty, entry_price, "buy")
        
        # For simulation purposes, we assume the order fills instantly
        self.entry_price = entry_price
        self.position_open = True
        print(f"\n✅ Position Opened: Qty={self.qty:.6f} @ ${self.entry_price:.2f}")
        print(f"   - Stop Loss: ${self.stop_loss_price:.2f}")
        print(f"   - Take Profit: ${self.take_profit_price:.2f}")

    def check_exit(self, current_price):
        pnl = 0.0
        exit_reason = None
        
        if current_price <= self.stop_loss_price:
            exit_reason = "Stop-Loss"
        elif current_price >= self.take_profit_price:
            exit_reason = "Take-Profit"
        
        if exit_reason:
            # In a live deployment, you'd place a sell order here
            # await self.api.place_order(SYMBOL, self.qty, current_price, "sell")

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

# ----- Main Bot Logic -----
async def main():
    gemini_api = GeminiAPI(GEMINI_API_KEY, GEMINI_API_SECRET, IS_SANDBOX)
    trade_manager = TradeManager(initial_balance=START_BALANCE, api=gemini_api)
    
    print(f"🚀 Bot started on {SYMBOL} ({TIMEFRAME}) with ${START_BALANCE:.2f} USD")
    print(f"   - Mode: {'Sandbox (Paper Trading)' if IS_SANDBOX else 'Live Trading'}")
    print("-" * 30)

    if not GEMINI_API_KEY or not GEMINI_API_SECRET:
        print("🛑 Critical Error: API keys not found. Shutting down.")
        return

    try:
        while True:
            candles_df = await gemini_api.get_candles(SYMBOL, TIMEFRAME)
            
            if candles_df is None or candles_df.empty:
                print("\nCould not fetch candle data. Retrying in 60s...")
                await asyncio.sleep(60)
                continue

            candles_df = add_indicators(candles_df)
            current_price = candles_df.iloc[-1]['close']
            
            # Use \r to overwrite the line, and end="" to prevent a newline
            print(f"\r📊 Checking {time.strftime('%Y-%m-%d %H:%M:%S')}, Price: ${current_price:.2f}, ADX: {candles_df.iloc[-1][f'ADX_{14}']:.2f}  ", end="")

            if not trade_manager.position_open:
                if check_buy_signal(candles_df):
                    await trade_manager.open_position(candles_df)
            else:
                trade_manager.check_exit(current_price)
            
            # This sleep is simple. A more robust solution aligns with candle closing times.
            await asyncio.sleep(60) 

    except asyncio.CancelledError:
        print("\n🛑 Bot shutting down.")
    finally:
        await gemini_api.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped manually.")
