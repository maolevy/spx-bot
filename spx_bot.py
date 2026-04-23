import yfinance as yf
import pandas as pd
import numpy as np
import anthropic
import requests
import schedule
import time
import json
import os
import pytz
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID        = os.getenv("TELEGRAM_CHAT_ID")

client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)

last_signal = {"action": None, "timestamp": None}


def calculate_rsi(prices, period=14):
    delta    = prices.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs       = avg_gain / avg_loss
    rsi      = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast    = prices.ewm(span=fast).mean()
    ema_slow    = prices.ewm(span=slow).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger(prices, period=20, std_dev=2):
    middle = prices.rolling(window=period).mean()
    std    = prices.rolling(window=period).std()
    upper  = middle + (std * std_dev)
    lower  = middle - (std * std_dev)
    return upper, middle, lower


def prepare_data():
    ticker = yf.Ticker("^GSPC")
    df     = ticker.history(period="5d", interval="5m")

    df["RSI"] = calculate_rsi(df["Close"])
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = calculate_macd(df["Close"])
    df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = calculate_bollinger(df["Close"])
    df["EMA_20"]     = df["Close"].ewm(span=20).mean()
    df["EMA_50"]     = df["Close"].ewm(span=50).mean()
    df["Change_Pct"] = df["Close"].pct_change() * 100

    return df.dropna()


def analyze_with_claude(df):
    current = df.iloc[-1]

    summary = f"""
=== נתוני SPX - {current.name.strftime('%Y-%m-%d %H:%M')} ===

מחיר נוכחי:  {current['Close']:.2f}
שינוי נר זה: {current['Change_Pct']:.2f}%

אינדיקטורים:
RSI(14):        {current['RSI']:.1f}
MACD Histogram: {current['MACD_Hist']:.3f}
EMA 20:         {current['EMA_20']:.2f}
EMA 50:         {current['EMA_50']:.2f}
BB Upper:       {current['BB_Upper']:.2f}
BB Lower:       {current['BB_Lower']:.2f}

10 נרות אחרונים:
{df.tail(10)[['Close','RSI','MACD_Hist']].to_string()}
"""

    prompt = f"""
אתה אנליסט טכני מנוסה.

{summary}

ענה אך ורק ב-JSON תקין:
{{
    "action":     "LONG" או "SHORT" או "NEUTRAL",
    "confidence": מספר בין 1 ל-10,
    "reason":     "הסבר קצר",
    "entry_zone": "טווח כניסה",
    "stop_loss":  "סטופ לוס",
    "target":     "יעד"
}}

כללים:
- LONG:    RSI מתחת ל-45, MACD Histogram עולה, מחיר מעל EMA20
- SHORT:   RSI מעל 60, MACD Histogram יורד, מחיר מתחת EMA20
- NEUTRAL: כל מצב אחר
"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    text = response.content[0].text
    return json.loads(text)


def format_message(signal, price):
    if signal["action"] == "LONG":
        emoji = "🟢"
        title = "LONG - כניסה לונג"
    elif signal["action"] == "SHORT":
        emoji = "🔴"
        title = "SHORT - כניסה שורט"
    else:
        return None

    conf = signal["confidence"]
    bars = "█" * conf + "░" * (10 - conf)
    now  = datetime.now().strftime("%H:%M:%S")

    return f"""{emoji} <b>SPX SIGNAL — {title}</b>
━━━━━━━━━━━━━━━━━━━━
🕐 זמן:       {now}
💰 מחיר:      {price:.2f}
📊 ביטחון:    {conf}/10  [{bars}]

📍 כניסה:     {signal['entry_zone']}
🛑 סטופ לוס:  {signal['stop_loss']}
🎯 יעד:       {signal['target']}

💬 <i>{signal['reason']}</i>
━━━━━━━━━━━━━━━━━━━━
⚠️ לא ייעוץ פיננסי"""


def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    response = requests.post(url, json={
        "chat_id":    CHAT_ID,
        "text":       message,
        "parse_mode": "HTML"
    })
    if response.status_code == 200:
        print("✅ הודעה נשלחה לטלגרם!")
    else:
        print(f"❌ שגיאת טלגרם: {response.text}")


def is_market_open():
    ny  = pytz.timezone("America/New_York")
    now = datetime.now(ny)
    if now.weekday() >= 5:
        return False
    open_time  = now.replace(hour=9,  minute=30, second=0)
    close_time = now.replace(hour=16, minute=0,  second=0)
    return open_time <= now <= close_time


def should_send(new_signal):
    global last_signal
    if last_signal["action"] == new_signal["action"]:
        if last_signal["timestamp"]:
            elapsed = (datetime.now() - last_signal["timestamp"]).seconds
            if elapsed < 7200:
                return False
    return True


def run_bot():
    global last_signal
    try:
        if not is_market_open():
            print(f"🔒 {datetime.now().strftime('%H:%M')} - השוק סגור")
            return

        print(f"🔍 {datetime.now().strftime('%H:%M:%S')} - מנתח שוק...")

        df            = prepare_data()
        current_price = df.iloc[-1]["Close"]
        signal        = analyze_with_claude(df)

        print(f"📊 אות: {signal['action']} | ביטחון: {signal['confidence']}/10")

        if signal["confidence"] < 6:
            print("⚠️ ביטחון נמוך - מדלג")
            return

        if not should_send(signal):
            print("⏭️ אות כפול - מדלג")
            return

        message = format_message(signal, current_price)
        if message:
            send_telegram(message)

        last_signal = {
            "action":    signal["action"],
            "timestamp": datetime.now()
        }

    except json.JSONDecodeError:
        print("❌ שגיאת JSON")
    except Exception as e:
        print(f"❌ שגיאה: {e}")


if __name__ == "__main__":
    print("🤖 SPX Bot מתחיל...")
    run_bot()
    schedule.every(15).minutes.do(run_bot)
    print("✅ הבוט פעיל - מנתח כל 15 דקות")
    while True:
        schedule.run_pending()
        time.sleep(30)