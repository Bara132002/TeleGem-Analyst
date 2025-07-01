import os
import requests
import pandas as pd
import pandas_ta as ta
from flask import Flask, request
from telegram import Bot, Update
from telegram.ext import Dispatcher, CommandHandler, MessageHandler, filters
import google.generativeai as genai
import json # Untuk debugging respon Alpha Vantage

# --- Untuk Local Running ---
# Comment jika memilih Render Web
from dotenv import load_dotenv; load_dotenv()

# --- Konfigurasi API & Environment Variables ---
# Pastikan variabel ini diset di Render (Environment Variables)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
WEBHOOK_URL = os.getenv("WEBHOOK_URL") # Ini akan didapatkan dari Render

# Inisialisasi Bot Telegram dan Gemini
bot = Bot(token=TELEGRAM_BOT_TOKEN)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-pro')

app = Flask(__name__)
dispatcher = Dispatcher(bot, None, use_context=True)

# --- Fungsi untuk Ambil Data dari Alpha Vantage ---
def get_alpha_vantage_data(symbol: str, interval: str = '60min', outputsize: str = 'compact'):
    """
    Mengambil data OHLC dari Alpha Vantage.
    Simbol: Misal "FX:EURUSD", "XAUUSD" (untuk gold)
    Interval: "1min", "5min", "15min", "30min", "60min", "daily", "weekly", "monthly"
    Outputsize: "compact" (100 data terakhir), "full" (semua data)
    """
    # Untuk Forex & Gold, gunakan TIME_SERIES_INTRADAY atau FX_DAILY / DIGITAL_CURRENCY_DAILY
    # Alpha Vantage memiliki struktur API yang berbeda untuk setiap jenis aset.
    # Contoh untuk Forex (misal EURUSD)
    if "FX:" in symbol.upper():
        function = "FX_INTRADAY" if "min" in interval else "FX_DAILY"
        url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol.replace('FX:','')}&interval={interval}&outputsize={outputsize}&apikey={ALPHA_VANTAGE_API_KEY}"
        print(f"Alpha Vantage URL (Forex): {url}") # Debugging
    # Contoh untuk Gold (XAUUSD) - Alpha Vantage tidak menyediakan XAUUSD langsung untuk intraday di free tier.
    # Anda mungkin perlu mencari alternatif API untuk Gold intraday jika ini krusial.
    # Sebagai alternatif, bisa pakai CRYPTO_INTRADAY untuk pasangan kripto jika relevan,
    # atau TIMESERIES_INTRADAY untuk saham/indeks dengan ticker yang sesuai.
    # Untuk demo XAUUSD, kita asumsikan menggunakan data 'DAILY' atau 'WEEKLY' jika intraday tidak ada di free tier.
    elif "XAUUSD" in symbol.upper():
        function = "DIGITAL_CURRENCY_DAILY" # Alpha Vantage tidak menyediakan Gold langsung
        # Akan lebih baik mencari API spesifik untuk Gold (misal dari Finnhub/TwelveData)
        # Untuk tujuan demonstrasi, kita bisa coba pakai FX_DAILY untuk USD/JPY dan mengasumsikan korelasi atau gunakan dummy.
        # Ini adalah keterbatasan Alpha Vantage free tier untuk XAUUSD intraday.
        print("Peringatan: Alpha Vantage free tier mungkin tidak menyediakan data intraday XAUUSD secara langsung.")
        print("Mungkin perlu menggunakan data harian atau mencari API Gold terpisah (misal Finnhub, Twelve Data).")
        # Untuk demo, kita bisa paksakan dengan 'FX_INTRADAY' jika memang ada pasangan 'XAU/USD' di sana, atau pakai FX_DAILY
        url = f"https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol=XAU&to_symbol=USD&interval={interval}&outputsize={outputsize}&apikey={ALPHA_VANTAGE_API_KEY}"
        # Jika FX_INTRADAY XAU/USD tidak bekerja, pertimbangkan TIME_SERIES_DAILY
        # url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=GC=F&apikey={ALPHA_VANTAGE_API_KEY}&outputsize={outputsize}" # Futures Gold
    else:
        # Default untuk saham/indeks umum
        function = "TIME_SERIES_INTRADAY" if "min" in interval else "TIME_SERIES_DAILY"
        url = f"https://www.alphavantage.co/query?function={function}&symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={ALPHA_VANTAGE_API_KEY}"
        print(f"Alpha Vantage URL (General): {url}") # Debugging

    try:
        response = requests.get(url)
        response.raise_for_status() # Cek jika ada error HTTP
        data = response.json()

        # print("Raw Alpha Vantage Data:", json.dumps(data, indent=2)) # Debugging

        # Parsing data yang berbeda tergantung fungsi API
        time_series_key = None
        if "FX_INTRADAY" in function:
            time_series_key = f"Time Series FX ({interval})"
        elif "FX_DAILY" in function:
            time_series_key = "Time Series FX (Daily)"
        elif "Time Series" in function: # For Stocks/Indices
            time_series_key = f"Time Series ({interval})" if "min" in interval else "Time Series (Daily)"
        elif "Digital Currency Daily" in function:
            time_series_key = "Time Series (Digital Currency Daily)" # For Crypto

        if time_series_key and time_series_key in data:
            raw_df = pd.DataFrame.from_dict(data[time_series_key], orient='index', dtype=float)
            raw_df.index = pd.to_datetime(raw_df.index)
            # Rename columns to standard OHLCV
            df = raw_df.rename(columns={
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume',
                '1a. open (USD)': 'open', # For Digital Currency
                '2a. high (USD)': 'high',
                '3a. low (USD)': 'low',
                '4a. close (USD)': 'close',
                '5. volume': 'volume',
                '6. market cap (USD)': 'market cap' # Drop this if not needed
            })
            df = df[['open', 'high', 'low', 'close', 'volume']].sort_index()
            return df
        elif "Error Message" in data:
            print(f"Alpha Vantage Error: {data['Error Message']}")
            return None
        elif "Note" in data:
            print(f"Alpha Vantage Note (Rate Limit?): {data['Note']}")
            return None
        else:
            print("Alpha Vantage: No relevant time series data found.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"HTTP Request failed: {e}")
        return None
    except json.JSONDecodeError:
        print("Failed to decode JSON response from Alpha Vantage.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# --- Fungsi untuk Analisa Trading & Prompt Gemini ---
def perform_trading_analysis(symbol: str, interval: str):
    try:
        # 1. Ambil Data
        df = get_alpha_vantage_data(symbol, interval)
        if df is None or df.empty:
            return None, "Tidak dapat mengambil data dari Alpha Vantage. Pastikan simbol/interval benar atau coba lagi nanti (batasan rate limit)."

        # Pastikan ada cukup data untuk indikator
        if len(df) < 20: # Contoh, RSI butuh 14, MACD butuh lebih
            return None, "Data tidak cukup untuk melakukan analisis teknikal yang relevan."

        # 2. Hitung Indikator Teknis menggunakan Pandas-TA
        df.ta.rsi(append=True) # RSI 14
        df.ta.macd(append=True) # MACD
        df.ta.ema(length=50, append=True) # EMA 50
        df.ta.ema(length=200, append=True) # EMA 200
        df.ta.bbands(append=True) # Bollinger Bands

        # Ambil nilai terbaru
        last_row = df.iloc[-1]
        current_price = round(last_row['close'], 4) # Bulatkan untuk harga, misal 4 desimal
        rsi_val = round(last_row['RSI_14'], 2)
        macd_hist = round(last_row['MACDh_12_26_9'], 4)
        ema50 = round(last_row['EMA_50'], 4)
        ema200 = round(last_row['EMA_200'], 4)
        bb_upper = round(last_row['BBL_5_2.0'], 4) # Lower Band (typo di docs pandas_ta, should be BBU)
        bb_lower = round(last_row['BBU_5_2.0'], 4) # Upper Band (typo di docs pandas_ta, should be BBL)
        # Corrected:
        bb_upper = round(df.iloc[-1]['BBL_5_2.0'], 4) # Corrected BBL/BBU based on recent pandas_ta versions
        bb_lower = round(df.iloc[-1]['BBU_5_2.0'], 4)

        # 3. Implementasi Logika Trading & Penentuan TP/SL/Confidence (Ini adalah bagian KRUSIAL & perlu disempurnakan)
        trade_direction = "NETRAL"
        entry_price = current_price
        tp_price = current_price
        sl_price = current_price
        confidence_percentage = 50 # Default confidence

        technical_reasons = []

        # --- Contoh Logika Sederhana (Ini adalah STARTER, perlu disempurnakan sesuai strategi Anda!) ---
        # Logika BUY
        if rsi_val < 30 and current_price > ema50: # RSI oversold & di atas EMA50
            trade_direction = "BUY"
            # Asumsi entry di harga saat ini
            entry_price = current_price
            # TP: Di dekat resistance atau batas atas BB, SL: Di bawah support atau batas bawah BB
            tp_price = round(current_price + (current_price * 0.005), 4) # Contoh: 0.5% dari harga
            sl_price = round(current_price - (current_price * 0.003), 4) # Contoh: 0.3% dari harga
            technical_reasons.append(f"RSI ({rsi_val}) di area oversold, mengindikasikan potensi pantulan naik.")
            technical_reasons.append(f"Harga berada di atas EMA50 ({ema50}), menunjukkan tren naik jangka pendek.")
            technical_reasons.append(f"Pola candlestick terakhir mungkin bullish (perlu deteksi pola).") # Placeholder
            confidence_percentage += 20 # Tingkatkan confidence
        
        # Logika SELL
        elif rsi_val > 70 and current_price < ema50: # RSI overbought & di bawah EMA50
            trade_direction = "SELL"
            entry_price = current_price
            tp_price = round(current_price - (current_price * 0.005), 4)
            sl_price = round(current_price + (current_price * 0.003), 4)
            technical_reasons.append(f"RSI ({rsi_val}) di area overbought, mengindikasikan potensi koreksi turun.")
            technical_reasons.append(f"Harga berada di bawah EMA50 ({ema50}), menunjukkan tren turun jangka pendek.")
            technical_reasons.append(f"Pola candlestick terakhir mungkin bearish (perlu deteksi pola).") # Placeholder
            confidence_percentage += 20 # Tingkatkan confidence
        
        # Logika Netral (jika tidak ada sinyal kuat)
        else:
            technical_reasons.append("Kondisi pasar saat ini netral atau konsolidasi, tidak ada sinyal trading yang jelas dari indikator utama.")
            confidence_percentage = 40 # Turunkan confidence jika tidak ada sinyal kuat

        # Anda dapat menambahkan logika lebih lanjut untuk:
        # - Pola Candlestick (perlu library eksternal atau implementasi manual)
        # - Support/Resistance Levels
        # - Divergensi (RSI/MACD dengan Harga)
        # - Konfirmasi dari Multiple Timeframes

        # Hitung Risk-Reward Ratio (jika ada setup yang jelas)
        risk_reward_ratio = "N/A"
        if trade_direction != "NETRAL":
            risk = abs(entry_price - sl_price)
            reward = abs(tp_price - entry_price)
            if risk > 0:
                risk_reward_ratio = f"1:{reward/risk:.2f}" # Format 1:X.XX

        # 4. Siapkan Prompt untuk Gemini
        prompt_for_gemini = f"""
        Anda adalah analis trading profesional. Berdasarkan data teknis yang disediakan, berikan analisa trading singkat dan langsung ke poin untuk {symbol} pada timeframe {interval}. Sertakan rekomendasi arah trading, harga entry, Take Profit (TP), Stop Loss (SL), dan tingkat konfidensi dalam persentase. Jelaskan alasan teknis utamanya dan berikan rasio risk-reward.

        Output harus dalam format yang mudah dibaca dan ringkas, seperti contoh ini:

        **Analisa Trading {symbol} {interval}:**

        **Rekomendasi:** [BUY/SELL/NETRAL]
        **Harga Entry:** [Harga]
        **Take Profit (TP):** [Harga]
        **Stop Loss (SL):** [Harga]
        **Tingkat Konfidensi:** [Persentase]%

        **Alasan Teknis Utama:**
        * [Poin 1]
        * [Poin 2]
        * ...

        **Rasio Risk-Reward:** [Rasio]

        **Informasi Tambahan:**
        - Harga Saat Ini: {current_price}
        - RSI (14): {rsi_val}
        - MACD Hist: {macd_hist}
        - EMA50: {ema50}
        - EMA200: {ema200}
        - BB Upper: {bb_upper}
        - BB Lower: {bb_lower}

        Data Analisa Teknis untuk Anda gunakan dalam output:
        - Aset: {symbol}
        - Timeframe: {interval}
        - Harga Saat Ini: {current_price}
        - Arah Rekomendasi: {trade_direction}
        - Harga Entry: {entry_price}
        - Harga Take Profit (TP): {tp_price}
        - Harga Stop Loss (SL): {sl_price}
        - Tingkat Konfidensi: {confidence_percentage}%
        - Alasan Teknis Utama:
        {chr(10).join([f"- {reason}" for reason in technical_reasons])}
        - Rasio Risk-Reward: {risk_reward_ratio}

        Pastikan untuk selalu menyertakan disclaimer singkat tentang risiko trading di akhir.
        """
        
        return prompt_for_gemini, None

    except Exception as e:
        print(f"Error in perform_trading_analysis: {e}") # Untuk debugging
        return None, f"Terjadi kesalahan dalam analisis data: {e}"

# --- Handler Bot Telegram ---
def start(update: Update, context):
    update.message.reply_text("Halo! Saya bot AI analisa trading Anda. Gunakan /analisa <simbol> <timeframe> untuk mendapatkan analisa.")

def analyze_command(update: Update, context):
    args = context.args
    if not args or len(args) < 2:
        update.message.reply_text("Penggunaan: /analisa <simbol> <timeframe>\nContoh: /analisa EURUSD 60min\nContoh Gold: /analisa XAUUSD daily")
        return

    symbol = args[0].upper()
    interval = args[1].lower() # Interval Alpha Vantage biasanya lowercase (e.g., "60min", "daily")

    update.message.reply_text(f"Menganalisis {symbol} di timeframe {interval}. Mohon tunggu sejenak...")

    prompt_data_for_gemini, error_message = perform_trading_analysis(symbol, interval)

    if error_message:
        update.message.reply_text(f"Gagal melakukan analisis: {error_message}")
        return

    try:
        # Kirim prompt_data_for_gemini ke model Gemini
        response = gemini_model.generate_content(prompt_data_for_gemini)
        ai_analysis_text = response.text
        
        # Tambahkan disclaimer secara otomatis di akhir
        final_message = ai_analysis_text + "\n\n_Disclaimer: Trading forex dan gold memiliki risiko tinggi. Informasi ini bukan saran keuangan dan hanya untuk tujuan edukasi._"
        update.message.reply_text(final_message)
    except Exception as e:
        update.message.reply_text(f"Maaf, terjadi kesalahan saat memproses dengan Gemini: {e}")
        # Jika Gemini gagal, Anda bisa coba kirim data mentah saja
        # update.message.reply_text(f"Analisis Teknis (tanpa Gemini):\n{prompt_data_for_gemini}")


# --- Webhook Endpoint untuk Flask ---
@app.route('/webhook', methods=['POST'])
def webhook():
    if request.method == "POST":
        update = Update.de_json(request.get_json(force=True), bot)
        dispatcher.process_update(update)
    return "ok"

# --- Main Running (untuk Render) ---
if __name__ == "__main__":
    # Render akan menyediakan variabel PORT. Pastikan aplikasi listen di port ini.
    port = int(os.environ.get("PORT", 5000))
    
    # Set webhook ke Telegram. Ini perlu dilakukan sekali saat deploy atau jika URL Render berubah.
    # Disarankan untuk melakukan ini secara manual di browser/curl setelah bot ter-deploy di Render
    # karena kita butuh WEBHOOK_URL yang dihasilkan Render.
    # Contoh manual: https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/setWebhook?url=<RENDER_APP_URL>/webhook
    
    app.run(host="0.0.0.0", port=port)