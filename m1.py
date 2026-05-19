import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from dtaidistance import dtw
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# =========================================================
# Настройки страницы
# =========================================================

st.set_page_config(
    page_title="Фрактальный анализ рынка",
    layout="wide"
)

st.title("Фрактальный анализ рынка ценных бумаг")

# =========================================================
# Пресеты активов
# =========================================================

assets = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Tesla": "TSLA",
    "NVIDIA": "NVDA",
    "Amazon": "AMZN",
    "Google": "GOOGL",
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "S&P500": "^GSPC",
    "NASDAQ": "^IXIC"
}

# =========================================================
# Боковая панель
# =========================================================

st.sidebar.header("Параметры анализа")

selected_asset = st.sidebar.selectbox(
    "Выберите актив",
    list(assets.keys())
)

ticker = assets[selected_asset]

period = st.sidebar.selectbox(
    "Диапазон истории",
    [
        "6mo",
        "1y",
        "2y",
        "5y",
        "10y"
    ],
    index=3
)

interval = st.sidebar.selectbox(
    "Таймфрейм",
    [
        "1d",
        "1wk",
        "1mo"
    ]
)

pattern_length = st.sidebar.slider(
    "Количество дней для фрактального анализа",
    min_value=10,
    max_value=200,
    value=60
)

forecast_days = st.sidebar.slider(
    "Длина прогноза",
    min_value=5,
    max_value=60,
    value=20
)

# =========================================================
# Загрузка данных
# =========================================================

@st.cache_data
def load_data(ticker_name, period_value, interval_value):
    data = yf.download(
        ticker_name,
        period=period_value,
        interval=interval_value,
        auto_adjust=True
    )

    data = data[['Close']].dropna()
    data.columns = ['Close']

    return data


data = load_data(ticker, period, interval)

if len(data) < pattern_length + forecast_days + 50:
    st.error("Недостаточно данных для анализа")
    st.stop()

# =========================================================
# Подготовка данных
# =========================================================

prices = data['Close'].values

# Последний участок графика
target_pattern = prices[-pattern_length:]

# Нормализация
scaler = MinMaxScaler()

target_scaled = scaler.fit_transform(
    target_pattern.reshape(-1, 1)
).flatten()

# =========================================================
# Поиск похожего фрактала
# =========================================================

best_distance = float("inf")
best_index = None
best_pattern = None

search_limit = len(prices) - pattern_length - forecast_days

for i in range(search_limit):

    historical_pattern = prices[i:i + pattern_length]

    historical_scaled = scaler.fit_transform(
        historical_pattern.reshape(-1, 1)
    ).flatten()

    distance = dtw.distance(
        target_scaled,
        historical_scaled
    )

    if distance < best_distance:
        best_distance = distance
        best_index = i
        best_pattern = historical_pattern

# =========================================================
# Расчет процента совпадения
# =========================================================

similarity_percent = max(
    0,
    100 - best_distance * 10
)

similarity_percent = min(similarity_percent, 100)

# =========================================================
# Построение прогноза
# =========================================================

future_part = prices[
    best_index + pattern_length:
    best_index + pattern_length + forecast_days
]

base_price = target_pattern[-1]
historical_base = best_pattern[-1]

forecast_relative = future_part / historical_base
forecast_prices = base_price * forecast_relative

# =========================================================
# Индексы времени
# =========================================================

chart_dates = data.index

forecast_dates = []

last_date = chart_dates[-1]

if interval == "1d":
    delta = timedelta(days=1)

elif interval == "1wk":
    delta = timedelta(weeks=1)

else:
    delta = timedelta(days=30)

for i in range(forecast_days):
    forecast_dates.append(
        last_date + delta * (i + 1)
    )

# =========================================================
# График
# =========================================================

fig = go.Figure()

# Основной график
fig.add_trace(
    go.Scatter(
        x=chart_dates,
        y=prices,
        mode='lines',
        name='Цена',
        line=dict(width=2)
    )
)

# Найденный похожий участок
fig.add_trace(
    go.Scatter(
        x=chart_dates[
            best_index:
            best_index + pattern_length
        ],
        y=best_pattern,
        mode='lines',
        name='Похожий фрактал',
        line=dict(
            dash='dot',
            width=3
        )
    )
)

# Прогноз
fig.add_trace(
    go.Scatter(
        x=forecast_dates,
        y=forecast_prices,
        mode='lines',
        name='Прогноз',
        line=dict(
            color='red',
            width=4
        )
    )
)

# Последний анализируемый участок
fig.add_trace(
    go.Scatter(
        x=chart_dates[-pattern_length:],
        y=target_pattern,
        mode='lines',
        name='Текущий участок',
        line=dict(
            color='green',
            width=4
        )
    )
)

fig.update_layout(
    title=f"Фрактальный анализ: {selected_asset}",
    xaxis_title="Дата",
    yaxis_title="Цена",
    height=800,
    template="plotly_dark",
    hovermode="x unified"
)

# =========================================================
# Вывод результатов
# =========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Процент совпадения",
        f"{similarity_percent:.2f}%"
    )

with col2:
    st.metric(
        "DTW расстояние",
        f"{best_distance:.4f}"
    )

with col3:
    st.metric(
        "Длина прогноза",
        f"{forecast_days} дней"
    )

st.plotly_chart(
    fig,
    use_container_width=True
)

# =========================================================
# Информация
# =========================================================

# st.subheader("Описание метода")
#
# st.write("""
# Приложение выполняет фрактальный анализ временного ряда:
#
# 1. Берется последний участок графика.
# 2. Выполняется поиск максимально похожего участка в истории.
# 3. Используется алгоритм Dynamic Time Warping (DTW).
# 4. После нахождения похожего паттерна строится прогноз,
#    исходя из того, как двигалась цена после найденного участка в прошлом.
# """)