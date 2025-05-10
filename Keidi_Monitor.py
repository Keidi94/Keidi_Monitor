import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import requests
from datetime import datetime

# ----------------------------
# 기본 설정
# ----------------------------
st.set_page_config(page_title="KEIDI Monitor", layout="wide")
st.title("KEIDI Monitor")
st.markdown("글로벌 자산(통화, 증시, 금리)을 모니터링하는 웹 대시보드입니다.")

# ----------------------------
# 공통 데이터 함수
# ----------------------------
def make_static_plotly(fig):
    fig.update_layout(
        dragmode=False,
        hovermode="x unified",
        xaxis_showspikes=True,
        yaxis_showspikes=True,
        showlegend=True,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
    )

    # ② hover 스타일 설정
    fig.update_layout(
        modebar_remove=[
            'zoom', 'pan', 'select', 'lasso2d',
            'zoomIn', 'zoomOut', 'resetScale'
        ],
        modebar_add=[
            'toImage', 'fullscreen', 'autoScale', 'toggleSpikelines'
        ]
    )
    return fig
def make_interactive_plotly(fig):
    fig.update_layout(
        dragmode=False,
        hovermode="x unified",
        xaxis_showspikes=True,
        yaxis_showspikes=True,
        xaxis=dict(spikemode="across", spikethickness=1),
        yaxis=dict(spikemode="across", spikethickness=1),
        showlegend=True,
    )
    fig.update_traces(mode="lines", hoverinfo="all")
    fig.update_layout(
        modebar_remove=[
            'zoom', 'pan', 'select', 'lasso2d',
            'zoomIn', 'zoomOut', 'resetScale'
        ],
        modebar_add=[
            'toImage', 'fullscreen', 'autoScale', 'toggleSpikelines'
        ]
    )
    return fig
def get_yf_data(ticker, label, period="10y"):
    ticker_obj = yf.Ticker(ticker)
    hist = ticker_obj.history(period=period)
    if hist.empty:
        st.error(f"{label} 데이터 없음")
        return pd.DataFrame()
    df = hist.reset_index()[["Date", "Close"]]
    df.rename(columns={"Date": "날짜", "Close": label}, inplace=True)
    return df
def plot_with_moving_averages(df, label):
    fig = go.Figure()

    # 원래 시세 선: 흰색 실선
    fig.add_trace(go.Scatter(
        x=df["날짜"], y=df[label],
        mode="lines",
        name=label,
        line=dict(color="white", width=2)
    ))

    # 이동평균선: 점선
    for window in [50, 200]:
        ma_col = f"{label}_MA{window}"
        if ma_col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["날짜"], y=df[ma_col],
                mode="lines",
                name=f"MA{window}",
                line=dict(dash="dot", width=1.5)
            ))

    # Plotly 인터랙션 설정 적용
    fig = make_interactive_plotly(fig)

    # ✅ RangeSelector + RangeSlider + y축 auto scale
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    # dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=3, label="3y", step="year", stepmode="backward"),
                    dict(count=5, label="5y", step="year", stepmode="backward"),
                    dict(step="all", label="전체")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        ),
        yaxis=dict(autorange=True)  # 👉 y축도 선택 구간에 맞춰 자동 조절
    )

    return fig
def add_moving_averages(df: pd.DataFrame, label: str, windows=[50, 200]):
    for window in windows:
        ma_col = f"{label}_MA{window}"
        df[ma_col] = df[label].rolling(window=window).mean()
    return df
def get_fred_data(series_id, start_date="2015-01-01"):
    api_key = "f9ce1a8f7a12a119685067823fb3755c"
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        st.error(f"FRED API 오류 (status code: {response.status_code})")
        return pd.DataFrame()

    data = response.json()

    # 👇 오류 메시지 처리
    if "observations" not in data:
        st.error(f"FRED API 오류: {data.get('error_message', '알 수 없는 오류')}")
        return pd.DataFrame()

    df = pd.DataFrame(data["observations"])
    df["날짜"] = pd.to_datetime(df["date"])
    df[series_id] = pd.to_numeric(df["value"], errors="coerce")
    return df[["날짜", series_id]]

# ----------------------------
# 통화 섹션
# ----------------------------
def show_currency_section():
    st.header("💱 통화 시장")

    # 통화쌍 목록: DXY + 주요 환율
    pairs = [
        ("DX-Y.NYB", "DXY"),          # 단독 차트로 크게
        ("EURUSD=X", "EUR/USD"),
        ("KRW=X", "USD/KRW"),
        ("JPYKRW=X", "JPY/KRW"),
        ("EURKRW=X", "EUR/KRW")
    ]

    # 1️⃣ DXY는 전체 너비로 단독 표시
    ticker, label = pairs[0]
    df = get_yf_data(ticker, label)
    if not df.empty:
        df = add_moving_averages(df, label)  # 이동평균선 추가
        fig = plot_with_moving_averages(df, label)
        st.plotly_chart(fig, use_container_width=True)

    # 2️⃣ 나머지 4개 통화쌍은 2열씩 분할
    for i in range(1, len(pairs), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(pairs):
                ticker, label = pairs[i + j]
                df = get_yf_data(ticker, label)
                if not df.empty:
                    df = add_moving_averages(df, label)
                    fig = plot_with_moving_averages(df, label)
                    cols[j].plotly_chart(fig, use_container_width=True)
# ----------------------------
# 증시 섹션
# ----------------------------
def show_equity_section():
    st.header("글로벌 증시")
    index_list = [
        ("^GSPC", "S&P 500"),
        ("^GDAXI", "독일 DAX"),
        ("^FCHI", "프랑스 CAC40"),
        ("^FTSE", "영국 FTSE100"),
        ("^N225", "일본 Nikkei 225"),
        ("^HSI", "홍콩 항셍지수"),
        ("000300.SS", "중국 CSI300"),
        ("^KS200", "KOSPI 200")
    ]
    for ticker, label in index_list:
        df = get_yf_data(ticker, label)
        if not df.empty:
            fig = px.line(df, x="날짜", y=label, title=label)
            st.plotly_chart(make_static_plotly(fig), use_container_width=True)

# ----------------------------
# 금리 섹션
# ----------------------------
def show_bond_section():
    st.header("미국 국채 금리")

    periods = {
        "1년": "1y",
        "3년": "3y",
        "10년": "10y",
        "20년": "20y"
    }

    tickers = {
        "3개월": "^IRX",
        "2년": "^FVX",
        "10년": "^TNX"
    }

    def get_multi_bond_df(period_code):
        df_all = pd.DataFrame()
        for label, ticker in tickers.items():
            df = get_yf_data(ticker, label, period=period_code)
            if not df.empty:
                if df_all.empty:
                    df_all = df
                else:
                    df_all = pd.merge(df_all, df, on="날짜", how="outer")
        df_all.dropna(subset=["3개월", "2년", "10년"], inplace=True)
        return df_all

    for label, period_code in periods.items():
        df = get_multi_bond_df(period_code)
        if df.empty:
            st.warning(f"{label} 데이터가 없습니다.")
            continue

        with st.expander(f"📈 미국채 금리 추세 ({label})", expanded=(label == "1년")):
            fig1 = px.line(df, x="날짜", y=["3개월", "2년", "10년"],
                           labels={"value": "금리", "variable": "만기"},
                           title=f"미국채 금리 ({label})")
            st.plotly_chart(make_static_plotly(fig1), use_container_width=True)

        df["10y-2y"] = df["10년"] - df["2년"]
        df["10y-3m"] = df["10년"] - df["3개월"]

        with st.expander(f"📉 금리 스프레드 ({label})", expanded=(label == "1년")):
            fig2 = px.line(df, x="날짜", y=["10y-2y", "10y-3m"],
                           labels={"value": "스프레드", "variable": "구간"},
                           title=f"미국채 금리 스프레드 ({label})")
            st.plotly_chart(make_static_plotly(fig2), use_container_width=True)

# ----------------------------
# 경제 섹션
# ----------------------------

def show_economy_section():
    st.header("🌎 글로벌 경제 지표")

    series_info = [
        ("UNRATE", "미국 실업률 (%)"),
        ("PCEC96", "미국 실질 소비지출 (PCE)"),
        ("NAPM", "ISM 제조업 PMI")
    ]

    for series_id, label in series_info:
        df = get_fred_data(series_id, start_date="2010-01-01")
        if df.empty:
            st.warning(f"{label} 데이터를 불러올 수 없습니다.")
            continue

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["날짜"], y=df[series_id],
            mode="lines",
            name=label,
            line=dict(color="lightblue")
        ))

        fig = make_interactive_plotly(fig)

        fig.update_layout(
            title=label,
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(count=3, label="3y", step="year", stepmode="backward"),
                        dict(count=5, label="5y", step="year", stepmode="backward"),
                        dict(step="all", label="전체")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            ),
            yaxis=dict(autorange=True)
        )

        st.plotly_chart(fig, use_container_width=True)
# ----------------------------
# 사이드바 메뉴 및 라우팅
# ----------------------------
st.sidebar.markdown("카테고리 선택")
category = st.sidebar.radio("항목", ["통화", "증시", "금리", "경제"], key="menu_category")

if category == "통화":
    show_currency_section()
elif category == "증시":
    show_equity_section()
elif category == "금리":
    show_bond_section()
elif category == "경제":
    show_economy_section()
