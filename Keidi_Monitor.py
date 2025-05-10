# ----------------------------
# 1. 기본 설정 및 라이브러리 불러오기
# ----------------------------
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import requests
from datetime import datetime, timedelta

st.set_page_config(page_title="KEIDI Monitor", layout="wide")
st.title("KEIDI Monitor")
st.markdown("글로벌 자산(통화, 증시, 금리)을 모니터링하는 웹 대시보드입니다.")


# ----------------------------
# 2. Plotly 차트 유틸리티 및 공통 설정 함수
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
        modebar_remove=['zoom', 'pan', 'select', 'lasso2d',
                        'zoomIn', 'zoomOut', 'resetScale'],
        modebar_add=['toImage', 'fullscreen', 'autoScale', 'toggleSpikelines']
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
        modebar_remove=['zoom', 'pan', 'select', 'lasso2d',
                        'zoomIn', 'zoomOut', 'resetScale'],
        modebar_add=['toImage', 'fullscreen', 'autoScale', 'toggleSpikelines'],
        
    )
    fig.update_traces(mode="lines", hoverinfo="all")
    return fig

def get_common_xaxis_layout():
    today = datetime.today()
    one_year_ago = today - timedelta(days=365)
    return dict(
        range=[one_year_ago, today],
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=3, label="3y", step="year", stepmode="backward"),
                dict(count=5, label="5y", step="year", stepmode="backward"),
                dict(step="all", label="전체")
            ]
        ),
        rangeslider=dict(visible=True),
        type="date"
    )


# ----------------------------
# 3. 데이터 수집 함수
# ----------------------------
def get_yf_data(ticker, label, period="10y"):
    ticker_obj = yf.Ticker(ticker)
    hist = ticker_obj.history(period=period)
    if hist.empty:
        st.error(f"{label} 데이터 없음")
        return pd.DataFrame()
    df = hist.reset_index()[["Date", "Close"]]
    df.rename(columns={"Date": "날짜", "Close": label}, inplace=True)
    return df

@st.cache_data(ttl=3600)
def get_fred_data(series_id, start_date=None):
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
    if "observations" not in data:
        st.error(f"FRED API 오류: {data.get('error_message', '알 수 없는 오류')}")
        return pd.DataFrame()

    df = pd.DataFrame(data["observations"])
    df["날짜"] = pd.to_datetime(df["date"])
    df[series_id] = pd.to_numeric(df["value"], errors="coerce")
    return df[["날짜", series_id]]

def add_moving_averages(df: pd.DataFrame, label: str, windows=[50, 200]):
    for window in windows:
        ma_col = f"{label}_MA{window}"
        df[ma_col] = df[label].rolling(window=window).mean()
    return df

# ----------------------------
# 4. 이동평균 포함 시계열 차트 생성
# ----------------------------
def plot_with_moving_averages(df, label):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["날짜"], y=df[label],
        mode="lines",
        name=label,
        line=dict(color="white", width=2)
    ))

    for window in [50, 200]:
        ma_col = f"{label}_MA{window}"
        if ma_col in df.columns:
            fig.add_trace(go.Scatter(
                x=df["날짜"], y=df[ma_col],
                mode="lines",
                name=f"MA{window}",
                line=dict(dash="dot", width=1.5)
            ))

    fig = make_interactive_plotly(fig)
    fig.update_layout(
        xaxis=get_common_xaxis_layout(),
        yaxis=dict(autorange=True),
        uirevision=None
    )
    return fig
# ----------------------------
# 4. 사용자 선택에 따른 시계열 필터 함수
# ----------------------------
def filter_df_by_period(df: pd.DataFrame, selected_period: str):
    df["날짜"] = df["날짜"].dt.tz_localize(None)
    today = datetime.today()
    if selected_period == "1년":
        cutoff = today - timedelta(days=365)
    elif selected_period == "3년":
        cutoff = today - timedelta(days=365 * 3)
    elif selected_period == "5년":
        cutoff = today - timedelta(days=365 * 5)
    else:
        return df
    return df[df["날짜"] >= cutoff]


# ----------------------------
# 5. 카테고리별 시각화 함수
# ----------------------------
def show_currency_section():
    st.header("💱 통화 시장")
    selected_period = st.radio("📆 차트 기간 선택:", ["1년", "3년", "5년", "전체"], horizontal=True, key="currency_period")

    pairs = [
        ("DX-Y.NYB", "DXY"),
        ("EURUSD=X", "EUR/USD"),
        ("KRW=X", "USD/KRW"),
        ("JPYKRW=X", "JPY/KRW"),
        ("EURKRW=X", "EUR/KRW")
    ]

    # DXY는 단독 출력
    ticker, label = pairs[0]
    df = get_yf_data(ticker, label)
    if not df.empty:
        df = add_moving_averages(df, label)
        df = filter_df_by_period(df, selected_period)
        fig = plot_with_moving_averages(df, label)
        st.plotly_chart(fig, use_container_width=True)

    for i in range(1, len(pairs), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(pairs):
                ticker, label = pairs[i + j]
                df = get_yf_data(ticker, label)
                if not df.empty:
                    df = add_moving_averages(df, label)
                    df = filter_df_by_period(df, selected_period)
                    fig = plot_with_moving_averages(df, label)
                    cols[j].plotly_chart(fig, use_container_width=True)

def show_equity_section():
    st.header("📈 글로벌 증시")
    st.markdown("아래에서 차트 기간을 선택하세요. 선택한 기간에 따라 y축도 자동 조정됩니다.")
    
    selected_period = st.radio("📆 차트 기간 선택:", ["1년", "3년", "5년", "전체"], horizontal=True)

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
            df = add_moving_averages(df, label)
            df = filter_df_by_period(df, selected_period)
            fig = plot_with_moving_averages(df, label)
            st.plotly_chart(fig, use_container_width=True)

def show_bond_section():
    st.header("💵 미국 국채 금리")
    selected_period = st.radio("📆 차트 기간 선택:", ["1년", "3년", "5년", "전체"], horizontal=True, key="bond_period")

    tickers = {
        "3개월": "^IRX", "2년": "^FVX", "10년": "^TNX"
    }

    def get_multi_bond_df(period_code):
        df_all = pd.DataFrame()
        for label, ticker in tickers.items():
            df = get_yf_data(ticker, label)
            if not df.empty:
                df["날짜"] = df["날짜"].dt.tz_localize(None)
                df = filter_df_by_period(df, period_code)
                if df_all.empty:
                    df_all = df
                else:
                    df_all = pd.merge(df_all, df, on="날짜", how="outer")
        return df_all.dropna()

    df = get_multi_bond_df(selected_period)
    if df.empty:
        st.warning("해당 기간의 금리 데이터를 불러올 수 없습니다.")
        return

    with st.expander(f"📈 미국채 금리 추세", expanded=True):
        fig1 = px.line(df, x="날짜", y=["3개월", "2년", "10년"],
                       labels={"value": "금리", "variable": "만기"},
                       title="미국채 금리")
        fig1.update_layout(yaxis=dict(autorange=True))
        st.plotly_chart(make_static_plotly(fig1), use_container_width=True)

    df["10y-2y"] = df["10년"] - df["2년"]
    df["10y-3m"] = df["10년"] - df["3개월"]

    with st.expander(f"📉 금리 스프레드", expanded=True):
        fig2 = px.line(df, x="날짜", y=["10y-2y", "10y-3m"],
                       labels={"value": "스프레드", "variable": "구간"},
                       title="미국채 금리 스프레드")
        fig2.update_layout(yaxis=dict(autorange=True))
        st.plotly_chart(make_static_plotly(fig2), use_container_width=True)

def show_economy_section():
    st.header("🌎 글로벌 경제 지표")
    selected_period = st.radio("📆 차트 기간 선택:", ["1년", "3년", "5년", "전체"], horizontal=True, key="econ_period")

    # 1️⃣ 미국 실업률
    for series_id, label in [("UNRATE", "미국 실업률 (%)")]:
        df = get_fred_data(series_id)  # 최장기간 자동 적용
        if df.empty:
            st.warning(f"{label} 데이터를 불러올 수 없습니다.")
            continue

        df = filter_df_by_period(df, selected_period)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["날짜"], y=df[series_id],
            mode="lines", name=label, line=dict(color="royalblue")
        ))
        fig = make_interactive_plotly(fig)
        fig.update_layout(title=label)
        st.plotly_chart(fig, use_container_width=True)

    # 2️⃣ Real Expenditure (RSAFS / CPIAUCSL)
    rsafs = get_fred_data("RSAFS")
    cpiaucsl = get_fred_data("CPIAUCSL")
    if not rsafs.empty and not cpiaucsl.empty:
        df = pd.merge(rsafs, cpiaucsl, on="날짜", how="inner")
        df["실질소비"] = df["RSAFS"] / df["CPIAUCSL"]
        df["MA12"] = df["실질소비"].rolling(window=12).mean()
        df = filter_df_by_period(df, selected_period)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["날짜"], y=df["실질소비"], name="실질 소비 지수"
        ))
        fig.add_trace(go.Scatter(
            x=df["날짜"], y=df["MA12"], name="12개월 이동평균", line=dict(dash="dot")
        ))
        fig = make_interactive_plotly(fig)
        fig.update_layout(
            title="🇺🇸 미국 실질 소비 추이 (Real Expenditure)",
            yaxis_title="지수 (RSAFS / CPIAUCSL)"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("실질 소비 지표를 불러올 수 없습니다.")

    # 3️⃣ CPI + 전년동월대비
    cpi = get_fred_data("CPIAUCSL")
    if not cpi.empty:
        df = cpi.copy()
        df["YoY"] = df["CPIAUCSL"].pct_change(12) * 100
        df = df.dropna()
        df = filter_df_by_period(df, selected_period)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=df["날짜"], y=df["CPIAUCSL"],
            name="CPI 지수", line=dict(color="darkblue")
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=df["날짜"], y=df["YoY"],
            name="전년동월대비 상승률 (%)", line=dict(color="crimson", dash="dot")
        ), secondary_y=True)

        fig = make_interactive_plotly(fig)
        fig.update_layout(
            title="🇺🇸 미국 소비자물가지수 (CPI) 및 전년동월비 상승률",
            yaxis=dict(title="CPI 지수"),
            yaxis2=dict(title="YoY 상승률 (%)", overlaying="y", side="right")
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("CPI 데이터를 불러올 수 없습니다.")

# ----------------------------
# 6. 사이드바 메뉴 및 라우팅
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
