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

def make_interactive_plotly(fig):   # Ploty 차트를 동적으로 설정합니다. 호버시 십자선 표시, 스냅형 정보 제공을 합니다. 확대/축소, 이동 등 상호작용 기능을 제거합니다. 선 스타일을 지정합니다.
    fig.update_layout(
        dragmode=False,
        hovermode="x unified",
        xaxis_showspikes=True,
        yaxis_showspikes=True,
        xaxis=dict(fixedrange=True, spikemode="across", spikethickness=1),
        yaxis=dict(fixedrange=True, spikemode="across", spikethickness=1),
        showlegend=True,
        modebar_remove=['zoom', 'pan', 'select', 'lasso2d',
                        'zoomIn', 'zoomOut', 'resetScale'],
        modebar_add=['toImage', 'fullscreen', 'autoScale', 'toggleSpikelines'],
        
    )
    fig.update_traces(mode="lines", hoverinfo="all", line=dict(width=2))
    return fig

# ----------------------------
# 3. 데이터 수집 함수
# ----------------------------
def get_yf_data(ticker, label, period="30y"):
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

def plot_with_moving_averages(df, label):   # 데이터프레임에서 'label' 칼럼을 기준으로 원 데이터 및 이동평균(보조 데이터)를 시각화 하는 Plotly 차트를 생성합니다. 
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["날짜"], y=df[label],
        mode="lines",
        name=label,
        line=dict(width=2)  # 주 지표표
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
def generate_time_filtered_dfs(df): # 입력된 전체 시계열 데이터 프레임을 기간별(3년, 10년, 20년, 전체)로 나눈 4개의 서브 데이터 프레임을 딕셔너리 형태로 반환합니다.    
    now = pd.Timestamp.today()
    # 🔧 날짜열에서 타임존 제거 (비교 오류 방지)
    df["날짜"] = df["날짜"].dt.tz_localize(None)
    return {
        "3년": df[df["날짜"] >= now - pd.DateOffset(years=3)],
        "10년": df[df["날짜"] >= now - pd.DateOffset(years=10)],
        "20년": df[df["날짜"] >= now - pd.DateOffset(years=20)],
        "최대": df
    }
def render_static_time_range_charts(df, value_col, title):  # generate_time_filtered_dfs()로 나눈 4개의 기간별 데이터에 대해 각각 plotly 차트를 생성하고 streamlit 앱에 4개의 2*2차트를 출력합니다.
    time_ranges = generate_time_filtered_dfs(df)
    range_labels = list(time_ranges.keys())

    for i in range(0, len(range_labels), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(range_labels):
                range_label = range_labels[i + j]
                subset = time_ranges[range_label]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=subset["날짜"], y=subset[value_col],
                    mode="lines", name=title
                ))
                fig.update_layout(
                    title=f"{title} ({range_label})",
                    hovermode="x unified",
                    dragmode=False,
                    xaxis=dict(fixedrange=True),
                    yaxis=dict(fixedrange=True),
                    margin=dict(l=40, r=40, t=60, b=40)
                )
                config = {
                    "displaylogo": False,
                    "modeBarButtonsToRemove": [
                        "zoom", "pan", "select", "lasso2d",
                        "zoomIn", "zoomOut", "resetScale"
                    ]
                }
                cols[j].plotly_chart(fig, use_container_width=True, config=config)
def render_static_time_range_charts_dual(df, main_col, sub_col, title): # generate_time_filtered_dfs()로 나눈 4개의 기간별 데이터에 대해 주 데이터와 12개월 이동평균 데이터를 각각 plotly 차트를 생성하고 streamlit 앱에 4개의 2*2차트를 출력합니다.
    time_ranges = generate_time_filtered_dfs(df)
    range_labels = list(time_ranges.keys())

    for i in range(0, len(range_labels), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(range_labels):
                range_label = range_labels[i + j]
                subset = time_ranges[range_label]
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=subset["날짜"], y=subset[main_col],
                    mode="lines", name=main_col,
                    line=dict(color="royalblue")
                ))
                fig.add_trace(go.Scatter(
                    x=subset["날짜"], y=subset[sub_col],
                    mode="lines", name=sub_col,
                    line=dict(color="gray", dash="dot")
                ))
                fig.update_layout(
                    title=f"{title} ({range_label})",
                    hovermode="x unified",
                    dragmode=False,
                    xaxis=dict(fixedrange=True),
                    yaxis=dict(fixedrange=True),
                    margin=dict(l=40, r=40, t=60, b=40)
                )
                config = {
                    "displaylogo": False,
                    "modeBarButtonsToRemove": [
                        "zoom", "pan", "select", "lasso2d",
                        "zoomIn", "zoomOut", "resetScale"
                    ]
                }
                cols[j].plotly_chart(fig, use_container_width=True, config=config)
def render_static_dual_axis_charts(df, main_col, sub_col, title, y1_title, y2_title):   # generate_time_filtered_dfs()로 나눈 4개의 기간별 데이터에 대해 주 데이터와 YoY 데이터를 각각 plotly 차트를 (주/보조축)생성하고 streamlit 앱에 4개의 2*2차트를 출력합니다.
    time_ranges = generate_time_filtered_dfs(df)
    range_labels = list(time_ranges.keys())

    for i in range(0, len(range_labels), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(range_labels):
                range_label = range_labels[i + j]
                subset = time_ranges[range_label]

                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(
                    x=subset["날짜"], y=subset[main_col],
                    mode="lines", name=main_col,
                    line=dict(color="darkblue")
                ), secondary_y=False)
                fig.add_trace(go.Scatter(
                    x=subset["날짜"], y=subset[sub_col],
                    mode="lines", name=sub_col,
                    line=dict(color="crimson", dash="dot")
                ), secondary_y=True)

                fig.update_layout(
                    title=f"{title} ({range_label})",
                    hovermode="x unified",
                    dragmode=False,
                    xaxis=dict(fixedrange=True),
                    yaxis=dict(fixedrange=True, title=y1_title),
                    yaxis2=dict(fixedrange=True, title=y2_title, overlaying="y", side="right"),
                    margin=dict(l=40, r=40, t=60, b=40),
                    legend=dict(orientation="h")
                )
                config = {
                    "displaylogo": False,
                    "modeBarButtonsToRemove": [
                        "zoom", "pan", "select", "lasso2d",
                        "zoomIn", "zoomOut", "resetScale"
                    ]
                }
                cols[j].plotly_chart(fig, use_container_width=True, config=config)
# ----------------------------
# 5. 카테고리별 시각화 함수
# ----------------------------
def show_currency_section():
    st.header("💱 통화 시장")

    currency_pairs = [
        ("DX-Y.NYB", "DXY"),
        ("EURUSD=X", "EUR/USD"),
        ("KRW=X", "USD/KRW"),
        ("JPYKRW=X", "JPY/KRW"),
        ("EURKRW=X", "EUR/KRW")
    ]

    for ticker, label in currency_pairs:
        df = get_yf_data(ticker, label)
        if df.empty:
            st.warning(f"{label} 데이터가 없습니다.")
            continue

        # 이동평균선 계산
        df["MA50"] = df[label].rolling(window=50).mean()
        df["MA200"] = df[label].rolling(window=200).mean()
        df = df.dropna(subset=[label, "MA50", "MA200"])

        # 4개의 기간별 데이터 생성
        time_ranges = generate_time_filtered_dfs(df)
        range_labels = list(time_ranges.keys())

        # 2×2 배열로 표시
        st.markdown("<hr style='border:2px solid #888;'>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align:center;'>{label}</h3>", unsafe_allow_html=True)
        for i in range(0, len(range_labels), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(range_labels):
                    range_label = range_labels[i + j]
                    subset = time_ranges[range_label]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=subset["날짜"], y=subset[label],
                        mode="lines", name=label,
                        line=dict(color="black", width=1)
                    ))
                    fig.add_trace(go.Scatter(
                        x=subset["날짜"], y=subset["MA50"],
                        mode="lines", name="MA50",
                        line=dict(color="blue", width=0.5)
                    ))
                    fig.add_trace(go.Scatter(
                        x=subset["날짜"], y=subset["MA200"],
                        mode="lines", name="MA200",
                        line=dict(color="red", width=0.5)
                    ))

                    fig.update_layout(
                        title=f"{label} ({range_label})",
                        hovermode="x unified",
                        dragmode=False,
                        xaxis=dict(fixedrange=True),
                        yaxis=dict(fixedrange=True),
                        margin=dict(l=40, r=40, t=60, b=40),
                        legend=dict(
                            orientation="h",
                            x=0, y=1,
                            xanchor="left", yanchor="top",)
                    )

                    config = {
                        "displaylogo": False,
                        "modeBarButtonsToRemove": [
                            "zoom", "pan", "select", "lasso2d",
                            "zoomIn", "zoomOut", "resetScale"
                        ]
                    }

                    cols[j].plotly_chart(fig, use_container_width=True, config=config)

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
        st.plotly_chart(make_interactive_plotly(fig1), use_container_width=True)

    df["10y-2y"] = df["10년"] - df["2년"]
    df["10y-3m"] = df["10년"] - df["3개월"]

    with st.expander(f"📉 금리 스프레드", expanded=True):
        fig2 = px.line(df, x="날짜", y=["10y-2y", "10y-3m"],
                       labels={"value": "스프레드", "variable": "구간"},
                       title="미국채 금리 스프레드")
        fig2.update_layout(yaxis=dict(autorange=True))
        st.plotly_chart(make_interactive_plotly(fig2), use_container_width=True)

def show_economy_section():
    st.header("🌎 글로벌 경제 지표")
    selected_period = st.radio("📆 차트 기간 선택:", ["1년", "3년", "5년", "전체"], horizontal=True, key="econ_period")

    # 필요한 모든 FRED 데이터 미리 가져오기
    fred_series = {
        "UNRATE": get_fred_data("UNRATE"),
        "RSAFS": get_fred_data("RSAFS"),
        "CPIAUCSL": get_fred_data("CPIAUCSL"),
        "PCEPILFE": get_fred_data("PCEPILFE"),
        "BAMLH0A0HYM2": get_fred_data("BAMLH0A0HYM2")

    }

    # 1️⃣ 미국 실업률
    df = fred_series["UNRATE"]
    if not df.empty:
        render_static_time_range_charts(df, "UNRATE", "🇺🇸 미국 실업률 (%)")
    else:
        st.warning("실업률 데이터를 불러올 수 없습니다.")

    # 2️⃣ Real Expenditure (RSAFS / CPIAUCSL)
    rsafs = fred_series["RSAFS"]
    cpi = fred_series["CPIAUCSL"]
    if not rsafs.empty and not cpi.empty:
        df = pd.merge(rsafs, cpi, on="날짜", how="inner")
        df["실질소비"] = df["RSAFS"] / df["CPIAUCSL"]
        df["MA12"] = df["실질소비"].rolling(window=12).mean()
        df = df.dropna(subset=["실질소비", "MA12"])

        render_static_time_range_charts_dual(df, "실질소비", "MA12", "🇺🇸 실질 소비 지수 (Real Expenditure)")
    else:
        st.warning("실질 소비 데이터를 불러올 수 없습니다.")

    # 3️⃣ CPI + 전년동월대비
    cpi = fred_series["CPIAUCSL"]
    if not cpi.empty:
        df = cpi.copy()
        df["YoY"] = df["CPIAUCSL"].pct_change(12) * 100
        df = df.dropna(subset=["CPIAUCSL", "YoY"])

        render_static_dual_axis_charts(
            df,
            main_col="CPIAUCSL",
            sub_col="YoY",
            title="🇺🇸 소비자물가지수 및 전년동월비",
            y1_title="CPI 지수",
            y2_title="YoY 상승률 (%)"
        )
    else:
        st.warning("CPI 데이터를 불러올 수 없습니다.")
    # 3️⃣ Core PCE + 전년동월대비
    PCEPILFE = fred_series["PCEPILFE"]
    if not PCEPILFE.empty:
        df = PCEPILFE.copy()
        df["YoY"] = df["PCEPILFE"].pct_change(12) * 100
        df = df.dropna(subset=["PCEPILFE", "YoY"])

        render_static_dual_axis_charts(
            df,
            main_col="PCEPILFE",
            sub_col="YoY",
            title="🇺🇸 개인 소비지출 및 전년동월비",
            y1_title="PCE 지수",
            y2_title="YoY 상승률 (%)"
        )
    else:
        st.warning("PCE 데이터를 불러올 수 없습니다.")
    # 4️⃣ ICE BofA US High Yield OAS (시계열 + 현재위치 비교)
    BAMLH0A0HYM2 = fred_series["BAMLH0A0HYM2"]
    if not BAMLH0A0HYM2.empty:
        df_filtered = filter_df_by_period(BAMLH0A0HYM2, selected_period)

        col1, col2 = st.columns(2)

        # 📈 왼쪽: 시계열 차트
        with col1:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=df_filtered["날짜"], y=df_filtered["BAMLH0A0HYM2"],
                mode="lines", name="하이일드", line=dict(color="orange")
            ))
            fig1 = make_interactive_plotly(fig1)
            fig1.update_layout(
                title="ICE BofA 미국 하이일드 스프레드 (시계열)",
                yaxis_title="스프레드 (bp)"
            )
            st.plotly_chart(fig1, use_container_width=True)

        # 📊 오른쪽: 위치 막대 차트
        with col2:
            min_val = BAMLH0A0HYM2["BAMLH0A0HYM2"].min()
            max_val = BAMLH0A0HYM2["BAMLH0A0HYM2"].max()
            avg_val = BAMLH0A0HYM2["BAMLH0A0HYM2"].mean()
            curr_val = BAMLH0A0HYM2["BAMLH0A0HYM2"].iloc[-1]

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                y=[max_val - min_val],
                base=min_val,
                marker_color="lightgray",
                opacity=0.6,
                width=0.4,
                name="과거 범위"
            ))
            fig2.add_trace(go.Scatter(
                x=["하이일드 스프레드"],
                y=[avg_val],
                mode="markers+text",
                name="평균값",
                marker=dict(symbol="square", size=12, color="dodgerblue"),
                text=[f"평균: {avg_val:.2f}"],
                textposition="top center"
            ))
            fig2.add_trace(go.Scatter(
                x=["하이일드 스프레드"],
                y=[curr_val],
                mode="markers+text",
                name="현재값",
                marker=dict(symbol="diamond", size=14, color="orange"),
                text=[f"현재: {curr_val:.2f}"],
                textposition="bottom center"
            ))
            fig2.update_layout(
                title="📊 역사적 수준 대비 현재 하이일드 스프레드",
                yaxis_title="스프레드 (bp)",
                barmode="overlay",
                height=600
            )
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("하이일드 스프레드 데이터를 불러올 수 없습니다.")

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
