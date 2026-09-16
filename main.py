import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------------------------------------------
# 기본 설정 (탭 제목, 아이콘, 화면 제목)
# ---------------------------------------------------
st.set_page_config(page_title="영화 유형 나누기", page_icon="🎬")
st.title("🎬 영화 유형 나누기")

# ---------------------------------------------------
# 데이터 불러오기
# ---------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df

df_raw = load_data()

# ---------------------------------------------------
# 속성 만들기
# 1) 스크린 수, 누적 관객 -> 상용로그
# 2) 10위권 일수 -> 그대로
# 3) 롱런 지수 = 누적 관객 / 첫 주 관객, 20 초과시 20으로 자름
# ---------------------------------------------------
df = df_raw.copy()

# 첫 주 관객이 0이거나 네 개 원본 값 중 결측이 있는 영화 제외
required_cols = ["first_scrn", "total_audi", "days_in_top10", "first_week_audi"]
df = df.dropna(subset=required_cols)
df = df[df["first_week_audi"] != 0]

df["log_first_scrn"] = np.log10(df["first_scrn"])
df["log_total_audi"] = np.log10(df["total_audi"])
df["days_in_top10_feat"] = df["days_in_top10"]
df["long_run_index"] = df["total_audi"] / df["first_week_audi"]
df["long_run_index"] = df["long_run_index"].clip(upper=20)

# 로그 계산 등에서 생길 수 있는 무한대/결측 제거
feature_cols_all = ["log_first_scrn", "log_total_audi", "days_in_top10_feat", "long_run_index"]
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols_all)

# 사람이 읽기 좋은 이름 매핑
feature_label = {
    "log_first_scrn": "스크린 수(로그)",
    "log_total_audi": "누적 관객(로그)",
    "days_in_top10_feat": "10위권 일수",
    "long_run_index": "롱런 지수",
}

# ---------------------------------------------------
# 전체 편수 / 묶은 편수 안내
# ---------------------------------------------------
total_count = len(df_raw)
used_count = len(df)
st.write(f"전체 편수: {total_count}편 / 묶은 편수: {used_count}편")

# ---------------------------------------------------
# 속성 선택 (2개 이상, 기본 4개 전부)
# ---------------------------------------------------
st.subheader("묶는 데 사용할 속성 고르기")
selected_features = st.multiselect(
    "군집화에 사용할 속성을 골라 주세요 (최소 2개)",
    options=list(feature_label.keys()),
    default=list(feature_label.keys()),
    format_func=lambda x: feature_label[x],
)

if len(selected_features) < 2:
    st.warning("속성을 2개 이상 선택해 주세요.")
    st.stop()

# ---------------------------------------------------
# 묶음 수 고르기
# ---------------------------------------------------
st.subheader("묶음 수 고르기")
n_clusters = st.slider("묶음 수를 골라 주세요", min_value=2, max_value=7, value=3, step=1)

# 묶음 기호 (최대 7개까지 지원)
cluster_symbols_all = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]
cluster_display_order = cluster_symbols_all[:n_clusters]

# ---------------------------------------------------
# 표준화
# ---------------------------------------------------
X = df[selected_features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------
# 고른 묶음 수로 K-means (난수 고정)
# ---------------------------------------------------
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
raw_labels = kmeans.fit_predict(X_scaled)
df["cluster_raw"] = raw_labels

# ---------------------------------------------------
# 누적 관객 평균이 큰 묶음부터 기호 부여
# ---------------------------------------------------
cluster_order = (
    df.groupby("cluster_raw")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)
label_map = {raw: cluster_display_order[i] for i, raw in enumerate(cluster_order)}
df["cluster"] = df["cluster_raw"].map(label_map)

# ---------------------------------------------------
# 2차원 산점도
# ---------------------------------------------------
st.subheader("2차원 산점도")
col1, col2 = st.columns(2)
with col1:
    x_axis_2d = st.selectbox(
        "가로축", options=selected_features, format_func=lambda x: feature_label[x], key="x2d"
    )
with col2:
    y_axis_2d = st.selectbox(
        "세로축",
        options=selected_features,
        format_func=lambda x: feature_label[x],
        index=min(1, len(selected_features) - 1),
        key="y2d",
    )

fig_2d = px.scatter(
    df,
    x=x_axis_2d,
    y=y_axis_2d,
    color="cluster",
    category_orders={"cluster": cluster_display_order},
    hover_name="movieNm",
    labels={x_axis_2d: feature_label[x_axis_2d], y_axis_2d: feature_label[y_axis_2d]},
    title="영화 유형 2차원 산점도",
)
st.plotly_chart(fig_2d, use_container_width=True)

# ---------------------------------------------------
# 3차원 산점도
# ---------------------------------------------------
st.subheader("3차원 산점도")

if len(selected_features) < 3:
    st.info("선택한 속성이 3개 미만이라 3차원 산점도를 그릴 수 없습니다. 속성을 3개 이상 선택해 주세요.")
else:
    col3, col4, col5 = st.columns(3)
    with col3:
        x_axis_3d = st.selectbox(
            "X축", options=selected_features, format_func=lambda x: feature_label[x], key="x3d"
        )
    with col4:
        y_axis_3d = st.selectbox(
            "Y축",
            options=selected_features,
            format_func=lambda x: feature_label[x],
            index=min(1, len(selected_features) - 1),
            key="y3d",
        )
    with col5:
        z_axis_3d = st.selectbox(
            "Z축",
            options=selected_features,
            format_func=lambda x: feature_label[x],
            index=min(2, len(selected_features) - 1),
            key="z3d",
        )

    fig_3d = px.scatter_3d(
        df,
        x=x_axis_3d,
        y=y_axis_3d,
        z=z_axis_3d,
        color="cluster",
        category_orders={"cluster": cluster_display_order},
        hover_name="movieNm",
        labels={
            x_axis_3d: feature_label[x_axis_3d],
            y_axis_3d: feature_label[y_axis_3d],
            z_axis_3d: feature_label[z_axis_3d],
        },
        title="영화 유형 3차원 산점도",
    )
    fig_3d.update_traces(marker=dict(size=3))
    st.plotly_chart(fig_3d, use_container_width=True)

# ---------------------------------------------------
# 묶음별 편수와 네 속성 평균(원래 단위)
# ---------------------------------------------------
st.subheader("묶음별 편수와 속성 평균 (원래 단위)")

summary_rows = []
for c in cluster_display_order:
    sub = df[df["cluster"] == c]
    row = {
        "묶음": c,
        "편수": len(sub),
        "스크린 수 평균": sub["first_scrn"].mean(),
        "누적 관객 평균": sub["total_audi"].mean(),
        "10위권 일수 평균": sub["days_in_top10"].mean(),
        "롱런 지수 평균": sub["long_run_index"].mean(),
    }
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df, use_container_width=True)

# ---------------------------------------------------
# 묶음별 누적 관객 상위 5편
# ---------------------------------------------------
st.subheader("묶음별 누적 관객 상위 5편")

for c in cluster_display_order:
    sub = df[df["cluster"] == c].sort_values("total_audi", ascending=False).head(5)
    st.markdown(f"**{c} 묶음**")
    st.write(", ".join(sub["movieNm"].tolist()))

# ---------------------------------------------------
# 엘보우 방법: 묶음 수 1~7에 대한 관성(inertia) 계산
# ---------------------------------------------------
st.subheader("묶음 수에 따른 관성(inertia) 변화")

k_range = list(range(1, 8))
inertias = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig_elbow = go.Figure()
fig_elbow.add_trace(
    go.Scatter(
        x=k_range,
        y=inertias,
        mode="lines+markers",
        name="관성",
    )
)
# 현재 고른 묶음 수 위치에 세로선
fig_elbow.add_vline(x=n_clusters, line_dash="dash", line_color="red")
fig_elbow.update_layout(
    title="묶음 수별 관성 변화 (엘보우 그래프)",
    xaxis_title="묶음 수",
    yaxis_title="관성 (거리 제곱 합)",
    xaxis=dict(tickmode="linear", tick0=1, dtick=1),
)
st.plotly_chart(fig_elbow, use_container_width=True)

# ---------------------------------------------------
# 묶음 수마다 관성 값과 이전 값 대비 감소량 표
# ---------------------------------------------------
st.subheader("묶음 수별 관성 값과 감소량")

decrease_list = [None]
for i in range(1, len(inertias)):
    decrease_list.append(inertias[i - 1] - inertias[i])

elbow_table = pd.DataFrame(
    {
        "묶음 수": k_range,
        "관성 값": inertias,
        "이전보다 줄어든 양": decrease_list,
    }
)
st.dataframe(elbow_table, use_container_width=True)

# ---------------------------------------------------
# 지금 고른 묶음 수의 실루엣 점수
# ---------------------------------------------------
sil_score = silhouette_score(X_scaled, raw_labels)
st.write(f"지금 고른 묶음 수({n_clusters}개)의 실루엣 점수: {sil_score:.4f} (범위 -1~1, 1에 가까울수록 묶음이 뚜렷함)")
