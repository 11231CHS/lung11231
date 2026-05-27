import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# -----------------------------
# 폰트 설정 (온라인 호환)
# -----------------------------
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------
# 페이지 설정
# -----------------------------
st.set_page_config(
    page_title="폐암 환자 군집 분석",
    page_icon="🫁",
    layout="wide"
)

# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>

/* 카드 */
.card {
    background: rgba(255,255,255,0.85);
    padding: 24px;
    border-radius: 24px;
    border: 1px solid rgba(20,184,166,0.15);
    box-shadow: 0 8px 24px rgba(15,23,42,0.08);
    margin-bottom: 20px;
}

/* 사이드바 */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff, #f0fdfa);
}

/* 버튼 */
.stButton>button {
    background: linear-gradient(135deg,#14b8a6,#0f766e);
    color: white;
    border: none;
    border-radius: 14px;
}

/* metric */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.85);
    border-radius: 18px;
    padding: 12px;
    border: 1px solid rgba(20,184,166,0.12);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# 모델 & 데이터
# -----------------------------
model = joblib.load("lung_model.pkl")
scaler = joblib.load("lung_scaler.pkl")
df = pd.read_csv("lung.csv")

# cluster 컬럼 없으면 자동 생성
if 'cluster' not in df.columns:
    df['cluster'] = model.predict(
        scaler.transform(df[['흡연', '알코올', '나이']])
    )

# -----------------------------
# 제목
# -----------------------------
st.title("🫁 폐암 환자 군집 분석 시스템")
st.caption("생활 습관 데이터를 기반으로 환자 군집을 분석합니다.")

# -----------------------------
# 사이드바
# -----------------------------
st.sidebar.header("환자 정보 입력")

smoke = st.sidebar.slider("🚬 흡연 정도", 0.0, 10.0, 3.0, 0.1)
alc = st.sidebar.slider("🍺 음주 정도", 0.0, 10.0, 2.0, 0.1)
age = st.sidebar.slider("🧓 나이", 20, 90, 50)

# -----------------------------
# 새 환자 데이터
# -----------------------------
new_patient = pd.DataFrame(
    [[smoke, alc, age]],
    columns=['흡연', '알코올', '나이']
)

# -----------------------------
# 예측
# -----------------------------
scaled = scaler.transform(new_patient)
pred_cluster = model.predict(scaled)[0]

# -----------------------------
# 위험도 계산
# -----------------------------
risk_percent = {
    0: 10,
    1: 45,
    2: 25,
    3: 85
}

risk = risk_percent[pred_cluster]
health_score = 100 - risk

# -----------------------------
# 군집 정보
# -----------------------------
cluster_name = {
    0: "🟢 매우 건강군",
    1: "🟡 중간 그룹",
    2: "🔵 건강군",
    3: "🔴 강한 위험군"
}

cluster_desc = {
    0: "흡연 및 음주 수치가 매우 낮고 건강 상태가 안정적인 그룹입니다.",
    1: "일부 위험 요인이 존재하지만 평균 수준의 건강 상태를 보이는 그룹입니다.",
    2: "전반적으로 건강하지만 생활 습관 관리가 필요한 그룹입니다.",
    3: "흡연·음주 위험도가 높아 폐 건강 관리가 중요한 고위험 그룹입니다."
}

doctor_tip = {
    0: "현재 생활 습관이 매우 안정적입니다. 규칙적인 운동을 유지하세요.",
    1: "흡연과 음주 빈도를 조금 줄이면 건강 유지에 도움이 됩니다.",
    2: "현재 상태는 양호하지만 꾸준한 건강 관리가 중요합니다.",
    3: "폐 건강 위험도가 높습니다. 정기 검진과 생활 습관 개선이 필요합니다."
}

# -----------------------------
# 결과 카드
# -----------------------------
st.subheader("📊 분석 결과")

with st.container():

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.metric(
        "예측 군집",
        f"{pred_cluster}번 군집"
    )

    st.success(cluster_name[pred_cluster])

    st.write(cluster_desc[pred_cluster])

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# 건강도 게이지
# -----------------------------
st.subheader("💚 건강도 분석")

g1, g2 = st.columns(2)

with g1:

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health_score,
        title={'text': "건강 점수"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [0, 40], 'color': "#ffcccc"},
                {'range': [40, 70], 'color': "#fff3cd"},
                {'range': [70, 100], 'color': "#d4edda"}
            ]
        }
    ))

    fig_gauge.update_layout(height=320)

    st.plotly_chart(fig_gauge, use_container_width=True)

with g2:

    st.metric("⚠️ 폐 건강 위험도", f"{risk}%")
    st.metric("💪 건강 점수", f"{health_score}점")

    if risk < 30:
        st.success("위험도가 낮은 상태입니다.")
    elif risk < 70:
        st.warning("생활 습관 관리가 필요합니다.")
    else:
        st.error("고위험 상태입니다. 검진 권장.")

# -----------------------------
# 그래프 + 특징
# -----------------------------
col1, col2 = st.columns([1.1, 1])

with col1:

    st.subheader("📈 군집 시각화")

    fig, ax = plt.subplots(figsize=(5,4))

    ax.scatter(
        df['흡연'],
        df['알코올'],
        c=df['cluster'],
        alpha=0.5
    )

    ax.scatter(
        smoke,
        alc,
        c='black',
        s=220,
        marker='X',
        label='새 환자'
    )

    ax.set_xlabel("흡연")
    ax.set_ylabel("알코올")
    ax.set_title("폐암 환자 군집")

    ax.legend()

    st.pyplot(fig)

with col2:

    st.subheader("📌 군집 특징")

    st.info("🟢 0번 군집 : 매우 건강한 생활 패턴")
    st.info("🟡 1번 군집 : 평균적인 위험도")
    st.info("🔵 2번 군집 : 비교적 건강한 그룹")
    st.info("🔴 3번 군집 : 높은 위험도를 가진 그룹")

# -----------------------------
# AI 건강 조언
# -----------------------------
st.subheader("👨‍⚕️ AI 건강 조언")

c1, c2 = st.columns([1, 6])

with c1:
    st.markdown(
        """
        <div style='font-size:90px; text-align:center;'>
        🧑‍⚕️
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("### AI 의료 분석 리포트")

    st.write(doctor_tip[pred_cluster])

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# 입력 정보
# -----------------------------
st.subheader("🧾 입력된 환자 정보")

st.markdown('<div class="card">', unsafe_allow_html=True)

st.write(f"🚬 흡연 정도 : {smoke}")
st.write(f"🍺 음주 정도 : {alc}")
st.write(f"🧓 나이 : {age}")

st.markdown('</div>', unsafe_allow_html=True)

