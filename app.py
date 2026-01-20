import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="EV ML Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------ LOAD ARTIFACTS ------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("ev_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
.metric-card {
    background-color: #1e293b;
    padding: 18px;
    border-radius: 14px;
    text-align: center;
}
.metric-title {
    font-size: 13px;
    color: #cbd5e1;
}
.metric-value {
    font-size: 26px;
    font-weight: bold;
    color: #22c55e;
}
@media (max-width: 768px) {
    .metric-value {
        font-size: 20px;
    }
}
</style>
""", unsafe_allow_html=True)

# ------------------ NAVIGATION ------------------
page = st.sidebar.radio(
    "🧭 Navigation",
    ["🏠 Overview", "🤖 Prediction", "📊 Analytics"]
)

# ------------------ SIDEBAR INPUTS ------------------
st.sidebar.header("🔧 Vehicle Inputs")

input_data = {}
validation_errors = []

for feature in feature_names:
    if feature.lower() == "cluster":
        input_data[feature] = st.sidebar.selectbox(feature, [0, 1, 2])
    else:
        val = st.sidebar.number_input(feature, value=0.0, step=1.0)
        if val < 0:
            validation_errors.append(f"{feature} cannot be negative")
        input_data[feature] = val

# ------------------ OVERVIEW PAGE ------------------
if page == "🏠 Overview":
    st.title("⚡ Electric Vehicle ML Dashboard")
    st.write(
        "This application predicts **EV Price, Market Demand, and Sales Count** "
        "using a trained machine learning model."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.info("✔ Built with Streamlit\n✔ Deployed via GitHub\n✔ Production-ready ML pipeline")

    with col2:
        st.success("Status: Online")

# ------------------ PREDICTION PAGE ------------------
elif page == "🤖 Prediction":
    st.title("🤖 EV Prediction")

    if validation_errors:
        for err in validation_errors:
            st.error(err)
        st.stop()

    if st.button("🚀 Run Prediction"):
        with st.spinner("Running model..."):
            df_input = pd.DataFrame([input_data])
            df_scaled = scaler.transform(df_input)
            prediction = model.predict(df_scaled)

        price, demand, count = prediction[0]

        # ---- Model Confidence (Heuristic) ----
        confidence = max(0.65, min(0.95, 1 - np.std(df_scaled) / 5))

        st.subheader("📈 Results")

        col1, col2, col3 = st.columns(3)

        col1.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Predicted Price ($)</div>
            <div class="metric-value">{price:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        col2.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Predicted Demand</div>
            <div class="metric-value">{demand:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)

        col3.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Sales Count</div>
            <div class="metric-value">{count:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(confidence)
        st.caption(f"📈 Model Confidence: **{confidence*100:.1f}%**")

# ------------------ ANALYTICS PAGE ------------------
elif page == "📊 Analytics":
    st.title("📊 Price vs Demand Analysis")

    if st.button("Generate Chart"):
        df_input = pd.DataFrame([input_data])
        df_scaled = scaler.transform(df_input)
        prediction = model.predict(df_scaled)

        price, demand, _ = prediction[0]

        fig, ax = plt.subplots()
        ax.scatter(price, demand)
        ax.set_xlabel("Predicted Price ($)")
        ax.set_ylabel("Predicted Demand")
        ax.set_title("Price vs Demand")
        st.pyplot(fig)

    st.info("Use this section to visually analyze how pricing impacts demand.")

