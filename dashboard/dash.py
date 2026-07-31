import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
import joblib
from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go
import plotly.express as px

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# API configuration – read from environment (Docker) or fallback to localhost
# ---------------------------------------------------------------------------
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# Custom CSS
st.markdown("""
    <style>
    /* Layout Styles */
    .main {
        max-width: 100% !important;
    }
    [data-testid="stAppViewContainer"] {
        padding: 0 2rem;
    }
    .stMarkdown {
        width: 100%;
    }
    
    /* Metric Colors */
    .metric-blue .metric-value {
        color: #0066FF !important;
    }
    .metric-red .metric-value {
        color: #FF0000 !important;
    }
    .metric-green .metric-value {
        color: #00AA00 !important;
    }
    
    /* Table Styles */
    .table-container {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #ddd;
        border-radius: 5px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
    }
    th {
        background-color: #8f8f8f !important;
        color: white !important;
        font-weight: bold;
        padding: 10px;
        text-align: left;
        border: 1px solid #ddd;
        position: sticky;
        top: 0;
    }
    td {
        background-color: white !important;
        padding: 10px;
        border: 1px solid #ddd;
    }
    .sick {
        background-color: #ffcccc !important;
    }
    .healthy {
        background-color: #ccebff !important;
    }
    
    /* Card Styles */
    .card {
        background-color: #F0F2F6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    
    .card-title {
        font-size : 20px;
    }

    .card-number {
        font-size: 40px;
        font-weight: bold;
        display: block;
        margin-top: 10px;
    }
            
    .card-number2{
        font-weight: bold;
    }
            
    /*Highlight for key insights*/
    .highlight-blue {
        background-color: #d0ebff;   
        color: #084c61;           
        font-weight: bold;
        padding: 2px 4px;
        border-radius: 4px;
    }
            
    .highlight-red {
        background-color: #ffd6d6; 
        color: #a30000;              
        font-weight: bold;
        padding: 2px 4px;
        border-radius: 4px;
    }
            
    .highlight {
        background-color: #D6FFD6; 
        color: #0D610E;              
        font-weight: bold;
        padding: 2px 4px;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🫀 Heart Disease Prediction Dashboard")

# Load dataset
path_raw = os.path.join(BASE_DIR, "..", "Data", "heart.csv")
path_cleaned = os.path.join(BASE_DIR, "..", "Data", "heart_cleaned.csv")

df = pd.read_csv(path_cleaned)
df_r = pd.read_csv(path_raw)

st.set_page_config(initial_sidebar_state="expanded", layout="wide")

# ---------------------------------------------------------------------------
# Helper: API call with error handling
# ---------------------------------------------------------------------------
def api_get(endpoint: str, params: dict | None = None) -> dict | list | None:
    try:
        resp = requests.get(f"{API_URL}{endpoint}", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to API at **{API_URL}**. Make sure the FastAPI server is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ API request timed out. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ API error: {e}")
        return None

def api_post(endpoint: str, payload: dict) -> dict | None:
    try:
        resp = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to API at **{API_URL}**. Make sure the FastAPI server is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ API request timed out. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ API error: {e}")
        return None

# Navigation
menu = st.sidebar.selectbox(
    "Navigation",
    ["Overview", "Model Evaluation", "Prediction", "Model Monitoring"],
)

# Overview Page
if menu == "Overview":
    st.header("Overview")
    st.markdown("---")
    
    st.write("This dataset contains clinical records of patients with 12 medical variables. The original dataset included 918 record. These cover demographic information, examination results, and symptoms related to heart health.")
    
    st.write("This dashboard is created to explore heart patient data, evaluate prediction models, and provide interactive risk predictions.")

    st.caption("Dataset Link: [Kaggle - Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)")

    df_display = df_r
    html_table = "<div class='table-container'><table><tr>"
    
    for col in df_display.columns:
        html_table += f"<th>{col}</th>"
    html_table += "</tr>"
    
    for idx, row in df_display.iterrows():
        html_table += "<tr>"
        for col in df_display.columns:
            val = row[col]
            if col == "HeartDisease":
                css_class = "sick" if val == 1 else "healthy"
                html_table += f"<td class='{css_class}'>{val}</td>"
            else:
                html_table += f"<td>{val}</td>"
        html_table += "</tr>"
    html_table += "</table></div>"
    
    st.markdown(html_table, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.markdown('<div class="card"><span class="card-title">Number of Rows</span><span class="card-number">918</span></div>', unsafe_allow_html=True)
    col2.markdown('<div class="card"><span class="card-title">Number of Columns</span><span class="card-number">12</span></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("📌 Key Statistics")

    col1, col2, col3 = st.columns(3)
    col1.markdown('<div class="card">📈 Average Age<br><span class="card-number2">54 years</span></div>', unsafe_allow_html=True)
    col2.markdown('<div class="card">❤️ Disease Rate<br><span class="card-number2">55,34%</span></div>', unsafe_allow_html=True)
    col3.markdown('<div class="card">👥 Total Patients<br><span class="card-number2">918</span></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.markdown("<div class='card'>♂️ Male Patients Total<br><span class='card-number2'>725</span></div>", unsafe_allow_html=True)
    col2.markdown("<div class='card'>♀️ Female Patients Total<br><span class='card-number2'>193</span></div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("*About half of the patients in this dataset are diagnosed with heart disease, highlighting the importance of predictive modeling.*")
    
    st.markdown("---")

    st.caption("Dataset Source: Kaggle - Heart Failure Prediction (fedesoriano, 2021). For research and educational purposes only.")

# Model Evaluation Page
elif menu == "Model Evaluation":
    st.header("Model Evaluation")
    st.markdown("---")
    st.subheader("Interactive ROC Curve Analysis")
    
    eval_data_path = os.path.join(BASE_DIR, "..", "outputs", "LRmodel_eval_data.pkl")

    if not os.path.exists(eval_data_path):
        st.warning("Evaluation file doesn't exist")
    else:
        eval_data = joblib.load(eval_data_path)
        y_test = eval_data['y_test']
        y_probs_lr = eval_data['logistic_regression_probs']
    
        fpr_lr, tpr_lr, _ = roc_curve(y_test, y_probs_lr)
        roc_auc_lr = auc(fpr_lr, tpr_lr)

        fig_roc = go.Figure()

        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], 
                                    mode='lines', 
                                    line=dict(dash='dash', color='gray'),
                                    name='Random Classifier'))

        fig_roc.add_trace(go.Scatter(x=fpr_lr, y=tpr_lr, 
                                    mode='lines', 
                                    line=dict(color='royalblue', width=3),
                                    name=f'Logistic Regression (AUC = {roc_auc_lr:.3f})',
                                    hovertemplate="<b>FPR</b>: %{x:.3f}<br><b>TPR</b>: %{y:.3f}<extra></extra>"))

        fig_roc.update_layout(
            title='Plot ROC Curve',
            xaxis_title='False Positive Rate (1 - Specificity)',
            yaxis_title='True Positive Rate (Sensitivity)',
            legend=dict(x=0.7, y=0.2),
            margin=dict(l=20, r=20, t=50, b=20),
            height=1000,
            width=1000
        )

        st.plotly_chart(fig_roc, use_container_width=False)
    
    st.subheader("Confusion Matrix")
    st.image(os.path.join(BASE_DIR, "..", "outputs", "LRconfusion_matrix.png"), caption="Confusion Matrix")

    st.write("**Summary**")
    st.write("The Logistic Regression model demonstrates high diagnostic accuracy with an AUC of 0.9280 The Confusion Matrix shows that the model is particularly strong at identifying positive cases (92 True Positives), ensuring that the majority of high-risk patients are correctly flagged for medical review.")

    st.divider()

    st.subheader("Model Performance Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Overall Accuracy", value="89.13%")
    
    with col2:
        st.metric(label="Recall (Sensitivity)", value="91.18%", delta="Top Metric")

    with col3:
        st.metric(label="Precision", value="89.42%")

    with col4:
        st.metric(label="F1-Score", value="90.29%")



    st.divider()
    with st.expander("Why track multiple metrics?"):
        st.markdown("""
    ### Understanding Model Performance
    
    * **Accuracy (89.13%) - The Overall Score**: Think of this as the model's general grade. Out of 100 people, the model correctly identifies about 85 of them. While this is a high score, accuracy doesn't tell the whole story in medical cases.
    
    * **Recall / Sensitivity (91.18%)** — :star: **Top Priority**: This is the most important metric. It measures how good the model is at "catching" people who are actually sick.
        -  **Goal**: We want to make sure we don't miss anyone.
        -  **Meaning**: Out of 100 people with heart disease, we successfully find 88 of them. Only 12 might be missed, which is a very safe margin for a first-level screening.
    
    * **Precision (89.42%)**: 
        This measures how often the model is "right" when it says someone is sick.
        -  **Meaning**: If the model flags 100 people as having heart disease, 85 of them truly have it. The other 15 are "false alarms"—people who are actually healthy but the model suggested a check-up just to be safe.
    
    * **F1-Score (90.29%)**: Imagine a scale balancing Recall (not missing sick people) and Precision (not giving too many false alarms). The F1-Score is the "balance point." A high score here means the model is doing a great job at both.
        
    * **ROC-AUC (0.9280)**: 
        This score tells us how "smart" the model is at telling the difference between a healthy heart and a sick heart.
        -  0.92 is Excellent. It means the model is very reliable at sorting patients into the right categories.
    """)
    
# Model Prediction Page
elif menu == "Prediction":
    st.header("Prediction Tool")
    st.info("Please fill in the patient's clinical information below.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### **General Information**")
        age = st.slider("Age", 20, 80, 40)
        sex_label = st.selectbox("Gender", ["Male", "Female"])
        chest = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
        bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
        cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)

    with col2:
        st.markdown("##### **Clinical Measurements**")
        fastingbs_label = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
        maxhr = st.number_input("Max Heart Rate (MaxHR)", min_value=60, max_value=210, value=150)
        exang = st.selectbox("Exercise Induced Angina", ["Y", "N"])
        oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=-2.0, max_value=6.0, value=1.0, format="%.2f")
        restingecg = st.selectbox("Resting ECG Result", ["Normal", "ST", "LVH"])
        slope = st.selectbox("ST Slope Type", ["Up", "Flat", "Down"])

    with st.expander("ℹ️ Clinical Measurement Definitions"):
        st.markdown("""
            ### **Understanding the Input Parameters**
            
            * **Age:** Your current age in years. Risk factors typically increase as you get older.
            * **Resting BP (Blood Pressure):** Your blood pressure measured in **mm Hg** while you are resting. High blood pressure (Hypertension) can strain your heart and arteries.
            * **Cholesterol:** Level of serum cholesterol in **mg/dl**. High levels can lead to a buildup of plaques in your arteries (atherosclerosis).
            * **Fasting Blood Sugar:** Set to **1** if your blood sugar is **> 120 mg/dl** after fasting, and **0** otherwise. High sugar levels can damage blood vessels over time.
            * **Max HR (Heart Rate):** The highest heart rate you can achieve during intense exercise. This is used to calculate your heart rate ratio, which measures cardiovascular fitness relative to your age.
            * **Oldpeak:** Measures the **ST depression** on your ECG induced by exercise relative to rest. It is a critical indicator of how your heart handles stress; higher values often suggest a lack of oxygen to the heart.
            
            ---
            
            ### **Categorical Indicators**
            
            * **Chest Pain Type:**
                * **TA (Typical Angina):** Classic chest pain caused by heart stress.
                * **ATA (Atypical Angina):** Chest pain that doesn't follow the typical pattern.
                * **NAP (Non-Anginal Pain):** Pain that is likely not heart-related.
                * **ASY (Asymptomatic):** No pain felt, but can be a "silent" indicator of heart issues in medical datasets.
            * **Resting ECG:** * **Normal:** No electrical issues.
                * **ST:** Abnormalities in the ST-T wave.
                * **LVH:** Signs of Left Ventricular Hypertrophy (thickening of the heart's walls).
            * **Exercise Angina:** Does chest pain occur specifically during physical activity? (**Yes/No**).
            * **ST Slope:** The slope of the ST segment during peak exercise:
                * **Up:** Usually healthy.
                * **Flat:** May indicate moderate risk.
                * **Down:** Strongly associated with coronary artery disease.
        """)

    st.markdown("---")

    if st.button("Predict Results"):
        payload = {
            "Age": age,
            "Sex": sex_label,
            "ChestPainType": chest,
            "RestingBP": bp,
            "Cholesterol": cholesterol,
            "FastingBS": fastingbs_label,
            "MaxHR": maxhr,
            "ExerciseAngina": exang,
            "Oldpeak": oldpeak,
            "RestingECG": restingecg,
            "ST_Slope": slope,
        }

        result = api_post("/predict", payload)

        if result is not None:
            st.subheader("Analysis Result")

            if result["prediction"] == 1:
                st.error(f"### Prediction: Heart Disease Detected")
            else:
                st.success(f"### Prediction: Healthy / No Disease")

            prob = result["probability"]
            st.write(f"**Risk Probability: {prob:.2f}**")
            st.progress(prob)

            if prob > 0.8:
                st.warning("🚨 **High Risk:** It is highly recommended to consult a cardiac specialist immediately.")
            elif prob > result["threshold"]:
                st.warning("⚠️ **Moderate Risk:** Risk indicators detected. Consider improving your lifestyle and consulting a healthcare provider.")
            else:
                st.info("✅ **Low Risk:** Your results are within a healthy range. Maintain a balanced diet and regular exercise.")

            with st.expander("📋 API Response Details"):
                st.json(result)

# ===================================================================
# Model Monitoring Page
# ===================================================================
elif menu == "Model Monitoring":
    st.header("📊 Model Monitoring")
    st.markdown("---")

    # --- Summary Cards ---
    summary = api_get("/monitoring/summary")

    if summary is not None:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Predictions", summary["total_predictions"])
        col2.metric("🔴 High Risk", summary["high_risk_count"])
        col3.metric("🟡 Moderate Risk", summary["moderate_risk_count"])
        col4.metric("🟢 Low Risk", summary["low_risk_count"])

        if summary["total_predictions"] > 0:
            st.markdown("---")

            # --- Distribution Pie Chart ---
            st.subheader("Risk Level Distribution")
            fig_pie = go.Figure(data=[
                go.Pie(
                    labels=["High Risk", "Moderate Risk", "Low Risk"],
                    values=[
                        summary["high_risk_count"],
                        summary["moderate_risk_count"],
                        summary["low_risk_count"],
                    ],
                    marker=dict(colors=["#ff4b4b", "#ffa500", "#00cc66"]),
                    hole=0.4,
                )
            ])
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No predictions logged yet. Make a prediction first!")

    st.markdown("---")

    # --- Prediction Logs Table ---
    st.subheader("Prediction History")

    logs = api_get("/monitoring/logs", params={"limit": 200, "offset": 0})

    if logs is None:
        st.warning("Could not fetch prediction logs from API.")
    elif len(logs) == 0:
        st.info("No prediction logs available yet.")
    else:
        df_logs = pd.DataFrame(logs)

        # Format columns for display
        display_cols = [
            "timestamp", "Age", "Sex", "ChestPainType", "RestingBP",
            "Cholesterol", "FastingBS", "MaxHR", "ExerciseAngina",
            "Oldpeak", "RestingECG", "ST_Slope", "probability",
            "risk_label", "response_time_ms",
        ]
        # Only keep columns that exist
        display_cols = [c for c in display_cols if c in df_logs.columns]

        df_display = df_logs[display_cols].copy()
        df_display.columns = [
            "Timestamp", "Age", "Sex", "ChestPainType", "RestingBP",
            "Cholesterol", "FastingBS", "MaxHR", "ExerciseAngina",
            "Oldpeak", "RestingECG", "ST_Slope", "Probability",
            "Risk Label", "Resp. Time (ms)",
        ]

        # Color-code risk labels
        def color_risk(val: str) -> str:
            if val == "High Risk":
                return "background-color: #ffcccc; color: #a30000"
            elif val == "Moderate Risk":
                return "background-color: #fff3cc; color: #996600"
            else:
                return "background-color: #ccffcc; color: #006600"

        styled = df_display.style.applymap(color_risk, subset=["Risk Label"])
        st.dataframe(styled, use_container_width=True, height=400)

        # Download button
        csv = df_display.to_csv(index=False)
        st.download_button(
            label="📥 Download Logs (CSV)",
            data=csv,
            file_name="prediction_logs.csv",
            mime="text/csv",
        )

    st.markdown("---")

    # --- API Health Check ---
    st.subheader("🔌 API Health")
    health = api_get("/health")
    if health is not None:
        st.json(health)
    else:
        st.error("API is not reachable.")

