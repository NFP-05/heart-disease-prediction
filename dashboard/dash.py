import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
            
    .cardn {
        background-color: #F0F2F6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        font-size: 16px;
        font-weight: normal;
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
    </style>
    """, unsafe_allow_html=True)

# Judul dashboard
st.title("🫀 Heart Disease Prediction Dashboard")

# Load dataset
path_cleaned = os.path.join(BASE_DIR, "..", "Data", "heart_cleaned.csv")

# 3. Baca filenya menggunakan path yang sudah fleksibel tadi
df = pd.read_csv(path_cleaned)

# Sidebar navigasi
menu = st.sidebar.selectbox("Navigation", ["Overview", "EDA", "Model Evaluation", "Prediction"])

if menu == "Overview":
    st.header("Overview")
    st.markdown("---")
    
    st.write("This dataset contains clinical records of patients with 12 medical variables. The original dataset included 918 record. These cover demographic information, examination results, and symptoms related to heart health.")
    
    st.write("This dashboard is created to explore heart patient data, evaluate prediction models, and provide interactive risk predictions.")

    st.caption("Dataset Link: [Kaggle - Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)")

    df_display = df
    html_table = "<div class='table-container'><table><tr>"
    
    # th
    for col in df_display.columns:
        html_table += f"<th>{col}</th>"
    html_table += "</tr>"
    
    # td
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

    col1, col2 = st.columns(2)
    col1.markdown('<div class="cardn">Number of Rows<span class="card-number">429</span></div>', unsafe_allow_html=True)
    col2.markdown('<div class="cardn">Number of Features<span class="card-number">12</span></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("📌 Key Statistics")

    col1, col2, col3 = st.columns(3)
    col1.markdown('<div class="card">📈 Average Age<br>56 years</div>', unsafe_allow_html=True)
    col2.markdown('<div class="card">❤️ Disease Rate<br>65.0%</div>', unsafe_allow_html=True)
    col3.markdown('<div class="card">👥 Total Patients<br>429</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("*About two-thirds of patients in this dataset are diagnosed with heart disease, highlighting the importance of predictive modeling.*")
    
    st.markdown("---")
    st.caption("Dataset Source: Kaggle - Heart Failure Prediction (fedesoriano, 2021). For research and educational purposes only.")

elif menu == "EDA":
    st.header("📊 Exploratory Data Analysis")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "Clinical Measurements", "Symptoms", "Target & Correlation"])
    with tab1:
        st.subheader("Age Distribution Among the Patients")
        gender_filter = st.radio("Select Gender for Age Distribution", ["👥 All", "👨 Male", "👩 Female"], key="age_gender_filter")
        if gender_filter == "👥 All":
            filtered_df = df
        elif gender_filter == "👨 Male":
            filtered_df = df[df["Sex"] == "M"]
        else:  
            filtered_df = df[df["Sex"] == "F"]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(filtered_df["Age"], bins=20, color="skyblue", edgecolor="black")
        ax.set_xlabel("Age")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        if gender_filter == "👥 All":
            st.write("Most of the patients in this dataset are middle‑aged adults, with the largest group falling between 50 and 60 years old. " \
            " The numbers taper off at younger and older ages, showing that heart disease tends to affect people most often in their mid‑life years." \
            " In simple terms, this chart tells us that the risk is highest around the age when many people are still active in their careers and family life.")
        elif gender_filter == "👨 Male":
            st.write("Among male patients, most are in their mid‑50s to early 60s. The numbers drop off at younger and older ages, " \
            "showing that men in their middle years are the largest group represented. " \
            "This highlights that heart disease risk for men tends to peak around mid‑life.")
        else:  # Female
            st.write("For female patients, the distribution is more spread out, with noticeable peaks around ages 50 and 60. " \
            "While fewer women are represented overall compared to men, the chart suggests that middle‑aged women also form an important group when studying heart disease risk.")

        st.markdown("---")

        st.subheader("Gender Distribution")
        disease_filter = st.radio("Select Heart Disease Status for Gender Distribution", ["💓 All", "💙 Healthy", "❤️ Heart Disease"], key="gender_disease_filter")
        if disease_filter == "💓 All":
            filtered_df_gender = df
        elif disease_filter == "💙 Healthy":
            filtered_df_gender = df[df["HeartDisease"] == 0]
        else: 
            filtered_df_gender = df[df["HeartDisease"] == 1]

        fig, ax = plt.subplots(figsize=(10, 5))
        gender_counts = filtered_df_gender["Sex"].value_counts()
        ax.bar(gender_counts.index, gender_counts.values, color=["steelblue", "salmon"], edgecolor="black")
        ax.set_xlabel("Gender")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        if disease_filter == "💓 All":
            st.write("This chart shows that most of the patients in our dataset are men, while women make up a smaller portion.  " \
            "The difference is quite noticeable, with men outnumbering women by a wide margin. " \
            "In simple terms, this tells us that heart disease in this dataset appears more frequently among men, highlighting how gender can play an important role in understanding health risks. ")
        elif disease_filter == "💙 Healthy":
            st.write("Looking only at healthy individuals, men slightly outnumber women. " \
            "This means that in the dataset, there are more men without heart disease than women, " \
            "though the difference is not as dramatic as in the overall population.")
        else:  
            st.write("When focusing on patients diagnosed with heart disease, the difference between genders becomes striking. Men make up the vast majority of cases, " \
            "while women represent only a small fraction. This emphasizes how much more common heart disease appears among men in this dataset.")

        st.markdown("---")  

    with tab2:
        st.subheader("Resting Blood Pressure")
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].hist(df["RestingBP"], bins=20, color="lightcoral", edgecolor="black")
        ax[0].set_xlabel("Resting Blood Pressure (mmHg)")
        ax[0].set_ylabel("Frequency")
        ax[0].set_title("Histogram")
        ax[1].boxplot(df["RestingBP"], vert=True)
        ax[1].set_ylabel("Resting Blood Pressure (mmHg)")
        ax[1].set_title("Boxplot")
        st.pyplot(fig)

        st.write("Most patients have resting blood pressure around 130–140 mmHg, which is slightly above the normal range. The histogram shows this cluster clearly. " \
        "The boxplot adds another layer, it highlights the median near 135 mmHg and reveals several outliers above 180 mmHg. " \
        "These outliers tell us that while many patients fall into a moderately elevated range, a smaller group experiences very high blood pressure, pointing to more severe hypertension risks.")

        st.markdown("---")

        st.subheader("Cholesterol")
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].hist(df["Cholesterol"], bins=20, color="lightgreen", edgecolor="black")
        ax[0].set_xlabel("Cholesterol (mg/dl)")
        ax[0].set_ylabel("Frequency")
        ax[0].set_title("Histogram")
        ax[1].boxplot(df["Cholesterol"], vert=True)
        ax[1].set_ylabel("Cholesterol (mg/dl)")
        ax[1].set_title("Boxplot")
        st.pyplot(fig)

        st.write("The histogram shows that most patients have cholesterol levels clustered around 250–300 mg/dl, which is higher than the recommended healthy range. " \
        "Then the boxplot explains that the median sits near 250 mg/dl, but there are many outliers above 400 mg/dl. These extreme values highlight that while elevated cholesterol is common, " \
        "a smaller group of patients face dangerously high levels that could significantly increase cardiovascular risk. Together, the two charts tell us that cholesterol problems are widespread, and for some individuals, they are severe.")

        st.markdown("---")

        st.subheader("Maximum Heart Rate Achieved")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df["MaxHR"], bins=20, color="lightyellow", edgecolor="black")
        ax.set_xlabel("Maximum Heart Rate (bpm)")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        st.write("Most patients reach a maximum heart rate between 110 and 160 beats per minute, with the peak around 120 bpm. " \
        "This shows the typical range of exertion capacity in the dataset. In everyday terms, most people’s hearts peak at a level consistent with moderate to vigorous activity.")

        st.markdown("---")

        st.subheader("Fasting Blood Sugar")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(6, 5))
            fbs_counts = df["FastingBS"].value_counts()
            labels = ["< 120 mg/dl", "> 120 mg/dl"]
            ax.bar([labels[int(i)] for i in fbs_counts.index], fbs_counts.values, color=["steelblue", "orange"], edgecolor="black")
            ax.set_ylabel("Count")
            plt.tight_layout()
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(6, 5))
            fbs_counts = df["FastingBS"].value_counts()
            labels = ["< 120 mg/dl", "> 120 mg/dl"]
            colors = ["steelblue", "orange"]
            ax.pie(fbs_counts.values, labels=[labels[int(i)] for i in fbs_counts.index], autopct="%1.1f%%", colors=colors, startangle=90)
            ax.set_title("Distribution")
            plt.tight_layout()
            st.pyplot(fig)
        
        st.write("The majority of patients about 80% have fasting blood sugar below 120 mg/dl, which is considered normal. Around 20% have higher levels, " \
        "indicating possible diabetes or pre‑diabetes. This chart makes it clear that while most patients maintain healthy blood sugar, a significant minority face elevated risks.")

        st.markdown("---")

    with tab3:
        st.subheader("Chest Pain Type Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        chest_counts = df["ChestPainType"].value_counts()
        ax.bar(chest_counts.index, chest_counts.values, color="skyblue", edgecolor="black")
        ax.set_xlabel("Chest Pain Type")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        st.write("Most patients report asymptomatic chest pain (ASY), meaning they don’t feel typical chest pain even though heart disease may be present. " \
        "This is important because silent cases can be harder to detect. Other types like non‑anginal pain (NAP) and typical angina (TA) are less common, showing that chest pain symptoms vary widely across patients.")        

        st.markdown("---")

        st.subheader("Exercise-Induced Angina")
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        angina_counts = df["ExerciseAngina"].value_counts()
        labels = ["No", "Yes"]
        colors = ["steelblue", "salmon"]
        # Bar chart
        ax[0].bar(labels, angina_counts.values, color=colors, edgecolor="black")
        ax[0].set_title("Bar Chart")
        # Pie chart
        ax[1].pie(angina_counts.values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
        ax[1].set_title("Pie Chart")
        st.pyplot(fig)

        st.write("The charts show that more than half of the patients do not experience angina during exercise, while about 40% do. " \
        "This tells us that exercise can trigger chest pain in a significant portion of patients, which is a warning sign of reduced blood flow to the heart.")

        st.markdown("---")

        st.subheader("Oldpeak Distribution")
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].hist(df["Oldpeak"], bins=20, color="lightgreen", edgecolor="black")
        ax[0].set_title("Histogram")
        ax[0].set_xlabel("Oldpeak")
        ax[0].set_ylabel("Frequency")
        ax[1].boxplot(df["Oldpeak"], vert=True)
        ax[1].set_title("Boxplot")
        ax[1].set_ylabel("Oldpeak")
        st.pyplot(fig)

        st.write("Most patients have Oldpeak values between 0 and 2, which indicates mild changes in the ECG after exercise. The boxplot highlights a median near 1 and a few outliers above 4, " \
        "showing that while many patients have small changes, some experience much more severe shifts that could signal higher risk.")

        st.markdown("---")

        st.subheader("ST Slope Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        slope_counts = df["ST_Slope"].value_counts()
        ax.bar(slope_counts.index, slope_counts.values, color="orange", edgecolor="black")
        ax.set_xlabel("ST Slope")
        ax.set_ylabel("Count")
        st.pyplot(fig)

        st.write("The majority of patients show a flat ST slope, which is often linked to abnormal heart function. Fewer patients have an upward slope, and only a small group show a downward slope. " \
        "This pattern suggests that abnormal ECG slopes are common in the dataset, reinforcing the importance of monitoring heart signals closely.")
    
        st.markdown("---")

    with tab4 :
        st.subheader("Heart Disease Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        target_counts = df["HeartDisease"].value_counts()
        labels = ["Healthy", "Heart Disease"]
        colors = ["skyblue", "lightcoral"]
        ax.pie(target_counts.values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
        ax.set_title("Proportion of Patients by Heart Disease Status")
        st.pyplot(fig)

        st.write("The pie chart shows that about two‑thirds of the patients in this dataset are healthy, while roughly one‑third have heart disease. This means that although most individuals do not show signs of heart disease, " \
        "a significant portion—more than one in three—are affected. In everyday terms, the chart highlights that heart disease is common enough to be a major concern within this population.")

        st.markdown("---")

        st.subheader("Feature Correlation Heatmap")
        numeric_df = df.select_dtypes(include=["int64", "float64"])
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Matrix of Numerical Features")
        st.pyplot(fig)

        st.markdown(
        "The heatmap reveals how different health measurements relate to each other and to heart disease. "
        "For example, maximum heart rate (MaxHR) has a negative correlation with heart disease, "
        "<span class='highlight-blue'>Patients with lower maximum heart rates are more likely to have the condition.</span> "
        "On the other hand, Oldpeak shows a positive correlation, indicating that higher ECG changes after exercise are linked to greater risk. "
        "Other features like age, resting blood pressure, and cholesterol show weaker relationships. "
        "In simple terms, this chart helps us see which factors matter most: <span class='highlight-red'>heart disease is more closely tied to exercise‑related signals than to cholesterol or blood pressure alone.</span>",
        unsafe_allow_html=True
        )

        st.markdown("---")

elif menu == "Model Evaluation":
    st.header("Model Evaluation")
    st.markdown("---")
    st.subheader("Plot ROC Curves")
    st.image(os.path.join(BASE_DIR, "..", "outputs", "roc_curves.png"), caption="ROC Curves")
    
    st.subheader("Confusion Matrix")
    st.image(os.path.join(BASE_DIR, "..", "outputs", "confusion_matrices.png"), caption="Confusion Matrices")

    st.write("**Summary**")
    st.write("Our model demonstrates high diagnostic accuracy with an AUC of 0.92. The Confusion Matrix shows that the model is particularly strong at identifying positive cases (85 True Positives), ensuring that the majority of high-risk patients are correctly flagged for medical review.")

    st.divider()

    st.subheader("Model Performance Summary")

    # Membuat 4 kolom untuk metrik utama
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Overall Accuracy", value="85.3%")
    
    with col2:
        # Kita beri warna hijau atau highlight karena ini fokus utamamu
        st.metric(label="Recall (Sensitivity)", value="88.2%", delta="Top Metric")

    with col3:
        st.metric(label="Precision", value="85.7%")

    with col4:
        st.metric(label="F1-Score", value="87.0%")

    st.divider() # Garis pemisah agar rapi
    with st.expander("Why track multiple metrics?"):
        st.markdown("""
    ### Understanding Model Performance
    
    * **Accuracy (85.3%) - The Overall Score**: Think of this as the model's general grade. Out of 100 people, the model correctly identifies about 85 of them. While this is a high score, accuracy doesn't tell the whole story in medical cases.
    
    * **Recall / Sensitivity (88.2%)** — :star: **Top Priority**: This is the most important metric. It measures how good the model is at "catching" people who are actually sick.
        -  **Goal**: We want to make sure we don't miss anyone.
        -  **Meaning**: Out of 100 people with heart disease, we successfully find 88 of them. Only 12 might be missed, which is a very safe margin for a first-level screening.
    
    * **Precision (85.7%)**: 
        This measures how often the model is "right" when it says someone is sick.
        -  **Meaning**: If the model flags 100 people as having heart disease, 85 of them truly have it. The other 15 are "false alarms"—people who are actually healthy but the model suggested a check-up just to be safe.
    
    * **F1-Score (87.0%)**: Imagine a scale balancing Recall (not missing sick people) and Precision (not giving too many false alarms). The F1-Score is the "balance point." A high score here means the model is doing a great job at both.
        
    * **ROC-AUC (0.92)**: 
        This score tells us how "smart" the model is at telling the difference between a healthy heart and a sick heart.
        -  0.92 is Excellent. It means the model is very reliable at sorting patients into the right categories.
    """)
    

elif menu == "Prediction":
    st.header("Prediction Tool")
    st.info("Please fill in the patient's clinical information below.")

    model_path = os.path.join(BASE_DIR, "..", "outputs", "best_model.pkl")
    encoders_path = os.path.join(BASE_DIR, "..", "outputs", "encoders.pkl")
    scaler_path = os.path.join(BASE_DIR, "..", "outputs", "scaler.pkl")
    train_columns_path = os.path.join(BASE_DIR, "..", "outputs", "train_columns.pkl")

    if not os.path.exists(model_path) or not os.path.exists(encoders_path) or not os.path.exists(scaler_path) or not os.path.exists(train_columns_path):
        st.warning("Model, encoder, scaler, atau train_columns belum tersedia. Jalankan preprocessing dan training terlebih dahulu.")
    else:
        model = joblib.load(model_path)
        encoders = joblib.load(encoders_path)
        scaler = joblib.load(scaler_path)
        train_columns = joblib.load(train_columns_path)

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
                * **Max HR (Heart Rate):** The highest heart rate you can achieve during intense exercise. Generally, a higher Max HR relative to your age suggests better cardiovascular fitness.
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
            # --- DATA PROCESSING LOGIC (Tetap sama dengan milikmu) ---
            sex = "M" if sex_label == "Male" else "F"
            fastingbs = 1 if fastingbs_label == "Yes" else 0

            try:
                sample = pd.DataFrame([
                    [age, sex, bp, cholesterol, fastingbs, maxhr, exang, oldpeak, chest, restingecg, slope]
                ], columns=[
                    'Age', 'Sex', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR',
                    'ExerciseAngina', 'Oldpeak', 'ChestPainType', 'RestingECG', 'ST_Slope'
                ])

                # Label encoding untuk fitur binary
                sample['Sex'] = encoders['label_encoders']['Sex'].transform(sample['Sex'])
                sample['ExerciseAngina'] = encoders['label_encoders']['ExerciseAngina'].transform(sample['ExerciseAngina'])

                # One-hot encoding untuk fitur multi-kategori
                sample = pd.get_dummies(sample, columns=['ChestPainType', 'RestingECG', 'ST_Slope'], drop_first=True)

                # Scale numeric features sama seperti preprocessing
                numeric_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
                sample[numeric_cols] = scaler.transform(sample[numeric_cols])

                # Pastikan kolom sama dengan feature training
                sample = sample.reindex(columns=train_columns, fill_value=0)

                # Prediksi
                pred = model.predict(sample)[0]
                prob = model.predict_proba(sample)[0][1]

                # --- HASIL PREDIKSI (UI Baru) ---
                st.subheader("Analysis Result")
                
                threshold = 0.6
                pred_custom = 1 if prob > threshold else 0

                if pred_custom == 1:
                    st.error(f"### Prediction: Heart Disease Detected")
                else:
                    st.success(f"### Prediction: Healthy / No Disease")
                
                # Menampilkan Probabilitas dengan Progress Bar
                st.write(f"**Risk Probability: {prob:.2f}**")
                st.progress(prob)
                
                if prob > 0.8:
                    st.warning("🚨 **High Risk:** It is highly recommended to consult a cardiac specialist immediately.")
                elif prob > threshold:
                    st.warning("⚠️ **Moderate Risk:** Risk indicators detected. Consider improving your lifestyle and consulting a healthcare provider.")
                else:
                    st.info("✅ **Low Risk:** Your results are within a healthy range. Maintain a balanced diet and regular exercise.")
                
            except Exception as e:
                st.error(f"Error during prediction: {e}")

