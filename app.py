
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Student Performance System", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.title-box {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    padding: 28px;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
}
.card {
    background: white;
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    margin-bottom: 18px;
}
.metric-card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
}
.metric-card h3 {
    color: #2a5298;
    font-size: 30px;
}
.success-box {
    background: #e8f7ee;
    color: #166534;
    padding: 18px;
    border-radius: 14px;
    border-left: 6px solid #22c55e;
    font-weight: 600;
}
.warning-box {
    background: #fff7ed;
    color: #9a3412;
    padding: 18px;
    border-radius: 14px;
    border-left: 6px solid #f97316;
    font-weight: 600;
}
.footer {
    text-align: center;
    color: #777;
    padding-top: 20px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "student_performance_model.joblib"

@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif os.path.exists("student-mat.csv"):
        df = pd.read_csv("student-mat.csv")
    else:
        return None
    df.columns = df.columns.str.strip()
    return df

def prepare_data(df):
    df = df.copy()
    df = df.drop_duplicates()

    if "G3" not in df.columns:
        st.error("Dataset must contain target column 'G3'.")
        return None

    if "G1" in df.columns and "G2" in df.columns:
        df["total_previous_grades"] = df["G1"] + df["G2"]
        df["average_previous_grade"] = (df["G1"] + df["G2"]) / 2

    if "absences" in df.columns:
        df["attendance_level"] = pd.cut(
            df["absences"],
            bins=[-1, 5, 15, 100],
            labels=["High Attendance", "Moderate Attendance", "Low Attendance"]
        )

    df["performance_category"] = pd.cut(
        df["G3"],
        bins=[-1, 9, 14, 20],
        labels=["Low", "Average", "High"]
    )
    return df

def train_model(df):
    X = df.drop(columns=["G3", "performance_category"], errors="ignore")
    y = df["G3"]

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=200, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred)
    }

    joblib.dump(model, MODEL_PATH)
    return model, metrics

st.sidebar.title("🎓 Student System")
st.sidebar.write("Data Science using Python")
uploaded_file = st.sidebar.file_uploader("Upload student-mat.csv", type=["csv"])

menu = st.sidebar.radio(
    "Navigation",
    ["Home", "Dataset Overview", "Visualization Dashboard", "Train Model", "Prediction System", "About Project"]
)

df_raw = load_data(uploaded_file)
df = prepare_data(df_raw) if df_raw is not None else None

st.markdown("""
<div class="title-box">
    <h1>🎓 Student Performance Analysis & Prediction System</h1>
    <p>Analyze academic patterns, visualize insights, and predict final grades using Python</p>
</div>
""", unsafe_allow_html=True)

if menu == "Home":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📌 Project Overview")
    st.write("""
    This system analyzes student academic performance and predicts final grades based on factors such as study time,
    previous grades, absences, family background, and other academic attributes.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-card"><h3>EDA</h3><p>Exploratory Data Analysis</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card"><h3>ML</h3><p>Prediction Model</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card"><h3>UI</h3><p>Streamlit Dashboard</p></div>', unsafe_allow_html=True)

elif menu == "Dataset Overview":
    st.header("📂 Dataset Overview")
    if df is None:
        st.markdown('<div class="warning-box">Please upload student-mat.csv from the sidebar.</div>', unsafe_allow_html=True)
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Records", df.shape[0])
        with c2:
            st.metric("Total Columns", df.shape[1])
        with c3:
            st.metric("Missing Values", int(df.isnull().sum().sum()))
        with c4:
            st.metric("Duplicates", int(df.duplicated().sum()))

        st.subheader("Dataset Preview")
        st.dataframe(df.head(15), use_container_width=True)

        st.subheader("Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)

        st.subheader("Missing Value Summary")
        missing_df = df.isnull().sum().reset_index()
        missing_df.columns = ["Column", "Missing Values"]
        st.dataframe(missing_df, use_container_width=True)

elif menu == "Visualization Dashboard":
    st.header("📊 Visualization Dashboard")
    if df is None:
        st.markdown('<div class="warning-box">Please upload student-mat.csv from the sidebar.</div>', unsafe_allow_html=True)
    else:
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Distribution of Final Grades")
            fig, ax = plt.subplots()
            sns.histplot(df["G3"], kde=True, ax=ax)
            ax.set_xlabel("Final Grade")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        with c2:
            st.subheader("Study Time vs Final Grade")
            fig, ax = plt.subplots()
            sns.barplot(x="studytime", y="G3", data=df, ax=ax)
            ax.set_xlabel("Study Time")
            ax.set_ylabel("Final Grade")
            st.pyplot(fig)

        c3, c4 = st.columns(2)

        with c3:
            st.subheader("Absences vs Final Grade")
            fig, ax = plt.subplots()
            sns.scatterplot(x="absences", y="G3", data=df, ax=ax)
            ax.set_xlabel("Absences")
            ax.set_ylabel("Final Grade")
            st.pyplot(fig)

        with c4:
            st.subheader("Performance Category Count")
            fig, ax = plt.subplots()
            sns.countplot(x="performance_category", data=df, ax=ax)
            ax.set_xlabel("Performance Category")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.heatmap(df.corr(numeric_only=True), annot=True, ax=ax)
        st.pyplot(fig)

elif menu == "Train Model":
    st.header("🤖 Train Prediction Model")
    if df is None:
        st.markdown('<div class="warning-box">Please upload student-mat.csv from the sidebar.</div>', unsafe_allow_html=True)
    else:
        if st.button("Train Random Forest Model"):
            model, metrics = train_model(df)
            st.markdown('<div class="success-box">Model trained and saved successfully!</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"{metrics['MAE']:.2f}")
            c2.metric("RMSE", f"{metrics['RMSE']:.2f}")
            c3.metric("R² Score", f"{metrics['R2']:.2f}")

elif menu == "Prediction System":
    st.header("🎯 Final Grade Prediction System")
    if df is None:
        st.markdown('<div class="warning-box">Please upload student-mat.csv from the sidebar.</div>', unsafe_allow_html=True)
    else:
        if not os.path.exists(MODEL_PATH):
            model, _ = train_model(df)
        else:
            model = joblib.load(MODEL_PATH)

        st.write("Enter student details below to predict final grade.")

        feature_cols = df.drop(columns=["G3", "performance_category"], errors="ignore").columns
        input_values = {}

        c1, c2, c3 = st.columns(3)

        with c1:
            if "G1" in feature_cols:
                input_values["G1"] = st.number_input("First Period Grade (G1)", 0.0, 20.0, 10.0)
            if "G2" in feature_cols:
                input_values["G2"] = st.number_input("Second Period Grade (G2)", 0.0, 20.0, 10.0)
            if "studytime" in feature_cols:
                input_values["studytime"] = st.selectbox("Study Time", [1, 2, 3, 4])

        with c2:
            if "absences" in feature_cols:
                input_values["absences"] = st.number_input("Absences", 0.0, 100.0, 5.0)
            if "failures" in feature_cols:
                input_values["failures"] = st.selectbox("Past Failures", [0, 1, 2, 3])
            if "schoolsup" in feature_cols:
                input_values["schoolsup"] = st.selectbox("School Support", sorted(df["schoolsup"].dropna().unique().tolist()))

        with c3:
            if "sex" in feature_cols:
                input_values["sex"] = st.selectbox("Gender", sorted(df["sex"].dropna().unique().tolist()))
            if "internet" in feature_cols:
                input_values["internet"] = st.selectbox("Internet Access", sorted(df["internet"].dropna().unique().tolist()))
            if "higher" in feature_cols:
                input_values["higher"] = st.selectbox("Wants Higher Education", sorted(df["higher"].dropna().unique().tolist()))

        if "G1" in input_values and "G2" in input_values and "total_previous_grades" in feature_cols:
            input_values["total_previous_grades"] = input_values["G1"] + input_values["G2"]
        if "G1" in input_values and "G2" in input_values and "average_previous_grade" in feature_cols:
            input_values["average_previous_grade"] = (input_values["G1"] + input_values["G2"]) / 2
        if "absences" in input_values and "attendance_level" in feature_cols:
            absences = input_values["absences"]
            input_values["attendance_level"] = "High Attendance" if absences <= 5 else "Moderate Attendance" if absences <= 15 else "Low Attendance"

        prediction_input = {}
        for col in feature_cols:
            if col in input_values:
                prediction_input[col] = [input_values[col]]
            elif df[col].dtype in ["int64", "float64"]:
                prediction_input[col] = [df[col].median()]
            else:
                prediction_input[col] = [df[col].mode()[0]]

        prediction_df = pd.DataFrame(prediction_input)

        if st.button("Predict Final Grade"):
            pred = model.predict(prediction_df)[0]
            pred = max(0, min(20, pred))

            st.markdown(f'<div class="success-box">Predicted Final Grade: {pred:.2f} / 20</div>', unsafe_allow_html=True)

            if pred < 10:
                st.error("Performance Category: Low")
            elif pred < 15:
                st.warning("Performance Category: Average")
            else:
                st.success("Performance Category: High")

elif menu == "About Project":
    st.header("ℹ️ About Project")
    st.subheader("Project Title")
    st.write("Student Performance Analysis and Prediction System")
    st.subheader("Objective")
    st.write("To analyze student academic performance and build a prediction system that estimates final grades.")
    st.subheader("Dataset Used")
    st.write("student-mat.csv")
    st.subheader("Conclusion")
    st.write("This project improves understanding of educational data analysis, dashboard creation, and prediction system development using Python.")

st.markdown('<div class="footer">Developed using Python, Streamlit, Pandas, Seaborn and Scikit-learn</div>', unsafe_allow_html=True)
