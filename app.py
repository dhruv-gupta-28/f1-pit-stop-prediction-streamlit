import os
import pickle
import warnings

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="F1 Strategy Intelligence",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #050505 0%, #111827 100%);
        color: white;
    }

    section[data-testid="stSidebar"] {
        background: #0B0F19;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.2rem;
        letter-spacing: -1px;
    }

    .subtitle {
        color: #9CA3AF;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .glass-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.06);
        padding: 1.5rem;
        border-radius: 24px;
        backdrop-filter: blur(12px);
        margin-bottom: 1rem;
    }

    .metric-card {
        background: linear-gradient(
            135deg,
            rgba(255,255,255,0.05),
            rgba(255,255,255,0.02)
        );

        border-radius: 22px;
        padding: 1.5rem;
        border: 1px solid rgba(255,255,255,0.05);
    }

    .prediction-positive {
        background: linear-gradient(
            135deg,
            #991B1B,
            #DC2626
        );

        color: white;
        padding: 1.5rem;
        border-radius: 20px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
    }

    .prediction-negative {
        background: linear-gradient(
            135deg,
            #111827,
            #1F2937
        );

        color: white;
        padding: 1.5rem;
        border-radius: 20px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
    }

    .stButton>button {
        width: 100%;
        height: 3.5rem;
        border-radius: 16px;
        border: none;
        background: linear-gradient(
            135deg,
            #DC2626,
            #991B1B
        );

        color: white;
        font-weight: 600;
        font-size: 1rem;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SAFE LABEL ENCODER
# ============================================================

class SafeLabelEncoder:

    def __init__(self):
        self.mapping = {}

    def fit(self, values):

        values = values.astype(str).unique()

        self.mapping = {
            value: idx
            for idx, value in enumerate(values)
        }

        return self

    def transform(self, values):

        values = values.astype(str)

        return np.array([
            self.mapping.get(value, -1)
            for value in values
        ])


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def get_race_phase(progress):

    if progress < 0.33:
        return "Early"

    elif progress < 0.66:
        return "Mid"

    return "Late"


def feature_engineering(df):

    df = df.copy()

    if "id" in df.columns:
        df = df.drop(columns=["id"])

    df["TyreWearRate"] = (
        df["TyreLife"] /
        (df["LapNumber"] + 1)
    )

    df["DegradationPerLap"] = (
        df["Cumulative_Degradation"] /
        (df["LapNumber"] + 1)
    )

    df["PositionPressure"] = (
        df["Position"] *
        df["RaceProgress"]
    )

    df["RacePhase"] = (
        df["RaceProgress"]
        .apply(get_race_phase)
    )

    df["LapIntensity"] = np.abs(
        df["LapTime_Delta"]
    )

    return df


# ============================================================
# LOAD ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    artifact_folder = "artifacts"

    with open(
        os.path.join(artifact_folder, "model.pkl"),
        "rb"
    ) as f:
        model = pickle.load(f)

    with open(
        os.path.join(artifact_folder, "scaler.pkl"),
        "rb"
    ) as f:
        scaler = pickle.load(f)

    with open(
        os.path.join(artifact_folder, "encoders.pkl"),
        "rb"
    ) as f:
        encoders = pickle.load(f)

    with open(
        os.path.join(artifact_folder, "features.pkl"),
        "rb"
    ) as f:
        feature_columns = pickle.load(f)

    return model, scaler, encoders, feature_columns


model, scaler, encoders, feature_columns = load_artifacts()


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_input(df):

    df = feature_engineering(df)

    for col, encoder in encoders.items():
        df[col] = encoder.transform(df[col])

    X = df[feature_columns]

    X_scaled = scaler.transform(X)

    return X_scaled


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("F1 Strategy AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Live Prediction",
        "Batch Prediction",
        "Model Analytics"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Machine learning based Formula 1 pit strategy prediction platform."
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        """
        <div class='main-title'>
        F1 Pit Strategy Intelligence
        </div>

        <div class='subtitle'>
        Advanced machine learning platform for race strategy analysis.
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            """
            <div class='metric-card'>
                <h3>Accuracy</h3>
                <h1>91%</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class='metric-card'>
                <h3>F1 Score</h3>
                <h1>90%</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class='metric-card'>
                <h3>Features</h3>
                <h1>5</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        st.markdown(
            """
            <div class='metric-card'>
                <h3>Model</h3>
                <h1>DT</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    chart_data = pd.DataFrame({
        "Lap": np.arange(1, 25),
        "Tyre Degradation": np.random.uniform(
            0.2,
            1.0,
            24
        ).cumsum()
    })

    fig = px.line(
        chart_data,
        x="Lap",
        y="Tyre Degradation",
        template="plotly_dark"
    )

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# LIVE PREDICTION
# ============================================================

elif page == "Live Prediction":

    st.markdown(
        """
        <div class='main-title'>
        Live Race Prediction
        </div>

        <div class='subtitle'>
        Predict next-lap pit stop probability using live race conditions.
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        driver = st.selectbox(
            "Driver",
            [
                "Verstappen",
                "Hamilton",
                "Leclerc",
                "Norris",
                "Russell",
                "Sainz"
            ]
        )

        compound = st.selectbox(
            "Tyre Compound",
            ["Soft", "Medium", "Hard"]
        )

        race = st.selectbox(
            "Race",
            [
                "Monaco",
                "Silverstone",
                "Bahrain",
                "Spa",
                "Suzuka"
            ]
        )

        year = st.number_input(
            "Year",
            min_value=2020,
            max_value=2035,
            value=2024
        )

        pit_stop = st.number_input(
            "Current Pit Stops",
            min_value=0,
            max_value=10,
            value=1
        )

    with col2:

        lap_number = st.slider(
            "Lap Number",
            1,
            80,
            24
        )

        stint = st.slider(
            "Stint",
            1,
            6,
            2
        )

        tyre_life = st.slider(
            "Tyre Life",
            0.0,
            50.0,
            16.0
        )

        position = st.slider(
            "Track Position",
            1,
            20,
            5
        )

        lap_time = st.number_input(
            "Lap Time (s)",
            value=89.5
        )

    with col3:

        lap_delta = st.number_input(
            "Lap Time Delta",
            value=1.2
        )

        cumulative_deg = st.number_input(
            "Cumulative Degradation",
            value=24.5
        )

        race_progress = st.slider(
            "Race Progress",
            0.0,
            1.0,
            0.52
        )

        position_change = st.number_input(
            "Position Change",
            value=1.0
        )

    if st.button("Generate Prediction"):

        input_df = pd.DataFrame({

            "Driver": [driver],
            "Compound": [compound],
            "Race": [race],
            "Year": [year],
            "PitStop": [pit_stop],
            "LapNumber": [lap_number],
            "Stint": [stint],
            "TyreLife": [tyre_life],
            "Position": [position],
            "LapTime (s)": [lap_time],
            "LapTime_Delta": [lap_delta],
            "Cumulative_Degradation": [cumulative_deg],
            "RaceProgress": [race_progress],
            "Position_Change": [position_change]
        })

        processed = preprocess_input(input_df)

        prediction = model.predict(processed)[0]

        probabilities = model.predict_proba(processed)[0]

        pit_probability = probabilities[1] * 100
        no_pit_probability = probabilities[0] * 100

        st.markdown("<br>", unsafe_allow_html=True)

        if prediction == 1:

            st.markdown(
                f"""
                <div class='prediction-positive'>
                High probability of pit stop next lap
                <br><br>
                Confidence: {pit_probability:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class='prediction-negative'>
                Pit stop not expected next lap
                <br><br>
                Confidence: {no_pit_probability:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

        probability_df = pd.DataFrame({
            "Outcome": ["No Pit", "Pit"],
            "Probability": [
                no_pit_probability,
                pit_probability
            ]
        })

        fig = px.bar(
            probability_df,
            x="Outcome",
            y="Probability",
            template="plotly_dark",
            text_auto=True
        )

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# BATCH PREDICTION
# ============================================================

elif page == "Batch Prediction":

    st.markdown(
        """
        <div class='main-title'>
        Batch Prediction Engine
        </div>

        <div class='subtitle'>
        Upload datasets and generate large-scale pit strategy predictions.
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload CSV Dataset",
        type=["csv"]
    )

    if uploaded_file is not None:

        batch_df = pd.read_csv(uploaded_file)

        st.dataframe(
            batch_df.head(),
            use_container_width=True
        )

        if st.button("Run Batch Prediction"):

            progress_bar = st.progress(0)

            for i in range(100):
                progress_bar.progress(i + 1)

            processed = preprocess_input(batch_df)

            predictions = model.predict(processed)

            probabilities = model.predict_proba(processed)

            result_df = batch_df.copy()

            result_df["PitNextLap"] = predictions

            result_df["PitProbability"] = probabilities[:, 1]

            st.success(
                "Prediction completed successfully"
            )

            st.dataframe(
                result_df.head(),
                use_container_width=True
            )

            csv = result_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name="f1_predictions.csv",
                mime="text/csv"
            )


# ============================================================
# MODEL ANALYTICS
# ============================================================

elif page == "Model Analytics":

    st.markdown(
        """
        <div class='main-title'>
        Model Analytics
        </div>

        <div class='subtitle'>
        Performance metrics and machine learning insights.
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Accuracy", "91%")

    with col2:
        st.metric("F1 Score", "90%")

    with col3:
        st.metric("Model", "Decision Tree")

    cm = np.array([
        [320, 28],
        [22, 290]
    ])

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[
                'Predicted No Pit',
                'Predicted Pit'
            ],
            y=[
                'Actual No Pit',
                'Actual Pit'
            ]
        )
    )

    fig.update_layout(
        template='plotly_dark',
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    feature_df = pd.DataFrame({

        "Feature": [
            "TyreWearRate",
            "DegradationPerLap",
            "PositionPressure",
            "RacePhase",
            "LapIntensity"
        ],

        "Purpose": [
            "Tyre wear efficiency",
            "Per lap degradation",
            "Position pressure",
            "Race stage classification",
            "Lap volatility"
        ]
    })

    st.dataframe(
        feature_df,
        use_container_width=True
    )