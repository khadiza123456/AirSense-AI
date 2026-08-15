
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model


# ==============================
# Page Configuration
# ==============================

st.set_page_config(
    page_title="AirSense AI",
    page_icon="🌍",
    layout="wide"
)


# ==============================
# Title
# ==============================

st.title("🌍 AirSense AI")
st.subheader("Deep Learning Based Air Quality Forecasting")

st.write(
    "AirSense AI predicts PM2.5 concentration using "
    "LSTM, RNN and GRU deep learning models."
)


# ==============================
# Load Models
# ==============================

@st.cache_resource
def load_models():

    scaler = joblib.load(
        "models/scaler.pkl"
    )

    lstm_model = load_model(
        "models/lstm_model.h5",
        compile=False
    )

    rnn_model = load_model(
        "models/rnn_model.h5",
        compile=False
    )

    gru_model = load_model(
        "models/gru_model.h5",
        compile=False
    )

    return scaler, lstm_model, rnn_model, gru_model


scaler, lstm_model, rnn_model, gru_model = load_models()


# ==============================
# Air Quality Input
# ==============================

st.header("🌫️ Air Quality Input")

col1, col2, col3 = st.columns(3)


with col1:

    pm10 = st.number_input(
        "PM10",
        value=50.0
    )

    so2 = st.number_input(
        "SO2",
        value=10.0
    )

    no2 = st.number_input(
        "NO2",
        value=20.0
    )

    co = st.number_input(
        "CO",
        value=0.5
    )


with col2:

    o3 = st.number_input(
        "O3",
        value=50.0
    )

    temp = st.number_input(
        "Temperature (°C)",
        value=20.0
    )

    pres = st.number_input(
        "Pressure",
        value=1010.0
    )


with col3:

    dewp = st.number_input(
        "Dew Point",
        value=10.0
    )

    rain = st.number_input(
        "Rain",
        value=0.0
    )

    wspm = st.number_input(
        "Wind Speed",
        value=2.0
    )


st.divider()


st.info(
    "Enter the air quality and weather parameters, "
    "then select a prediction option."
)


# ==============================
# Prepare Input Function
# ==============================

def prepare_input():

    input_data = np.array([[
        pm10,
        so2,
        no2,
        co,
        o3,
        temp,
        pres,
        dewp,
        rain,
        wspm
    ]])

    input_scaled = scaler.transform(
        input_data
    )

    # Create 24-hour sequence
    sequence = np.tile(
        input_scaled,
        (24, 1)
    )

    sequence = sequence.reshape(
        1,
        24,
        10
    )

    return sequence


# ==============================
# Individual Predictions
# ==============================

st.header("🔮 PM2.5 Prediction")


col1, col2, col3 = st.columns(3)


# ==============================
# LSTM Prediction
# ==============================

with col1:

    if st.button(
        "🧠 Predict using LSTM",
        use_container_width=True
    ):

        lstm_input = prepare_input()

        lstm_prediction = lstm_model.predict(
            lstm_input,
            verbose=0
        )[0][0]

        st.success(
            "LSTM Prediction Completed!"
        )

        st.metric(
            "LSTM Predicted PM2.5",
            f"{lstm_prediction:.2f}"
        )


# ==============================
# RNN Prediction
# ==============================

with col2:

    if st.button(
        "🔄 Predict using RNN",
        use_container_width=True
    ):

        rnn_input = prepare_input()

        rnn_prediction = rnn_model.predict(
            rnn_input,
            verbose=0
        )[0][0]

        st.success(
            "RNN Prediction Completed!"
        )

        st.metric(
            "RNN Predicted PM2.5",
            f"{rnn_prediction:.2f}"
        )


# ==============================
# GRU Prediction
# ==============================

with col3:

    if st.button(
        "⚡ Predict using GRU",
        use_container_width=True
    ):

        gru_input = prepare_input()

        gru_prediction = gru_model.predict(
            gru_input,
            verbose=0
        )[0][0]

        st.success(
            "GRU Prediction Completed!"
        )

        st.metric(
            "GRU Predicted PM2.5",
            f"{gru_prediction:.2f}"
        )


# ==============================
# Model Comparison
# ==============================

st.divider()

st.header("📊 Deep Learning Model Comparison")


if st.button(
    "📈 Compare LSTM vs RNN vs GRU",
    type="primary",
    use_container_width=True
):

    input_sequence = prepare_input()


    # LSTM Prediction

    lstm_prediction = lstm_model.predict(
        input_sequence,
        verbose=0
    )[0][0]


    # RNN Prediction

    rnn_prediction = rnn_model.predict(
        input_sequence,
        verbose=0
    )[0][0]


    # GRU Prediction

    gru_prediction = gru_model.predict(
        input_sequence,
        verbose=0
    )[0][0]


    # Comparison Data

    comparison_data = {

        "Model": [
            "LSTM",
            "RNN",
            "GRU"
        ],

        "Predicted PM2.5": [
            float(lstm_prediction),
            float(rnn_prediction),
            float(gru_prediction)
        ]
    }


    comparison_df = pd.DataFrame(
        comparison_data
    )


    st.subheader(
        "Prediction Comparison"
    )


    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )


    st.bar_chart(
        comparison_df.set_index(
            "Model"
        )
    )


# ==============================
# Air Quality Interpretation
# ==============================

st.divider()

st.header("🌍 Air Quality Interpretation")

st.write(
    "The predicted PM2.5 value can be used "
    "to understand the current air quality condition."
)


# ==============================
# About AirSense AI
# ==============================

st.divider()

st.header("About AirSense AI")

st.write(
    """
    AirSense AI is a deep learning based air quality
    forecasting system developed for PM2.5 prediction.

    The system uses air quality and meteorological
    parameters to estimate PM2.5 concentration.
    """
)


# ==============================
# Input Features
# ==============================

st.subheader("Input Features")

st.write(
    """
    PM10, SO2, NO2, CO, O3, Temperature,
    Pressure, Dew Point, Rainfall, and Wind Speed.
    """
)


# ==============================
# Models Used
# ==============================

st.subheader("Deep Learning Models Used")

st.write(
    """
    • LSTM — Long Short-Term Memory

    • RNN — Recurrent Neural Network

    • GRU — Gated Recurrent Unit
    """
)


# ==============================
# Evaluation Metrics
# ==============================

st.subheader("Evaluation Metrics")

st.write(
    """
    • MAE — Mean Absolute Error

    • RMSE — Root Mean Squared Error

    • R² Score — Coefficient of Determination
    """
)


# ==============================
# Final Message
# ==============================

st.divider()

st.success(
    "AirSense AI is ready for LSTM, RNN and GRU based PM2.5 forecasting."
)
