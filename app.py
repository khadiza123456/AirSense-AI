import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime

from tensorflow.keras.models import load_model


# ============================================================
# AIRSENSE AI
# Deep Learning Based Air Quality Forecasting
# ============================================================


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AirSense AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 44px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    .status-good {
        padding: 14px;
        border-radius: 12px;
        background-color: #d4edda;
        color: #155724;
        font-size: 20px;
        font-weight: 700;
        text-align: center;
        margin-top: 10px;
    }

    .status-moderate {
        padding: 14px;
        border-radius: 12px;
        background-color: #fff3cd;
        color: #856404;
        font-size: 20px;
        font-weight: 700;
        text-align: center;
        margin-top: 10px;
    }

    .status-sensitive {
        padding: 14px;
        border-radius: 12px;
        background-color: #ffe5b4;
        color: #8a4b08;
        font-size: 20px;
        font-weight: 700;
        text-align: center;
        margin-top: 10px;
    }

    .status-unhealthy {
        padding: 14px;
        border-radius: 12px;
        background-color: #f8d7da;
        color: #721c24;
        font-size: 20px;
        font-weight: 700;
        text-align: center;
        margin-top: 10px;
    }

    .status-very-unhealthy {
        padding: 14px;
        border-radius: 12px;
        background-color: #e6d5f7;
        color: #4b1f6f;
        font-size: 20px;
        font-weight: 700;
        text-align: center;
        margin-top: 10px;
    }

    .status-hazardous {
        padding: 14px;
        border-radius: 12px;
        background-color: #343a40;
        color: white;
        font-size: 20px;
        font-weight: 700;
        text-align: center;
        margin-top: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🌍 AirSense AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deep Learning Based Air Quality Forecasting System'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    """
    AirSense AI is an intelligent air-quality forecasting system
    that uses LSTM, RNN and GRU deep learning models to estimate
    PM2.5 concentration and interpret the predicted air-quality level.
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🌍 AirSense AI")

    st.write(
        """
        A deep learning based PM2.5 forecasting
        and weather monitoring application.
        """
    )

    st.divider()

    st.subheader("🧠 Models")
    st.write("• LSTM")
    st.write("• RNN")
    st.write("• GRU")

    st.divider()

    st.subheader("📊 Evaluation")
    st.write("• MAE")
    st.write("• RMSE")
    st.write("• R² Score")

    st.divider()

    st.caption("AirSense AI © 2026")


# ============================================================
# SESSION STATE
# ============================================================

if "weather_loaded" not in st.session_state:
    st.session_state.weather_loaded = False

if "weather_data" not in st.session_state:
    st.session_state.weather_data = None

if "location_info" not in st.session_state:
    st.session_state.location_info = None

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# ============================================================
# LOAD MODELS
# ============================================================

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

    return (
        scaler,
        lstm_model,
        rnn_model,
        gru_model
    )


try:

    scaler, lstm_model, rnn_model, gru_model = load_models()

    st.success(
        "✅ LSTM, RNN and GRU models loaded successfully."
    )

except Exception as e:

    st.error(
        "❌ Unable to load trained models."
    )

    st.code(str(e))
    st.stop()


# ============================================================
# WEATHER CONDITION
# ============================================================

def get_weather_condition(code):

    conditions = {

        0: ("☀️", "Clear Sky"),
        1: ("🌤️", "Mainly Clear"),
        2: ("⛅", "Partly Cloudy"),
        3: ("☁️", "Overcast"),

        45: ("🌫️", "Fog"),
        48: ("🌫️", "Depositing Rime Fog"),

        51: ("🌦️", "Light Drizzle"),
        53: ("🌦️", "Moderate Drizzle"),
        55: ("🌧️", "Dense Drizzle"),

        56: ("🌧️", "Light Freezing Drizzle"),
        57: ("🌧️", "Dense Freezing Drizzle"),

        61: ("🌧️", "Slight Rain"),
        63: ("🌧️", "Moderate Rain"),
        65: ("🌧️", "Heavy Rain"),

        66: ("🌧️", "Light Freezing Rain"),
        67: ("🌧️", "Heavy Freezing Rain"),

        71: ("🌨️", "Slight Snow"),
        73: ("🌨️", "Moderate Snow"),
        75: ("❄️", "Heavy Snow"),

        77: ("❄️", "Snow Grains"),

        80: ("🌦️", "Slight Rain Showers"),
        81: ("🌧️", "Moderate Rain Showers"),
        82: ("⛈️", "Violent Rain Showers"),

        85: ("🌨️", "Slight Snow Showers"),
        86: ("🌨️", "Heavy Snow Showers"),

        95: ("⛈️", "Thunderstorm"),
        96: ("⛈️", "Thunderstorm with Hail"),
        99: ("⛈️", "Thunderstorm with Heavy Hail")
    }

    return conditions.get(
        int(code),
        ("🌍", "Unknown")
    )


# ============================================================
# GEOCODING
# ============================================================

def get_location(location):

    url = (
        "https://geocoding-api.open-meteo.com/v1/search"
    )

    params = {
        "name": location,
        "count": 1,
        "language": "en",
        "format": "json"
    }

    response = requests.get(
        url,
        params=params,
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    if (
        "results" not in data
        or len(data["results"]) == 0
    ):
        return None

    result = data["results"][0]

    return {
        "name": result.get("name", location),
        "country": result.get("country", ""),
        "latitude": result.get("latitude"),
        "longitude": result.get("longitude"),
        "timezone": result.get("timezone", "auto")
    }


# ============================================================
# WEATHER API
# ============================================================

def get_weather(
    latitude,
    longitude,
    timezone
):

    url = (
        "https://api.open-meteo.com/v1/forecast"
    )

    params = {

        "latitude": latitude,
        "longitude": longitude,

        "current":
        "temperature_2m,"
        "relative_humidity_2m,"
        "apparent_temperature,"
        "precipitation,"
        "rain,"
        "weather_code,"
        "surface_pressure,"
        "wind_speed_10m,"
        "dew_point_2m",

        "timezone": timezone
    }

    response = requests.get(
        url,
        params=params,
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    current = data.get("current")

    if current is None:
        return None

    icon, condition = get_weather_condition(
        current.get("weather_code", 0)
    )

    return {

        "temperature":
        current.get("temperature_2m"),

        "humidity":
        current.get("relative_humidity_2m"),

        "feels_like":
        current.get("apparent_temperature"),

        "precipitation":
        current.get("precipitation"),

        "rain":
        current.get("rain"),

        "pressure":
        current.get("surface_pressure"),

        "wind_speed":
        current.get("wind_speed_10m"),

        "dew_point":
        current.get("dew_point_2m"),

        "condition":
        condition,

        "icon":
        icon,

        "weather_code":
        current.get("weather_code"),

        "time":
        current.get("time")
    }


# ============================================================
# LOCATION SEARCH
# ============================================================

st.divider()

st.header(
    "📍 Location & Current Weather"
)

location_col1, location_col2 = st.columns(
    [4, 1]
)

with location_col1:

    location_input = st.text_input(
        "Enter Any Location",
        placeholder=(
            "Example: Gazipur, Dhaka, "
            "Chittagong, London, Dubai..."
        )
    )

with location_col2:

    st.write("")

    search_button = st.button(
        "🔍 Search Weather",
        type="primary",
        use_container_width=True
    )


# ============================================================
# SEARCH WEATHER
# ============================================================

if search_button:

    if not location_input.strip():

        st.warning(
            "⚠️ Please enter a location."
        )

    else:

        with st.spinner(
            "🌍 Finding location and loading weather..."
        ):

            try:

                location_info = get_location(
                    location_input.strip()
                )

                if location_info is None:

                    st.error(
                        "❌ Location not found. "
                        "Please try another city."
                    )

                else:

                    weather_data = get_weather(
                        location_info["latitude"],
                        location_info["longitude"],
                        location_info["timezone"]
                    )

                    if weather_data is None:

                        st.error(
                            "❌ Weather data unavailable."
                        )

                    else:

                        st.session_state.weather_loaded = True

                        st.session_state.location_info = (
                            location_info
                        )

                        st.session_state.weather_data = (
                            weather_data
                        )

                        st.success(
                            "✅ Weather information loaded successfully."
                        )

            except Exception as e:

                st.error(
                    "❌ Unable to retrieve weather data."
                )

                st.caption(
                    str(e)
                )


# ============================================================
# DISPLAY WEATHER
# ============================================================

if st.session_state.weather_loaded:

    location_info = (
        st.session_state.location_info
    )

    weather = (
        st.session_state.weather_data
    )

    st.subheader(
        f"{weather['icon']} "
        f"{location_info['name']}, "
        f"{location_info['country']}"
    )

    w1, w2, w3, w4 = st.columns(4)

    with w1:

        st.metric(
            "🌡️ Temperature",
            f"{weather['temperature']:.1f} °C"
        )

    with w2:

        st.metric(
            "💧 Humidity",
            f"{weather['humidity']:.0f}%"
        )

    with w3:

        st.metric(
            "🌡️ Feels Like",
            f"{weather['feels_like']:.1f} °C"
        )

    with w4:

        st.metric(
            "🌬️ Wind Speed",
            f"{weather['wind_speed']:.1f} km/h"
        )

    w5, w6, w7, w8 = st.columns(4)

    with w5:

        st.metric(
            "☁️ Condition",
            weather["condition"]
        )

    with w6:

        st.metric(
            "🔽 Pressure",
            f"{weather['pressure']:.1f} hPa"
        )

    with w7:

        st.metric(
            "💦 Dew Point",
            f"{weather['dew_point']:.1f} °C"
        )

    with w8:

        st.metric(
            "🌧️ Rain",
            f"{weather['rain']:.1f} mm"
        )

    st.caption(
        f"📍 Coordinates: "
        f"{location_info['latitude']:.4f}, "
        f"{location_info['longitude']:.4f}"
    )

    st.caption(
        f"🕒 Observation Time: "
        f"{weather['time']}"
    )

    st.subheader(
        "🗺️ Selected Location"
    )

    map_df = pd.DataFrame(
        {
            "lat": [
                location_info["latitude"]
            ],

            "lon": [
                location_info["longitude"]
            ]
        }
    )

    st.map(
        map_df,
        zoom=10
    )

else:

    st.info(
        "Enter any location above to view current weather "
        "and its geographical position."
    )


# ============================================================
# AIR QUALITY INPUT
# ============================================================

st.divider()

st.header(
    "🌫️ Air Quality & Meteorological Input"
)

st.write(
    """
    Enter the air-quality parameters below.
    Weather-related parameters are automatically taken
    from the selected location when weather data is available.
    """
)


if st.session_state.weather_loaded:

    weather = st.session_state.weather_data

    default_temp = float(
        weather["temperature"]
    )

    default_pressure = float(
        weather["pressure"]
    )

    default_dewp = float(
        weather["dew_point"]
    )

    default_rain = float(
        weather["rain"]
    )

    default_wind = float(
        weather["wind_speed"]
    )

else:

    default_temp = 20.0
    default_pressure = 1010.0
    default_dewp = 10.0
    default_rain = 0.0
    default_wind = 2.0


c1, c2, c3 = st.columns(3)


with c1:

    pm10 = st.number_input(
        "PM10 (µg/m³)",
        min_value=0.0,
        value=50.0,
        step=1.0
    )

    so2 = st.number_input(
        "SO₂ (µg/m³)",
        min_value=0.0,
        value=10.0,
        step=1.0
    )

    no2 = st.number_input(
        "NO₂ (µg/m³)",
        min_value=0.0,
        value=20.0,
        step=1.0
    )

    co = st.number_input(
        "CO",
        min_value=0.0,
        value=0.5,
        step=0.1
    )


with c2:

    o3 = st.number_input(
        "O₃ (µg/m³)",
        min_value=0.0,
        value=50.0,
        step=1.0
    )

    temp = st.number_input(
        "Temperature (°C)",
        value=default_temp
    )

    pres = st.number_input(
        "Pressure (hPa)",
        value=default_pressure
    )


with c3:

    dewp = st.number_input(
        "Dew Point (°C)",
        value=default_dewp
    )

    rain = st.number_input(
        "Rain (mm)",
        min_value=0.0,
        value=default_rain
    )

    wspm = st.number_input(
        "Wind Speed (km/h)",
        min_value=0.0,
        value=default_wind
    )


# ============================================================
# INPUT SUMMARY
# ============================================================

with st.expander(
    "🔎 View Current Input Summary"
):

    input_summary = pd.DataFrame(
        {
            "Feature": [
                "PM10",
                "SO2",
                "NO2",
                "CO",
                "O3",
                "Temperature",
                "Pressure",
                "Dew Point",
                "Rain",
                "Wind Speed"
            ],

            "Value": [
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
            ]
        }
    )

    st.dataframe(
        input_summary,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PREPARE INPUT
# ============================================================

def prepare_input():

    input_data = np.array(
        [[
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
        ]],
        dtype=float
    )

    input_scaled = scaler.transform(
        input_data
    )

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


# ============================================================
# PM2.5 CLASSIFICATION
# ============================================================

def classify_pm25(value):

    value = max(
        0.0,
        float(value)
    )

    if value <= 9.0:

        return (
            "Good",
            "🟢",
            "Air quality is good.",
            "status-good"
        )

    elif value <= 35.4:

        return (
            "Moderate / Not Bad",
            "🟡",
            "Air quality is acceptable, "
            "but unusually sensitive people may need caution.",
            "status-moderate"
        )

    elif value <= 55.4:

        return (
            "Unhealthy for Sensitive Groups",
            "🟠",
            "Sensitive groups may experience health effects.",
            "status-sensitive"
        )

    elif value <= 125.4:

        return (
            "Unhealthy",
            "🔴",
            "Everyone may begin to experience health effects.",
            "status-unhealthy"
        )

    elif value <= 225.4:

        return (
            "Very Unhealthy",
            "🟣",
            "Health alert conditions.",
            "status-very-unhealthy"
        )

    else:

        return (
            "Hazardous",
            "⚫",
            "Health warning of emergency conditions.",
            "status-hazardous"
        )


# ============================================================
# DISPLAY RESULT
# ============================================================

def display_result(
    model_name,
    prediction
):

    prediction = max(
        0.0,
        float(prediction)
    )

    status, icon, message, css = classify_pm25(
        prediction
    )

    st.metric(
        f"{model_name} Predicted PM2.5",
        f"{prediction:.2f} µg/m³"
    )

    st.markdown(
        f"""
        <div class="{css}">
            {icon} {status}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        message
    )


# ============================================================
# INDIVIDUAL MODEL PREDICTION
# ============================================================

st.divider()

st.header(
    "🔮 PM2.5 Prediction"
)

p1, p2, p3 = st.columns(3)


# ============================================================
# LSTM PREDICTION
# ============================================================

with p1:

    if st.button(
        "🧠 Predict using LSTM",
        use_container_width=True
    ):

        with st.spinner(
            "LSTM is predicting..."
        ):

            sequence = prepare_input()

            prediction = lstm_model.predict(
                sequence,
                verbose=0
            )[0][0]

        prediction = max(
            0.0,
            float(prediction)
        )

        status, _, _, _ = classify_pm25(
            prediction
        )

        st.session_state.prediction_history.append(
            {
                "Time":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

                "Location":
                (
                    st.session_state.location_info["name"]
                    if st.session_state.weather_loaded
                    else "Manual Input"
                ),

                "Model": "LSTM",

                "PM2.5":
                prediction,

                "Status":
                status
            }
        )

        st.success(
            "✅ LSTM Prediction Completed!"
        )

        display_result(
            "LSTM",
            prediction
        )


# ============================================================
# RNN PREDICTION
# ============================================================

with p2:

    if st.button(
        "🔄 Predict using RNN",
        use_container_width=True
    ):

        with st.spinner(
            "RNN is predicting..."
        ):

            sequence = prepare_input()

            prediction = rnn_model.predict(
                sequence,
                verbose=0
            )[0][0]

        prediction = max(
            0.0,
            float(prediction)
        )

        status, _, _, _ = classify_pm25(
            prediction
        )

        st.session_state.prediction_history.append(
            {
                "Time":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

                "Location":
                (
                    st.session_state.location_info["name"]
                    if st.session_state.weather_loaded
                    else "Manual Input"
                ),

                "Model": "RNN",

                "PM2.5":
                prediction,

                "Status":
                status
            }
        )

        st.success(
            "✅ RNN Prediction Completed!"
        )

        display_result(
            "RNN",
            prediction
        )


# ============================================================
# GRU PREDICTION
# ============================================================

with p3:

    if st.button(
        "⚡ Predict using GRU",
        use_container_width=True
    ):

        with st.spinner(
            "GRU is predicting..."
        ):

            sequence = prepare_input()

            prediction = gru_model.predict(
                sequence,
                verbose=0
            )[0][0]

        prediction = max(
            0.0,
            float(prediction)
        )

        status, _, _, _ = classify_pm25(
            prediction
        )

        st.session_state.prediction_history.append(
            {
                "Time":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

                "Location":
                (
                    st.session_state.location_info["name"]
                    if st.session_state.weather_loaded
                    else "Manual Input"
                ),

                "Model": "GRU",

                "PM2.5":
                prediction,

                "Status":
                status
            }
        )

        st.success(
            "✅ GRU Prediction Completed!"
        )

        display_result(
            "GRU",
            prediction
        )


# ============================================================
# MODEL COMPARISON
# ============================================================

st.divider()

st.header(
    "📊 Deep Learning Model Comparison"
)

st.write(
    """
    Compare the PM2.5 predictions produced by
    LSTM, RNN and GRU using the same input data.
    """
)


if st.button(
    "📈 Compare LSTM vs RNN vs GRU",
    type="primary",
    use_container_width=True
):

    with st.spinner(
        "Running all deep learning models..."
    ):

        sequence = prepare_input()

        lstm_prediction = lstm_model.predict(
            sequence,
            verbose=0
        )[0][0]

        rnn_prediction = rnn_model.predict(
            sequence,
            verbose=0
        )[0][0]

        gru_prediction = gru_model.predict(
            sequence,
            verbose=0
        )[0][0]


    lstm_prediction = max(
        0.0,
        float(lstm_prediction)
    )

    rnn_prediction = max(
        0.0,
        float(rnn_prediction)
    )

    gru_prediction = max(
        0.0,
        float(gru_prediction)
    )


    comparison_df = pd.DataFrame(
        {
            "Model": [
                "LSTM",
                "RNN",
                "GRU"
            ],

            "Predicted PM2.5": [
                lstm_prediction,
                rnn_prediction,
                gru_prediction
            ]
        }
    )


    st.subheader(
        "📋 Prediction Results"
    )

    st.dataframe(
        comparison_df.style.format(
            {
                "Predicted PM2.5":
                "{:.2f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


    st.subheader(
        "📊 Prediction Comparison Chart"
    )

    chart_df = comparison_df.set_index(
        "Model"
    )

    st.bar_chart(
        chart_df
    )


    st.subheader(
        "🌍 Air Quality Status"
    )

    r1, r2, r3 = st.columns(3)

    with r1:

        st.markdown(
            "### 🧠 LSTM"
        )

        display_result(
            "LSTM",
            lstm_prediction
        )

    with r2:

        st.markdown(
            "### 🔄 RNN"
        )

        display_result(
            "RNN",
            rnn_prediction
        )

    with r3:

        st.markdown(
            "### ⚡ GRU"
        )

        display_result(
            "GRU",
            gru_prediction
        )


    predictions = {

        "LSTM":
        lstm_prediction,

        "RNN":
        rnn_prediction,

        "GRU":
        gru_prediction
    }


    lowest_model = min(
        predictions,
        key=predictions.get
    )

    highest_model = max(
        predictions,
        key=predictions.get
    )


    st.subheader(
        "🔎 Prediction Summary"
    )

    s1, s2 = st.columns(2)

    with s1:

        st.success(
            f"Lowest predicted PM2.5: "
            f"{lowest_model} — "
            f"{predictions[lowest_model]:.2f} µg/m³"
        )

    with s2:

        st.info(
            f"Highest predicted PM2.5: "
            f"{highest_model} — "
            f"{predictions[highest_model]:.2f} µg/m³"
        )


    location_name = "Manual Input"

    if st.session_state.weather_loaded:

        location_name = (
            st.session_state.location_info[
                "name"
            ]
        )


    for model_name, prediction in predictions.items():

        status, _, _, _ = classify_pm25(
            prediction
        )

        st.session_state.prediction_history.append(
            {
                "Time":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

                "Location":
                location_name,

                "Model":
                model_name,

                "PM2.5":
                prediction,

                "Status":
                status
            }
        )


# ============================================================
# ACTUAL MODEL PERFORMANCE METRICS
# ============================================================

st.divider()

st.header(
    "🏆 Model Performance"
)

st.write(
    """
    The following evaluation results were obtained from
    the test data after training the three deep learning models.
    Lower MAE and RMSE indicate better performance, while a
    higher R² score indicates better performance.
    """
)


MODEL_METRICS = {

    "LSTM": {
        "MAE": 18.415856375170975,
        "RMSE": 32.30336745816216,
        "R2": 0.8503018510455332
    },

    "RNN": {
        "MAE": 18.945461417151922,
        "RMSE": 32.84085751086531,
        "R2": 0.8452788046707861
    },

    "GRU": {
        "MAE": 18.13983097590164,
        "RMSE": 32.04416398003384,
        "R2": 0.8526945802507597
    }
}


metrics_df = pd.DataFrame(
    [
        {
            "Model": model,
            "MAE": values["MAE"],
            "RMSE": values["RMSE"],
            "R² Score": values["R2"]
        }

        for model, values
        in MODEL_METRICS.items()
    ]
)


st.dataframe(
    metrics_df.style.format(
        {
            "MAE": "{:.4f}",
            "RMSE": "{:.4f}",
            "R² Score": "{:.4f}"
        }
    ),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# PERFORMANCE CHARTS
# ============================================================

st.subheader(
    "📊 MAE Comparison"
)

mae_chart = metrics_df.set_index(
    "Model"
)[["MAE"]]

st.bar_chart(
    mae_chart
)


st.subheader(
    "📊 RMSE Comparison"
)

rmse_chart = metrics_df.set_index(
    "Model"
)[["RMSE"]]

st.bar_chart(
    rmse_chart
)


st.subheader(
    "📊 R² Score Comparison"
)

r2_chart = metrics_df.set_index(
    "Model"
)[["R² Score"]
]

st.bar_chart(
    r2_chart
)


# ============================================================
# BEST MODEL CALCULATION
# ============================================================

performance_df = metrics_df.copy()

performance_df["MAE_Rank"] = (
    performance_df["MAE"].rank(
        ascending=True
    )
)

performance_df["RMSE_Rank"] = (
    performance_df["RMSE"].rank(
        ascending=True
    )
)

performance_df["R2_Rank"] = (
    performance_df["R² Score"].rank(
        ascending=False
    )
)

performance_df["Total Rank"] = (
    performance_df["MAE_Rank"]
    +
    performance_df["RMSE_Rank"]
    +
    performance_df["R2_Rank"]
)


best_model = performance_df.loc[
    performance_df["Total Rank"].idxmin(),
    "Model"
]


best_row = performance_df[
    performance_df["Model"] == best_model
].iloc[0]


st.success(
    f"""
    🏆 Best Performing Model: {best_model}

    MAE: {best_row["MAE"]:.4f}

    RMSE: {best_row["RMSE"]:.4f}

    R² Score: {best_row["R² Score"]:.4f}

    GRU achieved the lowest MAE and RMSE and the highest
    R² score among the three evaluated models.
    """
)


# ============================================================
# PM2.5 INTERPRETATION TABLE
# ============================================================

st.divider()

st.header(
    "🌍 PM2.5 Air Quality Interpretation"
)

interpretation_df = pd.DataFrame(
    {
        "PM2.5 Range (µg/m³)": [
            "0 – 9.0",
            "9.1 – 35.4",
            "35.5 – 55.4",
            "55.5 – 125.4",
            "125.5 – 225.4",
            "> 225.4"
        ],

        "Air Quality": [
            "Good",
            "Moderate / Not Bad",
            "Unhealthy for Sensitive Groups",
            "Unhealthy",
            "Very Unhealthy",
            "Hazardous"
        ],

        "Indicator": [
            "🟢",
            "🟡",
            "🟠",
            "🔴",
            "🟣",
            "⚫"
        ]
    }
)


st.dataframe(
    interpretation_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# PREDICTION HISTORY
# ============================================================

st.divider()

st.header(
    "📋 Prediction History"
)

if len(
    st.session_state.prediction_history
) > 0:

    history_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    st.dataframe(
        history_df.style.format(
            {
                "PM2.5": "{:.2f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    if st.button(
        "🗑️ Clear Prediction History"
    ):

        st.session_state.prediction_history = []

        st.rerun()

else:

    st.info(
        "No predictions have been made yet."
    )


# ============================================================
# DOWNLOAD REPORT
# ============================================================

st.divider()

st.header(
    "📥 Download Prediction Report"
)

if len(
    st.session_state.prediction_history
) > 0:

    report_df = pd.DataFrame(
        st.session_state.prediction_history
    )

    csv_data = report_df.to_csv(
        index=False
    )

    st.download_button(
        label="📥 Download Prediction Report",
        data=csv_data,
        file_name="AirSense_AI_Prediction_Report.csv",
        mime="text/csv",
        use_container_width=True
    )

else:

    st.info(
        "Make at least one prediction to generate a report."
    )


# ============================================================
# ABOUT AIRSENSE AI
# ============================================================

st.divider()

st.header(
    "ℹ️ About AirSense AI"
)

st.write(
    """
    AirSense AI is a deep learning based PM2.5 forecasting
    system designed to analyze air-quality and meteorological
    parameters.

    The system integrates LSTM, RNN and GRU models for
    PM2.5 prediction and provides an interpretable
    air-quality category based on the predicted value.

    The application also provides location-based current
    weather information including temperature, humidity,
    weather condition, wind speed, pressure, dew point
    and rainfall.
    """
)


# ============================================================
# INPUT FEATURES
# ============================================================

st.subheader(
    "📌 Input Features"
)

feature_df = pd.DataFrame(
    {
        "Feature": [
            "PM10",
            "SO₂",
            "NO₂",
            "CO",
            "O₃",
            "Temperature",
            "Pressure",
            "Dew Point",
            "Rain",
            "Wind Speed"
        ],

        "Purpose": [
            "Air pollutant concentration",
            "Sulfur dioxide concentration",
            "Nitrogen dioxide concentration",
            "Carbon monoxide concentration",
            "Ozone concentration",
            "Meteorological condition",
            "Atmospheric pressure",
            "Moisture-related weather parameter",
            "Rainfall measurement",
            "Wind movement"
        ]
    }
)


st.dataframe(
    feature_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# MODELS USED
# ============================================================

st.subheader(
    "🧠 Deep Learning Models Used"
)

model_info_df = pd.DataFrame(
    {
        "Model": [
            "LSTM",
            "RNN",
            "GRU"
        ],

        "Full Name": [
            "Long Short-Term Memory",
            "Recurrent Neural Network",
            "Gated Recurrent Unit"
        ]
    }
)


st.dataframe(
    model_info_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# EVALUATION METRICS
# ============================================================

st.subheader(
    "📊 Evaluation Metrics"
)

st.write(
    """
    • MAE — Mean Absolute Error

    • RMSE — Root Mean Squared Error

    • R² — Coefficient of Determination
    """
)


# ============================================================
# FINAL MESSAGE
# ============================================================

st.divider()

st.success(
    """
    🌍 AirSense AI is ready for location-based weather
    monitoring, PM2.5 forecasting, model comparison,
    air-quality interpretation and prediction reporting.
    """
)
