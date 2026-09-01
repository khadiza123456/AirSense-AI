# 🌍 AirSense AI — PM2.5 Air Quality Forecasting

**AirSense AI** is a deep learning-based air quality forecasting project designed to predict **PM2.5 (Particulate Matter 2.5)** levels using historical air quality data.

The project applies machine learning and deep learning techniques to analyze air quality patterns and generate PM2.5 predictions. It also includes a **Streamlit web application** for an interactive user experience.

---

## 🎯 Project Objectives

* Predict future **PM2.5 concentration** using historical air quality data.
* Develop and evaluate deep learning models for air quality forecasting.
* Compare the performance of **LSTM, RNN, and GRU** models.
* Analyze model performance using standard regression evaluation metrics.
* Provide an interactive interface through **Streamlit**.
* Present forecasting results and visualizations in an easy-to-understand way.

---

## 📊 Dataset

The project uses a historical air quality dataset containing air pollutant and meteorological information.

### Main Features

* PM2.5
* PM10
* SO₂
* NO₂
* CO
* O₃
* Temperature
* Pressure
* Dew Point
* Rain
* Wind Speed

The dataset is stored in:

```text
Data/
└── raw/
    └── dataset.csv
```

---

## 🤖 Machine Learning & Deep Learning Models

AirSense AI includes the following forecasting models:

### LSTM

Long Short-Term Memory (LSTM) is used to capture long-term dependencies and temporal patterns in air quality data.

### RNN

Recurrent Neural Network (RNN) is used as a baseline sequential deep learning model for time-series forecasting.

### GRU

Gated Recurrent Unit (GRU) is used as an efficient recurrent architecture for learning temporal dependencies.

The trained models are stored in:

```text
Models/
├── GRU Model.h5
├── LSTM Model.h5
└── RNN Model.h5
```

---

## 📈 Evaluation Metrics

The forecasting models can be evaluated using:

* **MAE** — Mean Absolute Error
* **RMSE** — Root Mean Squared Error
* **R² Score** — Coefficient of Determination
* **MAPE** — Mean Absolute Percentage Error

These metrics are used to compare the prediction performance of the different models.

---

## 📉 Visualization

The project includes several visualization and analysis plots.

All generated plots are available in:

```text
Plots/
```

These visualizations help analyze:

* Actual vs. predicted PM2.5 values
* Training performance
* Model performance
* Air quality trends
* Forecasting results

---

## 🖥️ Streamlit Application

AirSense AI includes an interactive **Streamlit web application** that allows users to interact with the air quality forecasting system.

The application provides:

* PM2.5 forecasting
* Model-based prediction
* Air quality interpretation
* Interactive user interface
* Prediction results and visualization

The main application file is:

```text
app.py
```

---

## 📁 Project Structure

```text
AirSense-AI/
│
├── app/
│
├── Data/
│   ├── raw/
│   │   └── dataset.csv
│   └── Redmi.test/
│
├── Models/
│   ├── GRU Model.h5
│   ├── LSTM Model.h5
│   └── RNN Model.h5
│
├── Notebooks/
│   └── YS AI Project.ipynb
│
├── Output/
│
├── Plots/
│   ├── plot1.png
│   ├── plot2.png
│   ├── plot3.png
│   ├── plot4.png
│   └── plot5.png
│
├── main.py
├── app.py
├── Requirements.txt
├── README.md
└── .gitignore
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **TensorFlow**
* **Keras**
* **Matplotlib**
* **XGBoost**
* **Streamlit**
* **Jupyter Notebook**
* **Google Colab**

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/khadiza123456/AirSense-AI.git
```

Move into the project directory:

```bash
cd AirSense-AI
```

Install the required dependencies:

```bash
pip install -r Requirements.txt
```

---

## ▶️ Run the Application

To launch the Streamlit application locally:

```bash
streamlit run app.py
```

After running the command, open the local URL provided by Streamlit in your browser.

---

## ☁️ Streamlit Community Cloud Deployment

The application can also be deployed using **Streamlit Community Cloud**.

### Deployment Steps

1. Connect your GitHub repository to Streamlit Community Cloud.
2. Select the `AirSense-AI` repository.
3. Select `app.py` as the main application file.
4. Configure the required Python version if necessary.
5. Deploy the application.

---

## 🔬 Research & Development

This project demonstrates the application of deep learning techniques to **time-series air quality forecasting**.

The project focuses on understanding temporal patterns in air pollution data and evaluating recurrent neural network architectures for PM2.5 prediction.

---

## 👩‍💻 Author

**Khadiza Begum**

Department of Software Engineering
Daffodil International University

### Connect

* GitHub: https://github.com/khadiza123456
* LinkedIn: https://www.linkedin.com/in/khadiza-b-52618b3a2/

---

## ⭐ Project

**AirSense AI — Deep Learning Based PM2.5 Air Quality Forecasting**

*An academic project focused on air quality forecasting using machine learning and deep learning techniques.*
