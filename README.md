# 🏎️ F1 Pit Stop Prediction System

A Machine Learning powered Formula 1 pit stop prediction system built using:

- Python
- Scikit-Learn
- Streamlit
- Pandas
- NumPy

The application predicts whether a Formula 1 driver is likely to pit on the next lap using race telemetry and tyre degradation data.

---

# 🚀 Features

## ✅ Real-Time Prediction

- Predict pit stop decisions instantly
- Interactive racing dashboard
- Formula 1 themed UI

## ✅ Batch Prediction

- Upload CSV files
- Generate predictions for multiple race entries
- Download prediction results

## ✅ Feature Engineering

Custom engineered racing features:

- TyreWearRate
- DegradationPerLap
- PositionPressure
- RacePhase
- LapIntensity

## ✅ Machine Learning Pipeline

- Label Encoding
- Standard Scaling
- Decision Tree Classifier

---

# 📂 Project Structure

```text
project/
│
├── app.py
├── model.py
├── train.csv
├── test.csv
│
├── artifacts/
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── encoders.pkl
│   └── features.pkl
│
├── requirements.txt
├── .gitignore
└── README.md
```
