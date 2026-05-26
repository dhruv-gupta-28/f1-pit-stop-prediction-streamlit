# 🏎️ F1 Pit Stop Prediction System

A machine learning-powered Formula 1 pit stop prediction web application built with Streamlit. Predicts whether a Formula 1 driver will pit on the next lap using real race telemetry and tyre degradation data.

**[🚀 Try Live Demo](https://f1-pit-stop-prediction.streamlit.app/)**

## Overview

This interactive application leverages supervised machine learning to provide real-time pit stop predictions for Formula 1 races. It combines feature engineering with a trained Decision Tree classifier to analyze race conditions and make informed predictions about pit stop timing.

## Key Features

### 🎯 Real-Time Prediction

- Instant pit stop decision predictions
- Interactive racing dashboard with intuitive controls
- Formula 1 themed UI with professional styling

### 📊 Batch Prediction

- Upload CSV files with multiple race entries
- Generate predictions for entire datasets
- Download results in standard CSV format

### 🛠️ Smart Feature Engineering

Automatically engineered race-specific features:

- **TyreWearRate** - Tyre wear intensity over race progression
- **DegradationPerLap** - Tyre degradation efficiency
- **PositionPressure** - Strategic pressure during later race stages
- **RacePhase** - Race stage categorization (Early, Mid, Late)
- **LapIntensity** - Sudden changes in lap performance

### 🤖 Production Machine Learning Pipeline

- Label Encoding for categorical features
- Standard Scaling for numerical normalization
- Optimized Decision Tree Classifier (86% accuracy)

## Tech Stack

| Component               | Technology    |
| :---------------------- | :------------ |
| **Backend**             | Python 3      |
| **Web Framework**       | Streamlit     |
| **ML Library**          | Scikit-Learn  |
| **Data Processing**     | Pandas, NumPy |
| **Model Serialization** | Pickle        |

## Getting Started

### Prerequisites

- Python 3.8+
- pip or conda package manager

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/f1-pit-stop-prediction-streamlit.git
   cd f1-pit-stop-prediction-streamlit
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**

   ```bash
   streamlit run app.py
   ```

4. **Access the app:**
   - Open your browser to `http://localhost:8501`

## How to Use

### Single Race Prediction

1. Navigate to the **Prediction** tab
2. Enter race parameters:
   - Driver name
   - Tyre compound
   - Race name and year
   - Current pit stop count
   - Lap number and stint
   - Tyre life and position
   - Lap time metrics
   - Degradation and race progress
3. Click **Predict Pit Stop**
4. View instant prediction result with confidence metrics

### Batch Prediction

1. Navigate to the **Batch Prediction** tab
2. Upload your CSV file containing race data
3. Ensure CSV has all required columns
4. Click **Process & Predict**
5. Review predictions in the output table
6. Download results as CSV

## Project Structure

```
f1-pit-stop-prediction-streamlit/
│
├── app.py                    # Main Streamlit application
├── model.py                  # ML pipeline and utilities
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── .gitignore               # Git ignore rules
│
├── data/
│   ├── train.csv            # Training dataset
│   └── test.csv             # Test dataset
│
└── artifacts/
    ├── model.pkl            # Trained Decision Tree model
    ├── scaler.pkl           # StandardScaler artifact
    ├── encoders.pkl         # Label encoders for categorical features
    └── features.pkl         # Feature metadata
```

## Model Performance

The underlying Decision Tree Classifier delivers excellent prediction accuracy:

| Metric                | Score |
| :-------------------- | :---- |
| Accuracy              | 86%   |
| Weighted F1 Score     | 86%   |
| Recall (Pit Stops)    | 81%   |
| Precision (Pit Stops) | 60%   |

**Classification Report:**

| Class       | Precision | Recall | F1-Score |
| :---------- | :-------: | :----: | :------: |
| No Pit Stop |   0.95    |  0.87  |   0.91   |
| Pit Stop    |   0.60    |  0.81  |   0.69   |

## Architecture

```
┌─────────────────────────────────────┐
│      Streamlit Web Interface        │
├─────────────────────────────────────┤
│                                     │
│  ┌─ Single Prediction ─┐            │
│  │  ┌─ Batch Prediction ┐           │
│  │  │                   │           │
│  └──┴───────────────────┘           │
│           ↓                         │
│    ┌──────────────────┐             │
│    │   model.py       │             │
│    ├──────────────────┤             │
│    │ • Preprocessing  │             │
│    │ • Encoding       │             │
│    │ • Scaling        │             │
│    │ • Prediction     │             │
│    └────────┬─────────┘             │
│             ↓                       │
│    ┌──────────────────┐             │
│    │ Trained Artifacts│             │
│    ├──────────────────┤             │
│    │ • Model.pkl      │             │
│    │ • Scaler.pkl     │             │
│    │ • Encoders.pkl   │             │
│    │ • Features.pkl   │             │
│    └──────────────────┘             │
│                                     │
└─────────────────────────────────────┘
```

## Features Explained

### Input Features

The model accepts the following race parameters:

- **Driver & Race Info:** Driver name, race name, year, compound
- **Race Status:** Current pit stops, lap number, stint, position
- **Tyre Metrics:** Tyre life, cumulative degradation
- **Performance:** Lap time, lap time delta
- **Progress:** Race progress percentage, position change

### Engineered Features

These are automatically calculated from inputs:

1. **TyreWearRate** = Tyre Life / (Lap Number + 1)
2. **DegradationPerLap** = Cumulative Degradation / (Lap Number + 1)
3. **PositionPressure** = Position × Race Progress
4. **RacePhase** = Categorized from Race Progress
5. **LapIntensity** = |Lap Time Delta|

## Use Cases

- **Race Strategy Teams** - Real-time pit stop decision support
- **Analysts** - Race pattern analysis and prediction
- **Enthusiasts** - Learn about F1 strategy optimization
- **Education** - ML and sports analytics demonstration

## Limitations

- Model trained on historical F1 data
- Predictions depend on data accuracy
- Real-time F1 variables may not be accounted for
- Performance varies by track and season

## Future Enhancements

- Advanced ensemble models (XGBoost, Random Forest)
- Historical race data visualization
- Driver performance comparison
- Real-time race telemetry integration
- SMOTE for better imbalance handling
- API endpoint for external integration

## Dependencies

All required packages are listed in `requirements.txt`:

```
streamlit
scikit-learn
pandas
numpy
pickle
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Author

**Dhruv Gupta**  
B.Tech Computer Science Engineering  
Government Engineering College Ujjain (GECU)

## Acknowledgments

- Formula 1 community for race data
- Scikit-Learn team for ML tools
- Streamlit for web framework
- Original dataset from Kaggle

---

**[🚀 Try the Live Demo](https://f1-pit-stop-prediction.streamlit.app/)**
