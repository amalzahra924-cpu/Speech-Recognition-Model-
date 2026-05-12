# Speech Recognition Feature Prediction Model

This repository contains a machine learning model designed to predict **RMS Energy** from various audio features extracted from speech data.

## 📊 Model Details
- **Algorithm:** Gradient Boosting Regressor
- **Target Variable:** `rms_energy`
- **Input Features:** - Audio Duration, Sample Rate
  - MFCC Mean, Mel Spectrogram Mean
  - Pitch (Hz), Zero Crossing Rate
  - Delta and Delta-Delta Features

## 📈 Performance
Based on the latest training session, the model achieved:
- **R² Score:** ~0.85
- **Mean Absolute Error (MAE):** ~0.042

## 📁 Files in this Repo
- `speech_model.pkl`: The trained Gradient Boosting model.
- `scaler.pkl`: The StandardScaler used to normalize the input features.

## 🛠️ Usage
To use this model, load the files using `joblib`:
```python
import joblib
model = joblib.load('speech_model.pkl')
scaler = joblib.load('scaler.pkl')
```
