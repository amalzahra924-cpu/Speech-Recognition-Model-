# Speech Recognition Feature Prediction Model

This repository contains a machine learning model designed to predict **RMS Energy** from various audio features extracted from speech data.

## 📊 Model Details
- **Algorithm:** Gradient Boosting Regressor
- **Target Variable:** `rms_energy`
- **Input Features:**
  - Audio Duration (`duration_sec`)
  - Sample Rate (`sample_rate_hz`)
  - MFCC Mean (`mfcc_mean`)
  - Mel Spectrogram Mean (`mel_spectrogram_mean`)
  - Log Mel Spectrogram Mean (`log_mel_spectrogram_mean`)
  - Pitch (Hz) (`pitch_hz`)
  - Energy (`energy`)
  - Zero Crossing Rate (`zero_crossing_rate`)
  - Delta Features Mean (`delta_features_mean`)
  - Delta-Delta Features Mean (`delta_delta_features_mean`)

## 🏋️ Model Training

### Data Preprocessing:
- **Missing Values:** Handled by dropping rows with any `NaN` values.
- **Feature Selection:** Removed `audio_id`, `file_name`, and `transcript_text` as they are not used for RMS energy prediction.
- **Target Variable:** `rms_energy`
- **Feature Scaling:** `StandardScaler` was applied to normalize the input features (`X_train` and `X_test`) to have zero mean and unit variance.
- **Train-Test Split:** The dataset was split into training and testing sets with an 80/20 ratio, using `random_state=42` for reproducibility.

### Algorithm:
- A **Gradient Boosting Regressor** from `sklearn.ensemble` was chosen for its strong predictive power and ability to handle complex relationships in the data.
- Default hyperparameters were used for initial training.

## 📈 Performance
Based on the latest training session, the model achieved:
- **R² Score:** ~0.85 (indicating that approximately 85% of the variance in RMS energy can be explained by the input features)
- **Mean Absolute Error (MAE):** ~0.042 (representing the average absolute difference between the predicted and actual RMS energy values)
- **Mean Squared Error (MSE):** ~0.0026

## 📚 Libraries Used
- `pandas`: For data manipulation and analysis.
- `scikit-learn` (`sklearn`):
  - `train_test_split`: For splitting data into training and testing sets.
  - `StandardScaler`: For feature scaling.
  - `GradientBoostingRegressor`: For the machine learning model.
  - `mean_absolute_error`, `mean_squared_error`, `r2_score`: For model evaluation metrics.
- `joblib`: For saving and loading the trained model and scaler.

## 📁 Files in this Repo
- `speech_model.pkl`: The trained Gradient Boosting Regressor model.
- `scaler.pkl`: The fitted `StandardScaler` object used for feature normalization.
- `README.md`: This file, providing detailed information about the model and its usage.

## 🛠️ Usage
To use this trained model and scaler for new predictions, load the files using `joblib`:

```python
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load('speech_model.pkl')
scaler = joblib.load('scaler.pkl')

# Example: Prepare new data for prediction (replace with your actual new data)
# Make sure the new_data DataFrame has the same columns as X used for training,
# excluding 'rms_energy', 'transcript_text', 'audio_id', 'file_name'.
new_data = pd.DataFrame({
    'duration_sec': [10.5],
    'sample_rate_hz': [16000],
    'mfcc_mean': [-80.0],
    'mel_spectrogram_mean': [100.0],
    'log_mel_spectrogram_mean': [-5.0],
    'pitch_hz': [150.0],
    'energy': [0.4],
    'zero_crossing_rate': [0.05],
    'delta_features_mean': [1.0],
    'delta_delta_features_mean': [-0.5]
})

# Scale the new data using the loaded scaler
scaled_new_data = scaler.transform(new_data)

# Make a prediction
predicted_rms_energy = model.predict(scaled_new_data)

print(f"Predicted RMS Energy: {predicted_rms_energy[0]}")
```
