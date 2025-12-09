This repository contains the implementation, trained models, and analysis for a Machine Learning system designed to detect Cyber-Physical Attacks (DoS) and Mechanical Malfunctions in unmanned aerial robots using telemetry sensor data.

A major focus of the project is Explainable AI (XAI). Using SHAP and LIME, we interpret why each model makes a prediction—critical for safety, debugging, and trust in autonomous drone systems.

📂 Project Structure
├── saved-models/ # Trained ML/DL models + scalers
├── data/ # CSV datasets provided for the assignment
├── visual-outputs/ # Have all the outputs
├── AI_Assignment_03.ipynb # Main notebook with preprocessing, training, XAI
├── AAI-Project-Report.pdf # Detailed 10-page analysis & results
└── requirements.txt # Python dependencies

🚀 Key Components

1. Data Preprocessing

Asynchronous stream alignment: Forward-fill (ffill) to synchronize GPS, IMU, and Battery sensors operating at different frequencies.

Noise filtering: Statistical cleaning + StandardScaler to normalize high-variance telemetry.

2. Multi-Model Pipeline

Three architectures were developed and compared:

Model 2.2 — 1D CNN: Learns temporal sensor variations and local dependencies.

Model 2.4 — XGBoost: High-accuracy gradient boosting for structured telemetry features.

Model 2.6 — FNN: Deep neural classifier with BatchNorm + Dropout for regularization.

3. Explainability (XAI)

SHAP: Provides global feature importance (e.g., How does voltage drop influence DoS detection?).

LIME: Gives local explanations for individual abnormal flight segments (e.g., Why was this specific timestamp labeled Malfunction?).

📊 Performance Summary
Model Accuracy Highlights
XGBoost 100% Highest accuracy; strong feature threshold reasoning; fastest to train.
1D-CNN 99.9% Good for vibration/jitter-related malfunction patterns.
FNN 97.2% Reliable non-linear modeling baseline.

Insight: SHAP revealed that Battery Voltage behavior is the strongest indicator of DoS attacks (due to frozen telemetry), whereas Gyroscope Angular Velocity variability signals mechanical failure.

🛠️ Setup & Usage

1. Clone the Repository
   git clone https://github.com/RabiaZubair28/AI-Assignment-03.git
   cd AI-Assignment-03

2. Install Requirements
   pip install -r requirements.txt

3. Load and Use Pretrained Models

Open the main notebook or load models directly:

import joblib
import xgboost as xgb
import tensorflow as tf

# Load scaler and label encoder

scaler = joblib.load('saved_models/scaler.pkl')
le = joblib.load('saved_models/label_encoder.pkl')

# Load XGBoost model

model_xgb = xgb.XGBClassifier()
model_xgb.load_model('saved_models/xgboost_model.json')

# Load FNN model

model_fnn = tf.keras.models.load_model('saved_models/fnn_model.keras')

print("Models loaded successfully!")

📈 Visualizations Included

SHAP Summary Plot — global importance for sensors like Voltage, Yaw Rate, and Acceleration
LIME Local Explanation — detailed reasoning for a single malfunction/attack prediction

Full plots and analysis are available inside:
AAI-Project-Report.pdf && visual-outputs folder
