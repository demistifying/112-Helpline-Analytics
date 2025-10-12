# 🚨 112-Analytics - Goa Police Emergency Call Analytics System

Advanced predictive analytics dashboard for Goa Police 112 emergency helpline with real-time monitoring, forecasting, and intelligent insights.

## 🎯 Features

### 📊 Analytics Dashboard
- **Real-time KPIs**: Total calls, daily averages, peak hours, location coverage
- **Spatial Analysis**: Interactive maps with incident clustering and heatmaps
- **Temporal Analysis**: Daily trends, hourly patterns, high-activity day detection
- **Category Analysis**: Incident type distribution and statistics

### 🔮 Predictive Forecasting
1. **Call Volume Forecasting**
   - Gradient Boosting Regressor with festival-aware predictions
   - 7-day rolling averages and lag features
   - Accuracy: ~85-90%

2. **Event Type Prediction**
   - XGBoost Classifier for incident category distribution
   - 5 temporal features with StandardScaler normalization
   - Multi-class classification with weighted F1-score

3. **Peak Hour Prediction** ⭐
   - Advanced Gradient Boosting with 19 temporal features
   - Cyclical encodings, multi-scale lags, rolling windows
   - Hour-specific statistical patterns
   - Accuracy: ~90-95%

### 🔐 Authentication & Access Control
- Firebase Authentication integration
- Role-based access (Officers, Inspectors, Superintendents)
- Caller entry dashboard for authorized personnel
- Alert creation system for top officers

## 🏗️ Architecture

```
112-Analytics/
├── app.py                          # Main Streamlit application
├── auth_ui.py                      # Authentication UI components
├── caller_entry.py                 # Caller entry dashboard
├── data_integration.py             # Data integration utilities
├── modules/
│   ├── predictive_models.py        # ML models (GBR, XGBoost)
│   ├── feature_engineering.py      # Feature creation pipeline
│   ├── data_loader.py              # Data loading utilities
│   ├── analysis.py                 # Statistical analysis
│   ├── mapping.py                  # Geospatial visualizations
│   ├── spike_detection.py          # High-activity day detection
│   └── festivals_*.py              # Festival detection modules
├── data/                           # Dataset directory
└── requirements.txt                # Python dependencies
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Firebase project with Firestore enabled
- Git

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd 112-Analytics
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure Firebase**
- Create a Firebase project at https://console.firebase.google.com
- Enable Firestore Database
- Download `serviceAccountKey.json` from Project Settings > Service Accounts
- Place it in the project root directory

5. **Add dataset**
- Place your CSV dataset in `data/Dummy_Dataset_Full_Standardized.csv`
- Required columns: `call_ts`, `category`, `jurisdiction`, `latitude`, `longitude`

## 🎮 Usage

### Run the application
```bash
streamlit run app.py
```

### Access the dashboard
- Open browser at `http://localhost:8501`
- Register as a new officer or login with existing credentials
- Navigate through Analytics Dashboard and Predictive Forecasting tabs

## 🤖 Machine Learning Models

### 1. Call Volume Forecasting
**Algorithm**: Gradient Boosting Regressor
- **Features**: 7 (day_of_week, month, is_weekend, is_festival, lag_1, lag_7, rolling_7)
- **Hyperparameters**: 200 estimators, lr=0.1, max_depth=5
- **Metrics**: MAE, RMSE, MAPE, Accuracy

### 2. Event Type Prediction
**Algorithm**: XGBoost Classifier
- **Features**: 5 (hour, day_of_week, month, is_weekend, is_night)
- **Hyperparameters**: 100 estimators, lr=0.1, max_depth=6
- **Metrics**: Accuracy, F1-Score (weighted)

### 3. Peak Hour Prediction
**Algorithm**: Gradient Boosting Regressor (Fine-tuned)
- **Features**: 19 (temporal + cyclical + lags + rolling + statistical)
- **Hyperparameters**: 450 estimators, lr=0.025, max_depth=11
- **Metrics**: MAE, R², Accuracy

## 📈 Model Performance

| Model | Accuracy | Key Metric |
|-------|----------|------------|
| Call Volume | 85-90% | MAPE < 0.15 |
| Event Type | 75-80% | F1-Score > 0.73 |
| Peak Hour | 90-95% | R² > 0.85 |

## 🔧 Configuration

### Environment Variables
Create `.env` file (optional):
```
FIREBASE_PROJECT_ID=your-project-id
STREAMLIT_SERVER_PORT=8501
```

### Streamlit Config
`.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

## 📊 Data Format

Required CSV columns:
- `call_ts`: Timestamp (datetime)
- `category`: Incident type (string)
- `jurisdiction`: Police station/area (string)
- `latitude`: Latitude coordinate (float)
- `longitude`: Longitude coordinate (float)
- `station_sub`: Sub-station (optional)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

This project is developed for Goa Police Hackathon.

## 👥 Team

**Eternal Blue 2**
- Raghav Shrivastav

## 🙏 Acknowledgments

- Goa Police Department
- Streamlit Community
- scikit-learn & XGBoost teams

## 📞 Support

For issues and questions, please open an issue on GitHub.

---

**Built with ❤️ for Goa Police 112 Emergency Services**
