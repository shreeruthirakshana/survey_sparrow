import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import shap
from src.data_utils import preprocess_data

class ChurnModel:
    def __init__(self):
        self.model = RandomForestClassifier()
        self.scaler = StandardScaler()
        self.label_encoders = {}
    
    def train(self, df):
        # Preprocess data
        X, y = preprocess_data(df)
        self.scaler.fit(X)
        
        # Fit model
        self.model.fit(X, y)
        joblib.dump(self.model, 'model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.label_encoders, 'label_encoders.pkl')

    def load_model(self):
        self.model = joblib.load('model.pkl')
        self.scaler = joblib.load('scaler.pkl')
        self.label_encoders = joblib.load('label_encoders.pkl')

    def predict(self, input_data):
        # Preprocess input data
        df = pd.DataFrame([input_data])
        X_scaled, _ = preprocess_data(df)
        X_scaled = self.scaler.transform(X_scaled)
        return self.model.predict(X_scaled).tolist()

    def explain(self, input_data, explanation_type='global'):
        # Preprocess input data
        df = pd.DataFrame([input_data])
        X_scaled, _ = preprocess_data(df)
        X_scaled = self.scaler.transform(X_scaled)
        
        explainer = shap.Explainer(self.model)
        shap_values = explainer(X_scaled)
        
        if explanation_type == 'global':
            return shap.summary_plot(shap_values, X_scaled, feature_names=df.columns)
        elif explanation_type == 'local':
            return shap.force_plot(explainer.expected_value, shap_values[0], df.columns)
