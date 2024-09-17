
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def preprocess_data(df):
    drop_columns = [col for col in df.columns if 'id' in col.lower() or 'row' in col.lower()]
    df = df.drop(columns=drop_columns, errors='ignore')
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    
    df.fillna(df.mean(), inplace=True)
    
    numerical_cols = df.select_dtypes(include=['number']).columns
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

def train_model(df):
    X = preprocess_data(df.drop(columns=['churn']))
    y = df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    joblib.dump(model, "churn_model.pkl")

def load_model():
    return joblib.load("churn_model.pkl")

def predict_churn(df):
    model = load_model()
    df_processed = preprocess_data(df)
    return model.predict(df_processed)
