import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(df):
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    categorical_cols = df.select_dtypes(include=[object]).columns
    
    # Fill missing values for numeric columns with their mean
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Fill missing values for categorical columns with the most frequent value
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    
    # Encode categorical columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Feature scaling
    X = df.drop(columns='churn')
    y = df['churn']
    
    return X, y
