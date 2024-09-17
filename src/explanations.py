
import shap
import lime
import lime.lime_tabular
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.model import load_model, preprocess_data

def global_explanation(df):
    model = load_model()
    df_processed = preprocess_data(df)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df_processed)
    return shap.summary_plot(shap_values, df_processed)

def local_explanation(df, instance_idx):
    model = load_model()
    df_processed = preprocess_data(df)
    
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=df_processed.values,
        feature_names=df_processed.columns,
        class_names=['0', '1'],
        mode='classification'
    )
    
    explanation = explainer.explain_instance(
        df_processed.iloc[instance_idx].values,
        model.predict_proba
    )
    
    return explanation.as_list()

def surrogate_model_explanation(df):
    model = load_model()
    df_processed = preprocess_data(df)
    
    from sklearn.tree import DecisionTreeClassifier
    surrogate = DecisionTreeClassifier()
    surrogate.fit(df_processed, model.predict(df_processed))
    
    return surrogate
