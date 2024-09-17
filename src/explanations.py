import shap
import lime.lime_tabular
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def get_global_shap_explanation(model, X):
    """ Generate global SHAP explanations """
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    return shap.summary_plot(shap_values, X, feature_names=X.columns)

def get_local_shap_explanation(model, X, instance_idx):
    """ Generate local SHAP explanation for a single instance """
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    return shap.force_plot(shap_values[instance_idx])

def get_lime_explanation(model, X, instance_idx):
    """ Generate LIME explanation for a single instance """
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X.values,
        feature_names=X.columns,
        mode='classification'
    )
    explanation = explainer.explain_instance(X.iloc[instance_idx].values, model.predict_proba)
    return explanation.as_list()

def train_surrogate_model(X, y):
    """ Train a surrogate decision tree model for explainability """
    surrogate_model = DecisionTreeClassifier()
    surrogate_model.fit(X, y)
    return surrogate_model

def get_surrogate_explanation(surrogate_model, X, instance_idx):
    """ Generate explanation using a surrogate decision tree model """
    tree_explanation = surrogate_model.predict([X.iloc[instance_idx].values])
    return tree_explanation
