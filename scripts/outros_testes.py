# ============================================
# TASK 1 - Análise, Preparação e Modelos
# Dataset: training_data.csv (Kaggle DAA-TG)
# ============================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ==========================
# 1. Carregar os dados
# ==========================

train_path = "training_data.csv"
test_path = "test_data.csv"
sample_path = "example_submission.csv"

# encoding='latin1' por causa dos acentos em português
train = pd.read_csv(train_path, encoding="latin1")
test = pd.read_csv(test_path, encoding="latin1")
sample = pd.read_csv(sample_path, encoding="latin1")

print("Shape train:", train.shape)
print("Shape test:", test.shape)
print("\nColunas train:", train.columns.tolist())
print("Colunas test:", test.columns.tolist())

# ==========================
# 2. Análise Exploratória (EDA) básica
# ==========================

target_col = "AVERAGE_SPEED_DIFF"

print("\n=== Distribuição do target ===")
print(train[target_col].value_counts(dropna=False))

print("\n=== Distribuição relativa do target ===")
print(train[target_col].value_counts(normalize=True, dropna=False))

print("\n=== Valores em falta (train) ===")
print(train.isna().sum())

print("\n=== Valores em falta (test) ===")
print(test.isna().sum())

# Algumas colunas categóricas para ver valores
cat_cols_to_check = ["city_name", "LUMINOSITY", "AVERAGE_CLOUDINESS", "AVERAGE_RAIN"]
for c in cat_cols_to_check:
    if c in train.columns:
        print(f"\n=== Valores mais frequentes em {c} ===")
        print(train[c].value_counts().head(10))

print("\n=== Descrição numérica das variáveis numéricas ===")
print(train.select_dtypes(include=[np.number]).describe())

# ==========================
# 3. Engenharia de Features temporais
# ==========================

# Converter record_date para datetime
train["record_date"] = pd.to_datetime(train["record_date"])
test["record_date"] = pd.to_datetime(test["record_date"])

# Criar features temporais
for df in (train, test):
    df["hour"] = df["record_date"].dt.hour
    df["dayofweek"] = df["record_date"].dt.dayofweek  # 0 = segunda
    df["month"] = df["record_date"].dt.month

# ==========================
# 4. Filtrar apenas linhas com target conhecido
# ==========================

# Há 2200 linhas com AVERAGE_SPEED_DIFF = NaN -> não servem para treino
train_labeled = train[train[target_col].notna()].copy()
print("\nLinhas totais em train:", train.shape[0])
print("Linhas com target conhecido:", train_labeled.shape[0])

# ==========================
# 5. Preparar X e y
# ==========================

# Vamos remover o target e o record_date (já extraímos hour/dayofweek/month)
cols_to_drop = [target_col, "record_date"]

X = train_labeled.drop(columns=cols_to_drop)
y = train_labeled[target_col]

# Identificar colunas categóricas e numéricas
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()

print("\nColunas categóricas:", categorical_features)
print("Colunas numéricas:", numeric_features)

# Pré-processamento:
# - OneHotEncoder para categóricas
# - passthrough para numéricas
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)

# ==========================
# 6. Train/Validation split
# ==========================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTamanho X_train:", X_train.shape)
print("Tamanho X_val:", X_val.shape)

# ==========================
# 7. Função auxiliar para treinar e avaliar modelos
# ==========================

def treinar_e_avaliar(nome, modelo_base):
    """
    Cria um Pipeline com:
      - preprocess (OneHot + numéricas)
      - modelo_base (classificador)
    Treina no X_train, y_train e avalia no X_val, y_val.
    """
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", modelo_base),
        ]
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    print(f"\n==================== {nome} ====================")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_val, y_pred))

    return pipe, acc

# ==========================
# 8. Modelos da Task 1
# ==========================

# 8.1 DummyClassifier (baseline)
dummy_model = DummyClassifier(strategy="most_frequent")
dummy_pipe, dummy_acc = treinar_e_avaliar("Dummy (classe mais frequente)", dummy_model)

# 8.2 Decision Tree
tree_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=10  # limitar profundidade para evitar overfitting extremo
)
tree_pipe, tree_acc = treinar_e_avaliar("Decision Tree (max_depth=10)", tree_model)

# 8.3 Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf_pipe, rf_acc = treinar_e_avaliar("Random Forest (200 árvores)", rf_model)

print("\n=== Resumo de accuracy na validação ===")
print(f"Dummy:        {dummy_acc:.4f}")
print(f"DecisionTree: {tree_acc:.4f}")
print(f"RandomForest: {rf_acc:.4f}")

# Escolher o melhor modelo (aqui assumimos que foi o RandomForest)
best_model = rf_pipe
print("\nModelo escolhido (por agora): RandomForest")

# ==========================
# 9. (Opcional) Treinar o melhor modelo com todos os dados rotulados
# ==========================

best_model.fit(X, y)
print("\nTreino final do melhor modelo concluído com todos os dados rotulados.")

# Se quiserem, depois podes usar o 'best_model' para prever no test_data
# (para a Task 2 / Kaggle), por exemplo:
#
# test_features = test.drop(columns=["record_date"])
# preds_test = best_model.predict(test_features)
# print("\nPredições no test_data:", preds_test[:10])
#
# E depois construir um ficheiro de submissão seguindo o formato do example_submission.csv
