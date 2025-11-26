import pandas as pd
import numpy as np
from pathlib import Path

# =========================
# 0) Configuração
# =========================
DATA_DIR = Path(".")
TRAIN_FILE = DATA_DIR / "training_data.csv"
TEST_FILE  = DATA_DIR / "test_data.csv"
OUT_FILE   = DATA_DIR / "submission_hgb_optimizado.csv"

ID_COL = "RowId"
TARGET_TRAIN = "AVERAGE_SPEED_DIFF"   # coluna alvo no train
TARGET_SUB = "Speed_Diff"             # nome da coluna na submissão

ALLOWED_CLASSES = {"None", "Low", "Medium", "High", "Very_High"}

# =========================
# 1) Ler dados (robusto)
# =========================
def read_csv_robust(path):
    for enc in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path, encoding="latin1", engine="python", on_bad_lines="skip")

train = read_csv_robust(TRAIN_FILE)
test  = read_csv_robust(TEST_FILE)

# garantir RowId no test
if ID_COL not in test.columns:
    test[ID_COL] = np.arange(1, len(test) + 1)

# =========================
# 2) Engenharia temporal + features adicionais
# =========================
def add_time_feats(df):
    if "record_date" in df.columns:
        dt = pd.to_datetime(df["record_date"], errors="coerce")
        df["hour"]  = dt.dt.hour
        df["dow"]   = dt.dt.dayofweek
        df["month"] = dt.dt.month
        df["year"]  = dt.dt.year
        df.drop(columns=["record_date"], inplace=True)

def add_additional_feats(df):
    # Só se as colunas existirem
    if "dow" in df.columns:
        df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    else:
        df["is_weekend"] = 0

    if "hour" in df.columns:
        rush = df["hour"].between(7, 9) | df["hour"].between(17, 19)
        df["rush_hour"] = rush.astype(int)
    else:
        df["rush_hour"] = 0

    # Features cíclicas (se existirem colunas base)
    if "hour" in df.columns:
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    else:
        df["hour_sin"] = 0.0
        df["hour_cos"] = 0.0

    if "dow" in df.columns:
        df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)
    else:
        df["dow_sin"] = 0.0
        df["dow_cos"] = 0.0

    if "month" in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    else:
        df["month_sin"] = 0.0
        df["month_cos"] = 0.0


train_df = train.copy()
test_df  = test.copy()

# aplicar engenharia
add_time_feats(train_df)
add_time_feats(test_df)

add_additional_feats(train_df)
add_additional_feats(test_df)

# =========================
# 3) Target e features
# =========================
# target: garantir que None é uma classe e normalizar strings
y_raw = train_df[TARGET_TRAIN].astype(object)

def normalize_label(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return "None"
    s = str(s).strip()
    s_lower = s.lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "none": "None",
        "low": "Low",
        "medium": "Medium",
        "high": "High",
        "very_high": "Very_High",
        "veryhigh": "Very_High",
        "na": "None",
        "null": "None",
        "nan": "None",
    }
    return mapping.get(s_lower, s)

y = y_raw.apply(normalize_label)

bad = set(pd.Series(y).unique()) - ALLOWED_CLASSES
if bad:
    raise ValueError(f"Valores inesperados na target: {bad}. Esperados: {sorted(ALLOWED_CLASSES)}")

drop_cols = {ID_COL, TARGET_TRAIN}
feature_cols = [c for c in train_df.columns if c in test_df.columns and c not in drop_cols]

X = train_df[feature_cols].copy()
X_test = test_df[feature_cols].copy()

# =========================
# 4) Pré-processamento
# =========================
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import sklearn
from packaging import version

num_cols = [c for c in feature_cols if np.issubdtype(X[c].dtype, np.number)]
cat_cols = [c for c in feature_cols if c not in num_cols]

numeric_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
])

# OneHotEncoder compatível com várias versões
ohe_kwargs = {"handle_unknown": "ignore"}
try:
    OneHotEncoder(min_frequency=5)
    ohe_kwargs["min_frequency"] = 5
except TypeError:
    pass

if version.parse(sklearn.__version__) >= version.parse("1.4"):
    ohe_kwargs["sparse_output"] = False
else:
    ohe_kwargs["sparse"] = False

categorical_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(**ohe_kwargs)),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_tf, num_cols),
        ("cat", categorical_tf, cat_cols),
    ],
    remainder="drop"
)

# =========================
# 5) Modelo HGB (mais regularizado)
# =========================
from sklearn.ensemble import HistGradientBoostingClassifier

def build_hgb(random_state=42):
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=None,
        max_iter=600,
        max_leaf_nodes=25,
        min_samples_leaf=30,
        l2_regularization=1.0,
        validation_fraction=0.1,
        early_stopping=True,
        random_state=random_state,
    )

from sklearn.pipeline import Pipeline as SkPipeline

hgb = build_hgb(random_state=42)

pipe = SkPipeline(steps=[("prep", preprocess), ("hgb", hgb)])

# =========================
# 6) Validação (accuracy + macro-F1 + CV)
# =========================
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_sample_weight

# weights para lidar com desbalanceamento
sample_weights = compute_sample_weight(class_weight="balanced", y=y)

# 6.1 Cross-validation estratificado (macro-F1)
print("\n[CV] StratifiedKFold 5-fold - Macro-F1 (com sample_weight):")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

from sklearn.metrics import f1_score

print("\n[CV] StratifiedKFold 5-fold - Macro-F1 (com sample_weight):")

cv_scores = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
    X_tr_cv = X.iloc[train_idx]
    X_val_cv = X.iloc[val_idx]
    y_tr_cv = y.iloc[train_idx]
    y_val_cv = y.iloc[val_idx]

    w_tr_cv = sample_weights[train_idx]

    # novo modelo para cada fold (para não “vazar” informação entre folds)
    hgb_cv = build_hgb(random_state=42 + fold)
    pipe_cv = SkPipeline(steps=[("prep", preprocess), ("hgb", hgb_cv)])

    pipe_cv.fit(X_tr_cv, y_tr_cv, hgb__sample_weight=w_tr_cv)
    y_val_pred_cv = pipe_cv.predict(X_val_cv)

    f1_cv = f1_score(y_val_cv, y_val_pred_cv, average="macro")
    cv_scores.append(f1_cv)
    print(f"Fold {fold}: Macro-F1 = {f1_cv:.4f}")

cv_scores = np.array(cv_scores)
print("Scores por fold:", np.round(cv_scores, 4))
print("Média Macro-F1 CV:", cv_scores.mean().round(4))

# 6.2 Hold-out para ter matriz de confusão e relatório
X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
    X, y, sample_weights,
    test_size=0.20,
    random_state=42,
    stratify=y
)

pipe.fit(X_tr, y_tr, hgb__sample_weight=w_tr)
y_val_pred = pipe.predict(X_val)

acc = accuracy_score(y_val, y_val_pred)
f1m = f1_score(y_val, y_val_pred, average="macro")

print(f"\n[VALIDAÇÃO - HOLDOUT] Accuracy: {acc:.4f} | Macro-F1: {f1m:.4f}\n")
print("Classification report (val):")
print(classification_report(y_val, y_val_pred, digits=4))

print("Confusion matrix (val):")
labels_sorted = sorted(ALLOWED_CLASSES)
print(pd.DataFrame(
    confusion_matrix(y_val, y_val_pred, labels=labels_sorted),
    index=labels_sorted,
    columns=labels_sorted
))

# =========================
# 7) Treino final + ENSEMBLE para submissão
# =========================
print("\n[TRAIN] Treino final com ensemble de 3 modelos HistGradientBoosting...")

seeds = [42, 1337, 2025]
pipes_ensemble = []
probs_list = []

for seed in seeds:
    hgb_seed = build_hgb(random_state=seed)
    pipe_seed = SkPipeline(steps=[("prep", preprocess), ("hgb", hgb_seed)])
    pipe_seed.fit(X, y, hgb__sample_weight=sample_weights)
    pipes_ensemble.append(pipe_seed)

    probs = pipe_seed.predict_proba(X_test)
    probs_list.append(probs)

# média das probabilidades dos 3 modelos
probs_mean = np.mean(probs_list, axis=0)

# garantir que as classes são consistentes
classes = pipes_ensemble[0].named_steps["hgb"].classes_
for p in pipes_ensemble[1:]:
    assert np.array_equal(classes, p.named_steps["hgb"].classes_), "Inconsistência nas classes entre modelos do ensemble."

pred_indices = np.argmax(probs_mean, axis=1)
pred_test = classes[pred_indices]

submission = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET_SUB: pred_test})

# checks de formato
assert submission.columns.tolist() == [ID_COL, TARGET_SUB], "Cabeçalho inválido."
assert set(submission[TARGET_SUB].unique()).issubset(ALLOWED_CLASSES), "Labels inválidos na submissão."

if len(submission) != 1500:
    print(f"\n[AVISO] Submissão com {len(submission)} linhas (esperado: 1500). Verifica o test_data.csv.")
else:
    print("\nNúmero de linhas da submissão OK (1500).")

# distribuição das classes previstas
print("\nDistribuição das classes previstas no test (ensemble):")
print(submission[TARGET_SUB].value_counts().to_string())

submission.to_csv(OUT_FILE, index=False)
print(f"\n[OK] Submissão escrita em: {OUT_FILE.resolve()}")
print(submission.head())

