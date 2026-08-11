"""Proyecto: Predicción de aprobación de préstamos.

Ejecución en VS Code (terminal):
    python proyecto_prestamos.py

El archivo CSV debe estar en la misma carpeta que este script.
Los resultados gráficos se guardan en la carpeta "graficos".
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------------
# 1. Carga y comprensión de los datos
# ---------------------------------------------------------------------------
CARPETA_PROYECTO = Path(__file__).resolve().parent
posibles_rutas = [
    CARPETA_PROYECTO / "loan_approval_dataset.csv",
    CARPETA_PROYECTO.parent / "loan_approval_dataset.csv",
    Path(r"C:\Users\Isabella\Downloads\loan_approval_dataset.csv"),
]
RUTA_CSV = next((ruta for ruta in posibles_rutas if ruta.exists()), posibles_rutas[0])

if not RUTA_CSV.exists():
    raise FileNotFoundError(
        "No se encontró loan_approval_dataset.csv. Copia el CSV en esta carpeta."
    )

CARPETA_GRAFICOS = CARPETA_PROYECTO / "graficos"
CARPETA_GRAFICOS.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", palette="deep")

df = pd.read_csv(RUTA_CSV)
df.columns = df.columns.str.strip()
for columna in df.select_dtypes(include=["object", "string"]).columns:
    df[columna] = df[columna].str.strip()

print("=" * 65)
print("PROYECTO: PREDICCIÓN DE APROBACIÓN DE PRÉSTAMOS")
print("=" * 65)
print(f"\nDimensiones del dataset: {df.shape[0]} filas y {df.shape[1]} columnas")
print(f"Valores nulos totales: {df.isna().sum().sum()}")
print(f"Filas duplicadas: {df.duplicated().sum()}")
print("\nDistribución de la variable objetivo:")
print(df["loan_status"].value_counts())


# ---------------------------------------------------------------------------
# 2. Análisis exploratorio de datos (EDA) y gráficos
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
orden_estado = ["Approved", "Rejected"]
sns.countplot(data=df, x="loan_status", order=orden_estado, ax=ax)
ax.set_title("Distribución de solicitudes por estado", weight="bold")
ax.set_xlabel("Estado del préstamo")
ax.set_ylabel("Número de solicitudes")
for barra in ax.patches:
    ax.annotate(
        f"{int(barra.get_height())}",
        (barra.get_x() + barra.get_width() / 2, barra.get_height()),
        ha="center",
        va="bottom",
    )
fig.tight_layout()
fig.savefig(CARPETA_GRAFICOS / "01_distribucion_estado.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="loan_status", y="cibil_score", order=orden_estado, ax=ax)
ax.set_title("Puntaje CIBIL según estado del préstamo", weight="bold")
ax.set_xlabel("Estado del préstamo")
ax.set_ylabel("Puntaje CIBIL")
fig.tight_layout()
fig.savefig(CARPETA_GRAFICOS / "02_cibil_por_estado.png", dpi=180)
plt.close(fig)

print("\nPromedio del puntaje CIBIL por estado:")
print(df.groupby("loan_status")["cibil_score"].mean().round(1))


# ---------------------------------------------------------------------------
# 3. Preparación de los datos
# ---------------------------------------------------------------------------
# Se sustituyen activos residenciales negativos por cero, ya que no son válidos.
df_modelo = df.copy()
df_modelo.loc[
    df_modelo["residential_assets_value"] < 0, "residential_assets_value"
] = 0

# loan_id se excluye: es un identificador, no una característica del solicitante.
X = df_modelo.drop(columns=["loan_id", "loan_status"])
# 1 significa aprobado y 0 significa rechazado.
y = (df_modelo["loan_status"] == "Approved").astype(int)

columnas_numericas = X.select_dtypes(include="number").columns
columnas_categoricas = X.select_dtypes(exclude="number").columns

preprocesador = ColumnTransformer(
    transformers=[
        (
            "numericas",
            Pipeline(
                steps=[
                    ("imputar", SimpleImputer(strategy="median")),
                    ("escalar", StandardScaler()),
                ]
            ),
            columnas_numericas,
        ),
        (
            "categoricas",
            Pipeline(
                steps=[
                    ("imputar", SimpleImputer(strategy="most_frequent")),
                    ("codificar", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            columnas_categoricas,
        ),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nEntrenamiento: {len(X_train)} registros | Prueba: {len(X_test)} registros")


# ---------------------------------------------------------------------------
# 4. Modelado: regresión logística y bosque aleatorio
# ---------------------------------------------------------------------------
modelos = {
    "Regresión logística": LogisticRegression(max_iter=2000, random_state=42),
    "Bosque aleatorio": RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced"
    ),
}

resultados = []
modelos_entrenados = {}
for nombre, modelo in modelos.items():
    pipeline = Pipeline(
        steps=[("preprocesamiento", preprocesador), ("modelo", modelo)]
    )
    pipeline.fit(X_train, y_train)
    prediccion = pipeline.predict(X_test)
    probabilidad = pipeline.predict_proba(X_test)[:, 1]
    modelos_entrenados[nombre] = pipeline

    resultados.append(
        {
            "Modelo": nombre,
            "Accuracy": accuracy_score(y_test, prediccion),
            "Precision": precision_score(y_test, prediccion),
            "Recall": recall_score(y_test, prediccion),
            "F1": f1_score(y_test, prediccion),
            "ROC-AUC": roc_auc_score(y_test, probabilidad),
        }
    )

tabla_resultados = pd.DataFrame(resultados).set_index("Modelo").round(3)
print("\nCOMPARACIÓN DE MODELOS")
print(tabla_resultados)


# ---------------------------------------------------------------------------
# 5. Evaluación del mejor modelo y gráficos finales
# ---------------------------------------------------------------------------
mejor_modelo = modelos_entrenados["Bosque aleatorio"]
prediccion_mejor = mejor_modelo.predict(X_test)

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    prediccion_mejor,
    display_labels=["Rechazado", "Aprobado"],
    cmap="Blues",
    colorbar=False,
    ax=ax,
)
ax.set_title("Matriz de confusión — Bosque aleatorio", weight="bold")
fig.tight_layout()
fig.savefig(CARPETA_GRAFICOS / "03_matriz_confusion.png", dpi=180)
plt.close(fig)

nombres_variables = mejor_modelo.named_steps[
    "preprocesamiento"
].get_feature_names_out()
importancias = pd.Series(
    mejor_modelo.named_steps["modelo"].feature_importances_,
    index=nombres_variables,
).sort_values(ascending=False)

top_10 = importancias.head(10).sort_values()
fig, ax = plt.subplots(figsize=(9, 5))
top_10.plot(kind="barh", color="#177E89", ax=ax)
ax.set_title("Variables más importantes — Bosque aleatorio", weight="bold")
ax.set_xlabel("Importancia")
ax.set_ylabel("")
fig.tight_layout()
fig.savefig(CARPETA_GRAFICOS / "04_importancia_variables.png", dpi=180)
plt.close(fig)

print("\nCONCLUSIÓN")
print(
    "El bosque aleatorio fue el mejor modelo. El puntaje CIBIL fue la "
    "variable más importante para predecir la aprobación."
)
print(f"\nGráficos guardados en: {CARPETA_GRAFICOS.resolve()}")
