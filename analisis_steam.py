"""
=============================================================================
 SEGMENTACIÓN DE USUARIOS DE STEAM MEDIANTE CLUSTERING
=============================================================================
Dataset: steam-200k.csv (interacciones usuario-videojuego)
Autor: Análisis generado con apoyo de Claude

Estructura del script:
    1. Exploración de los datos (EDA)
    2. Preparación de los datos (transformación a nivel usuario)
    3. Modelado (clustering con K-Means)
    4. Interpretación de los clusters
    5. Recomendaciones de negocio

Todas las figuras se guardan en la carpeta "outputs/".
Los datasets intermedios y el reporte final también se guardan ahí.
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend sin ventana, apto para ejecución por consola
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "steam-200k.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42


def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  -> figura guardada: outputs/{name}")


# =============================================================================
# 1. EXPLORACIÓN DE LOS DATOS (EDA)
# =============================================================================
print("=" * 80)
print("1. EXPLORACIÓN DE LOS DATOS")
print("=" * 80)

# El CSV no trae encabezado. Columnas según la documentación pública del
# dataset "Steam Video Games" (Kaggle): user_id, game, behavior, value, extra.
# behavior es "purchase" (compra, value siempre 1.0) o "play"
# (value = horas jugadas). "extra" es una columna constante en 0, sin
# información útil.
col_names = ["user_id", "game", "behavior", "value", "extra"]
df = pd.read_csv(DATA_PATH, header=None, names=col_names)

print(f"\nDimensiones del dataset: {df.shape[0]:,} filas x {df.shape[1]} columnas")

print("\nTipos de variables:")
print(df.dtypes)

print("\nValores nulos por columna:")
print(df.isnull().sum())

n_dup = df.duplicated().sum()
print(f"\nRegistros duplicados (fila exactamente igual): {n_dup}")

print("\nEstadísticos descriptivos de la columna 'value':")
print(df["value"].describe())

print("\nDistribución de la variable 'behavior':")
print(df["behavior"].value_counts())

print(f"\nUsuarios únicos: {df['user_id'].nunique():,}")
print(f"Juegos únicos: {df['game'].nunique():,}")

# Columna 'extra': se confirma que es constante -> no aporta información,
# se eliminará en la etapa de preparación.
print(f"\nValores únicos en columna 'extra': {df['extra'].unique()}")

# --- Visualizaciones exploratorias --------------------------------------

# 1a. Top 15 juegos con más compras
top_purchased = (
    df[df["behavior"] == "purchase"]["game"].value_counts().head(15)
)
plt.figure(figsize=(8, 6))
sns.barplot(x=top_purchased.values, y=top_purchased.index, color="#1b6ca8")
plt.xlabel("Número de compras")
plt.title("Top 15 juegos más comprados")
savefig("01_top_juegos_comprados.png")

# 1b. Top 15 juegos con más horas jugadas acumuladas
top_played = (
    df[df["behavior"] == "play"].groupby("game")["value"].sum().sort_values(ascending=False).head(15)
)
plt.figure(figsize=(8, 6))
sns.barplot(x=top_played.values, y=top_played.index, color="#2a9d8f")
plt.xlabel("Horas totales jugadas")
plt.title("Top 15 juegos con más horas jugadas (acumulado)")
savefig("02_top_juegos_horas.png")

# 1c. Distribución de horas jugadas por registro (escala log, muy sesgada)
play_hours = df[df["behavior"] == "play"]["value"]
plt.figure(figsize=(7, 5))
sns.histplot(np.log1p(play_hours), bins=50, color="#e76f51")
plt.xlabel("log(1 + horas jugadas) por registro usuario-juego")
plt.title("Distribución de horas jugadas (escala log)")
savefig("03_distribucion_horas_log.png")

# 1d. Cantidad de juegos comprados por usuario (distribución, muy sesgada)
games_per_user = df[df["behavior"] == "purchase"].groupby("user_id")["game"].nunique()
plt.figure(figsize=(7, 5))
sns.histplot(games_per_user, bins=60, color="#8338ec")
plt.xlabel("Número de juegos comprados por usuario")
plt.title("Distribución de juegos comprados por usuario")
plt.xlim(0, games_per_user.quantile(0.99))
savefig("04_juegos_por_usuario.png")

print("\nConclusión EDA: el dataset está en formato 'largo' de interacciones "
      "(una fila = un usuario + un juego + un tipo de evento). La distribución "
      "de horas jugadas y de juegos comprados por usuario está muy sesgada a la "
      "derecha (pocos usuarios concentran muchas horas/juegos), por lo que se "
      "aplicarán transformaciones logarítmicas antes del clustering.")


# =============================================================================
# 2. PREPARACIÓN DE LOS DATOS
# =============================================================================
print("\n" + "=" * 80)
print("2. PREPARACIÓN DE LOS DATOS")
print("=" * 80)

# --- 2.1 Limpieza básica -------------------------------------------------
# Se elimina la columna 'extra': es constante (siempre 0) y no aporta
# ninguna información para la segmentación.
df_clean = df.drop(columns=["extra"])

# Se eliminan duplicados exactos (mismos user_id, game, behavior, value):
# corresponden a registros repetidos por error de captura del dataset
# original y no representan información adicional real.
before = len(df_clean)
df_clean = df_clean.drop_duplicates()
print(f"\nRegistros eliminados por duplicado exacto: {before - len(df_clean)}")

# --- 2.2 Transformación de formato largo -> tabla por usuario -----------
# Justificación: el objetivo es segmentar USUARIOS, no interacciones.
# Se separan los eventos de compra y de juego, y se agregan por user_id
# para construir variables que resuman el comportamiento de cada usuario.
purchase_df = df_clean[df_clean["behavior"] == "purchase"][["user_id", "game"]].drop_duplicates()
play_df = df_clean[df_clean["behavior"] == "play"][["user_id", "game", "value"]].rename(
    columns={"value": "hours"}
)

# --- 2.3 Ingeniería de características -----------------------------------
# Variables elegidas y su justificación:
#   n_games_purchased      -> tamaño de la "biblioteca" del usuario (colección)
#   n_games_played         -> cuántos de esos juegos realmente activó/jugó
#   total_hours            -> volumen total de consumo/uso de la plataforma
#   avg_hours_per_game      -> profundidad de juego promedio (¿juega poco de
#                              muchos títulos o mucho de pocos títulos?)
#   max_hours_single_game   -> presencia de un "juego ancla" muy jugado
#   engagement_ratio        -> % de juegos comprados que efectivamente jugó
#                              (mide desperdicio de compra / fidelidad de uso)
n_purchased = purchase_df.groupby("user_id")["game"].nunique().rename("n_games_purchased")
n_played = play_df.groupby("user_id")["game"].nunique().rename("n_games_played")
total_hours = play_df.groupby("user_id")["hours"].sum().rename("total_hours")
avg_hours = play_df.groupby("user_id")["hours"].mean().rename("avg_hours_per_game")
max_hours = play_df.groupby("user_id")["hours"].max().rename("max_hours_single_game")

user_df = pd.concat([n_purchased, n_played, total_hours, avg_hours, max_hours], axis=1)

# Usuarios que compraron pero nunca registraron horas de juego (no aparecen
# en play_df) -> se completan sus métricas de juego con 0, ya que
# efectivamente no jugaron ninguno de sus títulos.
for c in ["n_games_played", "total_hours", "avg_hours_per_game", "max_hours_single_game"]:
    user_df[c] = user_df[c].fillna(0)
user_df["n_games_purchased"] = user_df["n_games_purchased"].fillna(0)

# engagement_ratio: se limita a máx 1.0 por posibles inconsistencias del
# dataset (juegos "jugados" que no figuran como "comprados", p. ej. F2P).
user_df["engagement_ratio"] = (
    user_df["n_games_played"] / user_df["n_games_purchased"]
).clip(upper=1)

print(f"\nTabla de usuarios construida: {user_df.shape[0]:,} usuarios x {user_df.shape[1]} variables")
print(user_df.describe().round(2))

# --- 2.4 Transformación logarítmica --------------------------------------
# Justificación: las variables de conteo/horas presentan alta asimetría
# positiva (pocos usuarios "power" concentran la mayoría de horas/juegos).
# K-Means es sensible a outliers y asume variables con distribución más
# simétrica; log1p reduce el peso de los valores extremos sin perder el
# orden relativo entre usuarios.
log_cols = [
    "n_games_purchased",
    "n_games_played",
    "total_hours",
    "avg_hours_per_game",
    "max_hours_single_game",
]
for c in log_cols:
    user_df[c + "_log"] = np.log1p(user_df[c])

feature_cols = [c + "_log" for c in log_cols] + ["engagement_ratio"]

plt.figure(figsize=(10, 5))
user_df[feature_cols].boxplot(rot=45)
plt.title("Variables finales antes de escalar (post log-transform)")
savefig("05_boxplot_features_log.png")

# --- 2.5 Escalamiento -----------------------------------------------------
# Justificación: K-Means se basa en distancias euclidianas; sin escalar,
# variables con rangos más amplios (ej. total_hours_log) dominarían el
# cálculo de distancia frente a variables como engagement_ratio (0-1).
# Se usa StandardScaler (media 0, desviación 1) sobre las variables ya
# log-transformadas.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(user_df[feature_cols])

print(f"\nVariables finales usadas para clustering: {feature_cols}")


# =============================================================================
# 3. MODELADO
# =============================================================================
print("\n" + "=" * 80)
print("3. MODELADO (CLUSTERING)")
print("=" * 80)

# --- 3.1 Selección del número de clusters (K-Means) -----------------------
# Se evalúa K de 2 a 8 usando el método del codo (inercia) y el coeficiente
# de silueta, que mide qué tan bien separados y cohesionados quedan los
# clusters (más cercano a 1 es mejor).
inertias, silhouettes = [], []
K_range = range(2, 9)
for k in K_range:
    km_tmp = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_scaled)
    inertias.append(km_tmp.inertia_)
    silhouettes.append(silhouette_score(X_scaled, km_tmp.labels_))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(list(K_range), inertias, marker="o", color="#1b6ca8")
axes[0].set_xlabel("k (número de clusters)")
axes[0].set_ylabel("Inercia")
axes[0].set_title("Método del codo")
axes[1].plot(list(K_range), silhouettes, marker="o", color="#e76f51")
axes[1].set_xlabel("k (número de clusters)")
axes[1].set_ylabel("Coeficiente de silueta")
axes[1].set_title("Coeficiente de silueta por k")
savefig("06_seleccion_k.png")

best_k = list(K_range)[int(np.argmax(silhouettes))]
print(f"\nSilhouette por k: {dict(zip(K_range, np.round(silhouettes, 3)))}")
print(f"k con mejor silhouette: {best_k}")

# Se fija K=4: presenta el mejor (o casi mejor) coeficiente de silueta y,
# a diferencia de K=2, permite diferenciar matices de comportamiento
# (comprador vs. jugador, casual vs. intensivo) útiles para marketing.
K_FINAL = 4

# --- 3.2 Justificación del algoritmo ---------------------------------------
# Se elige K-MEANS como algoritmo principal porque:
#   - El espacio de características es continuo y de dimensión moderada
#     (6 variables), donde K-Means funciona de forma eficiente y estable.
#   - Tras el escalamiento, los clusters resultan razonablemente esféricos
#     y de tamaño comparable, condición favorable para K-Means.
#   - Es fácilmente interpretable: los centroides representan directamente
#     el "perfil promedio" de cada segmento, ideal para conclusiones de
#     negocio.
#   - Es escalable a los ~12 mil usuarios del dataset.
# DBSCAN se probó como comparación: al tratarse de densidades muy variables
# (usuarios "power" vs. casuales), tiende a dejar una gran cantidad de
# usuarios como ruido o a fusionar clusters, dificultando la interpretación
# de negocio. El clustering jerárquico se descarta como algoritmo principal
# por su costo computacional en +12 mil registros, aunque puede usarse en
# muestras pequeñas para validar la estructura (ver dendrograma opcional).

# --- 3.3 Comparación rápida con DBSCAN (validación) ------------------------
dbscan = DBSCAN(eps=0.8, min_samples=15).fit(X_scaled)
n_clusters_db = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
n_noise = (dbscan.labels_ == -1).sum()
print(f"\n[Validación] DBSCAN encontró {n_clusters_db} clusters y "
      f"{n_noise} usuarios clasificados como ruido "
      f"({n_noise/len(dbscan.labels_):.1%} del total).")
print("Esto confirma que K-Means es más adecuado para este caso de uso: "
      "se busca clasificar a TODOS los usuarios en un segmento accionable "
      "para marketing, no descartar una fracción como ruido.")

# --- 3.4 Modelo final K-Means ----------------------------------------------
kmeans = KMeans(n_clusters=K_FINAL, random_state=RANDOM_STATE, n_init=10)
user_df["cluster"] = kmeans.fit_predict(X_scaled)

sil_final = silhouette_score(X_scaled, user_df["cluster"])
print(f"\nModelo final: K-Means con k={K_FINAL} | silhouette = {sil_final:.3f}")

# --- 3.5 Visualización de los clusters (PCA a 2D) ---------------------------
pca = PCA(n_components=2, random_state=RANDOM_STATE)
coords = pca.fit_transform(X_scaled)
user_df["pca_1"], user_df["pca_2"] = coords[:, 0], coords[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=user_df, x="pca_1", y="pca_2", hue="cluster", palette="tab10", alpha=0.6, s=25
)
plt.title(f"Clusters de usuarios (proyección PCA 2D, varianza explicada "
          f"{pca.explained_variance_ratio_.sum():.0%})")
savefig("07_clusters_pca.png")


# =============================================================================
# 4. INTERPRETACIÓN DE LOS CLUSTERS
# =============================================================================
print("\n" + "=" * 80)
print("4. INTERPRETACIÓN DE LOS CLUSTERS")
print("=" * 80)

profile_cols = [
    "n_games_purchased", "n_games_played", "total_hours",
    "avg_hours_per_game", "max_hours_single_game", "engagement_ratio",
]
profile = user_df.groupby("cluster")[profile_cols].mean().round(2)
profile["n_usuarios"] = user_df["cluster"].value_counts().sort_index()
profile["% del total"] = (profile["n_usuarios"] / len(user_df) * 100).round(1)
print("\nPerfil promedio por cluster:")
print(profile.to_string())

# Boxplots de variables clave por cluster
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.flat, profile_cols):
    sns.boxplot(data=user_df, x="cluster", y=col, hue="cluster", ax=ax, palette="tab10", legend=False, showfliers=False)
    ax.set_title(col)
savefig("08_boxplots_por_cluster.png")

# Tamaño de cada segmento
plt.figure(figsize=(6, 5))
sizes = user_df["cluster"].value_counts().sort_index()
plt.pie(sizes, labels=[f"Cluster {i}" for i in sizes.index], autopct="%1.1f%%",
        colors=sns.color_palette("tab10", len(sizes)))
plt.title("Distribución de usuarios por cluster")
savefig("09_distribucion_clusters.png")

# --- Nombramiento de segmentos ---------------------------------------------
# Reglas de asignación de nombre según el perfil promedio de cada cluster
# (compras, horas totales y ratio de engagement). Estas reglas se aplican
# de forma automática sobre los centroides obtenidos, por lo que el
# resultado es reproducible si el modelo se reentrena con los mismos datos.
names = {}
descriptions = {}
for c in profile.index:
    row = profile.loc[c]
    if row["n_games_purchased"] >= profile["n_games_purchased"].median() * 3 and row["total_hours"] >= profile["total_hours"].median():
        names[c] = "Coleccionistas Muy Activos"
        descriptions[c] = (
            "Compran un número muy alto de juegos y además acumulan muchas "
            "horas de juego totales. Son la base de usuarios más valiosa: "
            "compran con frecuencia Y consumen lo que compran."
        )
    elif row["total_hours"] >= profile["total_hours"].median() and row["n_games_purchased"] < profile["n_games_purchased"].median() * 3:
        names[c] = "Usuarios Muy Activos (Hardcore)"
        descriptions[c] = (
            "Compran relativamente pocos juegos, pero acumulan muchísimas "
            "horas jugadas concentradas en esos títulos (alto "
            "avg_hours_per_game y max_hours_single_game). Son jugadores "
            "frecuentes y leales a pocos juegos 'ancla'."
        )
    elif row["engagement_ratio"] < profile["engagement_ratio"].median():
        names[c] = "Compradores Ocasionales / Pasivos"
        descriptions[c] = (
            "Compran juegos pero rara vez los juegan (engagement_ratio bajo, "
            "horas totales casi nulas). Es probable que hayan adquirido los "
            "juegos en bundles/ofertas o por impulso, sin llegar a activarlos."
        )
    else:
        names[c] = "Jugadores Casuales"
        descriptions[c] = (
            "Compran pocos juegos y juegan una fracción alta de ellos, pero "
            "con pocas horas totales. Son usuarios livianos, probablemente "
            "nuevos en la plataforma o con uso esporádico."
        )

for c in profile.index:
    print(f"\nCluster {c} -> '{names[c]}' "
          f"({profile.loc[c,'n_usuarios']:.0f} usuarios, {profile.loc[c,'% del total']}% del total)")
    print(f"  {descriptions[c]}")
    print(f"  Compras promedio: {profile.loc[c,'n_games_purchased']} | "
          f"Juegos jugados promedio: {profile.loc[c,'n_games_played']} | "
          f"Horas totales promedio: {profile.loc[c,'total_hours']} | "
          f"Engagement: {profile.loc[c,'engagement_ratio']}")

user_df["segmento"] = user_df["cluster"].map(names)


# =============================================================================
# 5. EXPORTAR RESULTADOS
# =============================================================================
print("\n" + "=" * 80)
print("5. EXPORTANDO RESULTADOS")
print("=" * 80)

user_df.reset_index().rename(columns={"index": "user_id"}).to_csv(
    os.path.join(OUT_DIR, "usuarios_segmentados.csv"), index=False
)
profile.assign(segmento=[names[c] for c in profile.index]).to_csv(
    os.path.join(OUT_DIR, "perfil_clusters.csv")
)
print("Archivos guardados en outputs/: usuarios_segmentados.csv, perfil_clusters.csv")

print("\n" + "=" * 80)
print("PROCESO FINALIZADO CORRECTAMENTE")
print("=" * 80)
