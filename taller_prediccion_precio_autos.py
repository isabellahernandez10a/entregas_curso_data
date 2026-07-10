"""
================================================================================
TALLER: PREDICCION DEL PRECIO DE VEHICULOS
================================================================================

Objetivo del negocio:
    Estimar el precio de un vehiculo a partir de sus caracteristicas tecnicas
    y comerciales (motor, dimensiones, tipo de combustible, marca, etc.), con
    el fin de apoyar decisiones de fijacion de precios, valoracion de
    inventario o analisis de mercado para una empresa del sector automotor.

Dataset:
    CarPrice_Assignment.csv
    Fuente original: Kaggle - "Car Price Prediction Multiple Linear Regression"
    https://www.kaggle.com/datasets/hellbuoy/car-price-prediction

Estructura del script (sigue la guia del taller):
    Parte 1 - Comprension del problema
    Parte 2 - Exploracion de datos (EDA)
    Parte 3 - Preparacion de datos
    Parte 4 - Modelado
    Parte 5 - Evaluacion del modelo
    Parte 6 - Interpretacion de resultados

Cada parte esta claramente delimitada y comentada para facilitar la lectura
y la sustentacion del taller.
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend sin interfaz grafica, para guardar imagenes a disco
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Carpeta donde se guardaran las graficas generadas por el script
OUTPUT_DIR = "graficas"
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Semilla fija para que los resultados sean reproducibles
RANDOM_STATE = 42


# ==============================================================================
# PARTE 1 - COMPRENSION DEL PROBLEMA
# ==============================================================================
#
# 1. Variable objetivo (target): "price"  (precio del vehiculo, en dolares)
#
# 2. Tipo de problema: REGRESION.
#    Es un problema de regresion porque la variable objetivo ("price") es
#    numerica y continua (puede tomar cualquier valor decimal dentro de un
#    rango), no una categoria o clase discreta. El modelo debe predecir un
#    numero (por ejemplo 13495.0), no una etiqueta como "barato"/"caro".
#
# 3. Contexto del problema:
#    El dataset contiene 205 registros de vehiculos con informacion tecnica
#    (motor, dimensiones, consumo de combustible, tipo de traccion, etc.) y
#    comercial (marca/modelo). El objetivo del negocio es construir un modelo
#    que, a partir de estas caracteristicas, sea capaz de estimar el precio
#    de venta de un vehiculo. Esto es util, por ejemplo, para una compania
#    que quiere fijar precios competitivos de nuevos modelos, para plataformas
#    de compra/venta de vehiculos usados que necesitan tasar automoviles, o
#    para estudios de mercado que buscan entender que caracteristicas
#    impactan mas el precio final.
#
# ==============================================================================


def cargar_datos(ruta_csv: str) -> pd.DataFrame:
    """
    Carga el dataset de precios de vehiculos desde un archivo CSV.

    Parameters
    ----------
    ruta_csv : str
        Ruta al archivo CSV con los datos.

    Returns
    -------
    pd.DataFrame
        DataFrame con los datos cargados.
    """
    df = pd.read_csv(ruta_csv)
    return df


# ==============================================================================
# PARTE 2 - EXPLORACION DE DATOS (EDA)
# ==============================================================================

def exploracion_datos(df: pd.DataFrame):
    """
    Realiza una exploracion basica del dataset:
        - Dimension del dataset
        - Valores nulos por columna
        - Identificacion de variables numericas y categoricas
        - Estadisticas descriptivas de la variable objetivo
        - Generacion de graficos de apoyo (distribucion y correlacion)

    Parameters
    ----------
    df : pd.DataFrame
        Dataset original.
    """
    print("=" * 80)
    print("PARTE 2 - EXPLORACION DE DATOS (EDA)")
    print("=" * 80)

    # --- Dimension del dataset --------------------------------------------
    print(f"\nDimension del dataset: {df.shape[0]} filas x {df.shape[1]} columnas")

    # --- Valores nulos -------------------------------------------------------
    nulos = df.isnull().sum()
    nulos = nulos[nulos > 0]
    print("\nValores nulos por columna:")
    if nulos.empty:
        print("  No se encontraron valores nulos en el dataset.")
    else:
        print(nulos)

    # --- Identificacion de variables numericas y categoricas -----------------
    # Se excluye "car_ID" porque es un identificador, no una caracteristica
    # predictiva real, y "price" porque es la variable objetivo.
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    columnas_numericas = [c for c in columnas_numericas if c not in ["car_ID", "price"]]

    columnas_categoricas = df.select_dtypes(include=["object", "str"]).columns.tolist()

    print(f"\nVariables numericas ({len(columnas_numericas)}):")
    print(columnas_numericas)

    print(f"\nVariables categoricas ({len(columnas_categoricas)}):")
    print(columnas_categoricas)

    # --- Variable objetivo -----------------------------------------------
    print("\nVariable objetivo: 'price'")
    print(df["price"].describe())

    # --- Grafico 1: Distribucion de la variable objetivo (price) -----------
    plt.figure(figsize=(8, 5))
    sns.histplot(df["price"], kde=True, color="steelblue")
    plt.title("Distribucion del precio de los vehiculos")
    plt.xlabel("Precio (USD)")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_distribucion_precio.png", dpi=120)
    plt.close()

    # --- Grafico 2: Matriz de correlacion entre variables numericas --------
    plt.figure(figsize=(12, 9))
    matriz_corr = df[columnas_numericas + ["price"]].corr()
    sns.heatmap(matriz_corr, annot=False, cmap="coolwarm", center=0)
    plt.title("Matriz de correlacion entre variables numericas y el precio")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_matriz_correlacion.png", dpi=120)
    plt.close()

    # --- Grafico 3 (extra): Relacion entre tamano de motor y precio --------
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="enginesize", y="price", hue="fueltype")
    plt.title("Relacion entre tamano del motor y precio")
    plt.xlabel("Tamano del motor (enginesize)")
    plt.ylabel("Precio (USD)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_enginesize_vs_price.png", dpi=120)
    plt.close()

    # --- Grafico 4 (extra): Precio promedio por tipo de carroceria ---------
    plt.figure(figsize=(8, 5))
    precio_por_carroceria = df.groupby("carbody")["price"].mean().sort_values()
    sns.barplot(x=precio_por_carroceria.values, y=precio_por_carroceria.index, color="darkorange")
    plt.title("Precio promedio por tipo de carroceria (carbody)")
    plt.xlabel("Precio promedio (USD)")
    plt.ylabel("Tipo de carroceria")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_precio_por_carroceria.png", dpi=120)
    plt.close()

    print(f"\nSe generaron 4 graficos en la carpeta '{OUTPUT_DIR}/':")
    print("  01_distribucion_precio.png")
    print("  02_matriz_correlacion.png")
    print("  03_enginesize_vs_price.png")
    print("  04_precio_por_carroceria.png")

    return columnas_numericas, columnas_categoricas


# ==============================================================================
# PARTE 3 - PREPARACION DE DATOS
# ==============================================================================

def preparar_datos(df: pd.DataFrame, columnas_numericas: list, columnas_categoricas: list):
    """
    Prepara los datos para el modelado:
        - Limpieza y simplificacion de la variable CarName (se extrae la marca)
        - Manejo de valores faltantes (imputacion, por robustez del pipeline)
        - Separacion en X (predictoras) e y (objetivo)
        - Division en conjuntos de entrenamiento y prueba (80/20)
        - Construccion de un preprocesador que codifica variables categoricas
          (One-Hot Encoding) y escala/imputa variables numericas

    Parameters
    ----------
    df : pd.DataFrame
        Dataset original.
    columnas_numericas : list
        Lista de columnas numericas predictoras.
    columnas_categoricas : list
        Lista de columnas categoricas predictoras.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test, preprocesador (ColumnTransformer)
    """
    print("\n" + "=" * 80)
    print("PARTE 3 - PREPARACION DE DATOS")
    print("=" * 80)

    df = df.copy()

    # La columna "CarName" trae marca + modelo (ej: "alfa-romero giulia").
    # Usar el nombre completo generaria demasiadas categorias unicas (147 de
    # 205 filas), lo que produce sobreajuste y una codificacion inmanejable.
    # Por eso se extrae unicamente la MARCA, que es una variable categorica
    # mucho mas util y generalizable para el modelo.
    df["marca"] = df["CarName"].str.split(" ").str[0].str.lower()

    # Correccion de algunos errores de escritura evidentes en la marca,
    # comunes en este dataset (ej. "maxda" -> "mazda", "porcshce" -> "porsche")
    correcciones_marca = {
        "maxda": "mazda",
        "porcshce": "porsche",
        "toyouta": "toyota",
        "vokswagen": "volkswagen",
        "vw": "volkswagen",
    }
    df["marca"] = df["marca"].replace(correcciones_marca)

    # Se reemplaza CarName por la nueva variable "marca" en la lista de
    # columnas categoricas predictoras, y se descarta CarName y car_ID
    # (identificadores sin valor predictivo real)
    columnas_categoricas_final = [c for c in columnas_categoricas if c != "CarName"] + ["marca"]

    # --- Separacion en X (predictoras) e y (objetivo) -----------------------
    columnas_predictoras = columnas_numericas + columnas_categoricas_final
    X = df[columnas_predictoras]
    y = df["price"]

    print(f"\nVariables predictoras (X): {len(columnas_predictoras)} columnas")
    print(f"Variable objetivo (y): 'price'")
    print(f"Numero de registros: {X.shape[0]}")

    # --- Manejo de valores faltantes -----------------------------------------
    # Aunque este dataset no tiene valores nulos, se incluye un imputador
    # como buena practica: en produccion, datos nuevos podrian traer nulos.
    #   - Variables numericas: se imputan con la MEDIANA (robusta a atipicos)
    #   - Variables categoricas: se imputan con la moda (valor mas frecuente)
    transformador_numerico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    transformador_categorico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # Codificacion de variables categoricas mediante One-Hot Encoding
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocesador = ColumnTransformer(transformers=[
        ("num", transformador_numerico, columnas_numericas),
        ("cat", transformador_categorico, columnas_categoricas_final),
    ])

    # --- Division en entrenamiento (80%) y prueba (20%) ----------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    print(f"\nConjunto de entrenamiento: {X_train.shape[0]} registros (80%)")
    print(f"Conjunto de prueba:        {X_test.shape[0]} registros (20%)")

    return X_train, X_test, y_train, y_test, preprocesador


# ==============================================================================
# PARTE 4 - MODELADO
# ==============================================================================

def entrenar_modelos(X_train, y_train, preprocesador):
    """
    Entrena dos modelos de regresion distintos usando un Pipeline que
    encadena el preprocesamiento (imputacion + codificacion) con el modelo:

        1. Regresion Lineal Multiple: modelo simple e interpretable, usado
           como linea base (baseline).
        2. Random Forest Regressor: modelo de ensamble basado en arboles,
           capaz de capturar relaciones no lineales entre las variables.

    Parameters
    ----------
    X_train, y_train : datos de entrenamiento
    preprocesador : ColumnTransformer con la logica de preprocesamiento

    Returns
    -------
    dict
        Diccionario {nombre_modelo: pipeline_entrenado}
    """
    print("\n" + "=" * 80)
    print("PARTE 4 - MODELADO")
    print("=" * 80)

    modelos = {
        "Regresion Lineal": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, max_depth=None
        ),
    }

    pipelines_entrenados = {}

    for nombre, modelo in modelos.items():
        print(f"\nEntrenando modelo: {nombre} ...")
        pipeline = Pipeline(steps=[
            ("preprocesador", preprocesador),
            ("modelo", modelo),
        ])
        pipeline.fit(X_train, y_train)
        pipelines_entrenados[nombre] = pipeline
        print(f"  {nombre} entrenado correctamente.")

    return pipelines_entrenados


# ==============================================================================
# PARTE 5 - EVALUACION DEL MODELO
# ==============================================================================

def evaluar_modelos(pipelines_entrenados: dict, X_test, y_test):
    """
    Evalua cada modelo entrenado sobre el conjunto de prueba usando las
    metricas estandar para problemas de REGRESION:
        - MAE  (Error Absoluto Medio)
        - MSE  (Error Cuadratico Medio)
        - RMSE (Raiz del Error Cuadratico Medio, para facilitar interpretacion)
        - R2   (Coeficiente de determinacion)

    Parameters
    ----------
    pipelines_entrenados : dict
        Diccionario {nombre_modelo: pipeline_entrenado}
    X_test, y_test : datos de prueba

    Returns
    -------
    pd.DataFrame
        Tabla comparativa con las metricas de cada modelo.
    """
    print("\n" + "=" * 80)
    print("PARTE 5 - EVALUACION DEL MODELO")
    print("=" * 80)

    resultados = []

    for nombre, pipeline in pipelines_entrenados.items():
        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        resultados.append({
            "Modelo": nombre,
            "MAE": round(mae, 2),
            "MSE": round(mse, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4),
        })

        print(f"\nModelo: {nombre}")
        print(f"  MAE  (Error Absoluto Medio):        {mae:,.2f}")
        print(f"  MSE  (Error Cuadratico Medio):       {mse:,.2f}")
        print(f"  RMSE (Raiz del Error Cuad. Medio):   {rmse:,.2f}")
        print(f"  R2   (Coef. de determinacion):       {r2:.4f}")

    tabla_resultados = pd.DataFrame(resultados)

    # Grafico comparativo: valores reales vs. predichos para el mejor modelo
    mejor_nombre = tabla_resultados.sort_values("R2", ascending=False).iloc[0]["Modelo"]
    mejor_pipeline = pipelines_entrenados[mejor_nombre]
    y_pred_mejor = mejor_pipeline.predict(X_test)

    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred_mejor, alpha=0.7, color="teal")
    limite_min = min(y_test.min(), y_pred_mejor.min())
    limite_max = max(y_test.max(), y_pred_mejor.max())
    plt.plot([limite_min, limite_max], [limite_min, limite_max], "r--", label="Prediccion perfecta")
    plt.xlabel("Precio real (USD)")
    plt.ylabel("Precio predicho (USD)")
    plt.title(f"Valores reales vs. predichos - {mejor_nombre}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_real_vs_predicho.png", dpi=120)
    plt.close()

    print(f"\nMejor modelo segun R2: {mejor_nombre}")
    print(f"Grafico guardado en: {OUTPUT_DIR}/05_real_vs_predicho.png")

    return tabla_resultados, mejor_nombre, mejor_pipeline


def importancia_variables(mejor_pipeline, mejor_nombre, columnas_numericas, columnas_categoricas_final):
    """
    Extrae y grafica las variables mas influyentes segun el mejor modelo,
    para apoyar la interpretacion de resultados (Parte 6).

    Solo aplica directamente a modelos basados en arboles (Random Forest) o
    a la Regresion Lineal (a traves de sus coeficientes).

    Parameters
    ----------
    mejor_pipeline : Pipeline entrenado (preprocesador + modelo)
    mejor_nombre : str, nombre del mejor modelo
    columnas_numericas, columnas_categoricas_final : listas de columnas usadas
    """
    modelo = mejor_pipeline.named_steps["modelo"]
    preprocesador = mejor_pipeline.named_steps["preprocesador"]

    # Se recuperan los nombres de las columnas ya codificadas (One-Hot)
    nombres_categoricos = preprocesador.named_transformers_["cat"]["onehot"].get_feature_names_out(
        columnas_categoricas_final
    )
    nombres_totales = list(columnas_numericas) + list(nombres_categoricos)

    if hasattr(modelo, "feature_importances_"):
        importancias = modelo.feature_importances_
        etiqueta_y = "Importancia relativa"
    elif hasattr(modelo, "coef_"):
        importancias = np.abs(modelo.coef_)
        etiqueta_y = "Magnitud del coeficiente (valor absoluto)"
    else:
        print("El modelo seleccionado no expone importancia de variables directamente.")
        return

    df_importancia = pd.DataFrame({
        "variable": nombres_totales,
        "importancia": importancias,
    }).sort_values("importancia", ascending=False).head(10)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=df_importancia, x="importancia", y="variable", color="mediumseagreen")
    plt.title(f"Top 10 variables mas influyentes - {mejor_nombre}")
    plt.xlabel(etiqueta_y)
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/06_importancia_variables.png", dpi=120)
    plt.close()

    print("\nTop 10 variables mas influyentes en la prediccion del precio:")
    print(df_importancia.to_string(index=False))
    print(f"\nGrafico guardado en: {OUTPUT_DIR}/06_importancia_variables.png")

    return df_importancia


# ==============================================================================
# PARTE 6 - INTERPRETACION
# ==============================================================================
#
# NOTA: las respuestas concretas dependen de los resultados numericos que
# arroje la ejecucion del script (tabla de metricas y de importancia de
# variables, impresas en consola y guardadas como graficos). A continuacion
# se deja la estructura de analisis; el detalle numerico se completa al
# revisar la salida de "evaluar_modelos" e "importancia_variables".
#
# 1. Que tan bueno es el modelo?
#    Se evalua principalmente con el R2 sobre el conjunto de prueba: un R2
#    cercano a 1 indica que el modelo explica gran parte de la variabilidad
#    del precio; un R2 bajo o negativo indicaria que el modelo generaliza mal.
#    El MAE y el RMSE deben interpretarse en las mismas unidades del precio
#    (dolares): un MAE de, por ejemplo, 1500 significa que en promedio el
#    modelo se equivoca en unos 1500 dolares al estimar el precio.
#
# 2. Que variable parece influir mas en la prediccion?
#    Se determina revisando el grafico "06_importancia_variables.png" y la
#    tabla impresa por la funcion "importancia_variables". En datasets de
#    este tipo, suelen destacar variables relacionadas con el motor
#    (enginesize, curbweight, horsepower) y la marca del vehiculo.
#
# 3. Que se mejoraria con mas tiempo?
#    - Probar tecnicas de seleccion de variables (RFE, VIF) para reducir
#      colinealidad, especialmente relevante para la Regresion Lineal.
#    - Ajustar hiperparametros del Random Forest (GridSearchCV / RandomizedSearchCV).
#    - Probar otros modelos: Gradient Boosting, XGBoost, regularizacion
#      (Ridge/Lasso) para la regresion lineal.
#    - Realizar validacion cruzada (cross-validation) en lugar de una sola
#      division train/test, para obtener metricas mas robustas.
#    - Analizar y tratar posibles valores atipicos (outliers) en el precio.
#    - Enriquecer el dataset con mas variables de mercado (antiguedad,
#      kilometraje, region, demanda, etc.).
#
# ==============================================================================


def main():
    """
    Funcion principal: ejecuta el flujo completo del taller de punta a punta.
    """
    ruta_csv = "CarPrice_Assignment.csv"

    # Parte 1: el contexto y la definicion del problema estan documentados
    # arriba, como comentarios, ya que son analisis conceptual, no codigo.

    # Carga de datos
    df = cargar_datos(ruta_csv)

    # Parte 2: EDA
    columnas_numericas, columnas_categoricas = exploracion_datos(df)

    # Parte 3: preparacion de datos
    X_train, X_test, y_train, y_test, preprocesador = preparar_datos(
        df, columnas_numericas, columnas_categoricas
    )

    # Lista final de columnas categoricas usada en el preprocesamiento
    # (igual a la usada dentro de preparar_datos, se reconstruye aqui para
    # poder recuperar los nombres de las variables tras el One-Hot Encoding)
    columnas_categoricas_final = [c for c in columnas_categoricas if c != "CarName"] + ["marca"]

    # Parte 4: modelado (se entrenan 2 modelos)
    pipelines_entrenados = entrenar_modelos(X_train, y_train, preprocesador)

    # Parte 5: evaluacion de los modelos
    tabla_resultados, mejor_nombre, mejor_pipeline = evaluar_modelos(
        pipelines_entrenados, X_test, y_test
    )

    print("\nTabla comparativa de metricas:")
    print(tabla_resultados.to_string(index=False))

    # Importancia de variables del mejor modelo (apoya la Parte 6)
    importancia_variables(mejor_pipeline, mejor_nombre, columnas_numericas, columnas_categoricas_final)

    print("\n" + "=" * 80)
    print("PROCESO FINALIZADO. Revisa la carpeta 'graficas/' para ver todas las visualizaciones.")
    print("=" * 80)

    return tabla_resultados


if __name__ == "__main__":
    main()

