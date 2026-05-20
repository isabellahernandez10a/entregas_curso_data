# =============================================================================
# TALLER — Limpieza y Análisis de Datos con Pandas y NumPy
# Autor: isabella hernandez aristizabal
# Archivo de datos: ventas_sucias_5000.csv
# =============================================================================

import pandas as pd
import numpy as np

# PARTE 1 — EXPLORACIÓN DE DATOS
print("PARTE 1 — EXPLORACIÓN DE DATOS")

# Cargar el dataset
df = pd.read_csv("ventas_sucias_5000 - copia (2).csv")

# --- Primeras filas ---
print("\nPrimeras 5 filas del dataset:")
print(df.head())

# --- Información general ---
print("\nInformación general del dataset (df.info()):")
df.info()

# --- Resumen estadístico ---
print("\nResumen estadístico (df.describe()):")
print(df.describe(include="all"))

# --- Preguntas de exploración ---
print("\nRESPUESTAS A LAS PREGUNTAS DE EXPLORACIÓN:")

filas, columnas = df.shape
print(f"\n1. ¿Cuántas filas y columnas tiene la base de datos?")
print(f"   {filas} filas y {columnas} columnas.")

print(f"\n2. ¿Qué tipos de datos identificas?")
print(df.dtypes)
print("""
    cliente   : object  (texto - nombres de clientes)
    producto  : object  (texto - nombre del producto)
    precio    : float64 (numérico - debería ser siempre positivo)
    cantidad  : object  (debería ser int, pero contiene valores como 'three')
    pais      : object  (texto - con inconsistencias de capitalización/alias)
    metodo_pago: object (texto - con variaciones de mayúsculas)
    fecha     : object  (debería ser datetime)
""")

print(f"3. ¿Encuentras columnas con problemas o inconsistencias?")
print(f"   → 'cantidad' tiene valores no numéricos ('three') → tipo incorrecto.")
print(f"   → 'precio' tiene valores nulos (NaN).")
print(f"   → 'cantidad' tiene valores nulos (NaN).")
print(f"   → 'pais' tiene variantes: 'colombia', 'Colombia', 'COL', 'col', 'peru', 'Perú', etc.")
print(f"   → 'metodo_pago' tiene variantes: 'transferencia', 'TRANSFERENCIA', 'Tarjeta', etc.")
print(f"   → 'fecha' está en formato string, no datetime.")
print(f"   → Existe al menos un precio outlier (999999.0).")

# PARTE 2 — LIMPIEZA DE DATOS
print("PARTE 2 — LIMPIEZA DE DATOS")
# Trabajamos sobre una copia para no perder el original
df_limpio = df.copy()

# --- 1. Corregir tipo de 'cantidad' ---
# Primero reemplazamos valores no numéricos textuales conocidos
df_limpio["cantidad"] = df_limpio["cantidad"].replace("three", np.nan)
# Convertimos a numérico; los valores que no puedan convertirse quedan como NaN
df_limpio["cantidad"] = pd.to_numeric(df_limpio["cantidad"], errors="coerce")
print(f"\n'cantidad': valores 'three' reemplazados por NaN y columna convertida a numérico.")

# --- 2. Corregir tipo de 'precio' ---
df_limpio["precio"] = pd.to_numeric(df_limpio["precio"], errors="coerce")
print(f"'precio': convertido a numérico (ya era float, verificado).")

# --- 3. Corregir tipo de 'fecha' ---
df_limpio["fecha"] = pd.to_datetime(df_limpio["fecha"], errors="coerce")
print(f"'fecha': convertida a datetime.")

# --- 4. Valores nulos ---
print(f"\n Valores nulos antes de tratarlos:")
print(df_limpio.isnull().sum())

nulos_precio_antes = df_limpio["precio"].isnull().sum()
nulos_cantidad_antes = df_limpio["cantidad"].isnull().sum()

# Imputamos precio con la mediana (más robusta ante outliers)
mediana_precio = df_limpio["precio"].median()
df_limpio["precio"] = df_limpio["precio"].fillna(mediana_precio)
print(f"\n 'precio': {nulos_precio_antes} nulos imputados con la mediana ({mediana_precio:.2f}).")

# Imputamos cantidad con la mediana
mediana_cantidad = df_limpio["cantidad"].median()
df_limpio["cantidad"] = df_limpio["cantidad"].fillna(mediana_cantidad)
print(f"'cantidad': {nulos_cantidad_antes} nulos imputados con la mediana ({mediana_cantidad:.0f}).")

# Convertir cantidad a entero (usamos Int64 que soporta NA, aunque ya no debería haberlos)
df_limpio["cantidad"] = df_limpio["cantidad"].astype("Int64")
print(f"'cantidad': convertida a entero (Int64).")

# --- 5. Estandarizar columna 'pais' ---
# Mapeamos todas las variantes a un nombre canónico
mapa_pais = {
    "colombia": "Colombia",
    "Colombia": "Colombia",
    "col":      "Colombia",
    "COL":      "Colombia",
    "chile":    "Chile",
    "Chile":    "Chile",
    "peru":     "Perú",
    "Perú":     "Perú",
    "perú":     "Perú",
}
df_limpio["pais"] = df_limpio["pais"].map(mapa_pais).fillna(df_limpio["pais"])
print(f"\n 'pais' estandarizado. Valores únicos: {df_limpio['pais'].unique()}")

# --- 6. Estandarizar columna 'metodo_pago' ---
df_limpio["metodo_pago"] = df_limpio["metodo_pago"].str.strip().str.title()
# Normalizar "Transferencia" que puede quedar como "Transferencia"
print(f"'metodo_pago' estandarizado. Valores únicos: {df_limpio['metodo_pago'].unique()}")

# --- 7. Revisar outliers en 'precio' con IQR ---
Q1 = df_limpio["precio"].quantile(0.25)
Q3 = df_limpio["precio"].quantile(0.75)
IQR = Q3 - Q1
limite_superior = Q3 + 3 * IQR  # usamos 3×IQR para no ser demasiado agresivos
outliers_precio = df_limpio[df_limpio["precio"] > limite_superior]
print(f"\n Outliers en 'precio' (por encima de {limite_superior:.2f}):")
print(outliers_precio[["cliente", "producto", "precio", "cantidad", "pais"]])

# Eliminamos filas con precio outlier extremo (999999 es claramente un error)
registros_antes = len(df_limpio)
df_limpio = df_limpio[df_limpio["precio"] <= limite_superior]
registros_despues = len(df_limpio)
print(f"Eliminados {registros_antes - registros_despues} registros con precios outlier extremos.")

# --- Resumen final de limpieza ---
print(f"\nValores nulos tras la limpieza:")
print(df_limpio.isnull().sum())
print(f"\nDimensiones finales del dataset limpio: {df_limpio.shape}")

print("""
📝 RESPUESTAS PARTE 2:
   Problemas encontrados:
   - 'cantidad' contenía strings como 'three' en lugar de números → reemplazado por NaN y luego imputado con la mediana.
   - 'precio' y 'cantidad' tenían valores nulos → imputados con la mediana de cada columna (más robusta que la media ante outliers).
   - 'pais' tenía múltiples variantes del mismo país → estandarizado con un diccionario de mapeo.
   - 'metodo_pago' tenía mayúsculas/minúsculas inconsistentes → estandarizado con .str.title().
   - 'fecha' era tipo object → convertida a datetime.
   - 'precio' = 999999 es un valor atípico extremo → eliminado (1 registro).
   
   ¿Se eliminaron registros?
   → Sí, se eliminó 1 registro con precio = 999999, que corresponde a un error de captura evidente.
   → Los nulos en precio y cantidad fueron IMPUTADOS (no eliminados) para conservar la mayor cantidad de datos.
""")



# PARTE 3 — ANÁLISIS CON PANDAS

print("=" * 60)
print("PARTE 3 — ANÁLISIS CON PANDAS")


# Nueva columna 'total'
df_limpio["total"] = df_limpio["precio"] * df_limpio["cantidad"]

total_vendido  = df_limpio["total"].sum()
promedio_ventas = df_limpio["total"].mean()
venta_maxima   = df_limpio["total"].max()
venta_minima   = df_limpio["total"].min()

print(f"\nTotal vendido:       ${total_vendido:>15,.2f}")
print(f"Promedio de ventas:  ${promedio_ventas:>15,.2f}")
print(f"Venta máxima:        ${venta_maxima:>15,.2f}")
print(f"Venta mínima:        ${venta_minima:>15,.2f}")

# Top 5 productos por valor total vendido
print("\nTop 5 productos con mayor valor total vendido:")
top5_productos = (
    df_limpio.groupby("producto")["total"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)
print(top5_productos.apply(lambda x: f"${x:,.2f}"))

# País con más ventas (por suma total)
print("\nPaís con más ventas (total $):")
ventas_por_pais = df_limpio.groupby("pais")["total"].sum().sort_values(ascending=False)
print(ventas_por_pais.apply(lambda x: f"${x:,.2f}"))
print(f"\n   → El país con más ventas es: {ventas_por_pais.idxmax()} con ${ventas_por_pais.max():,.2f}")

# PARTE 4 — INTRODUCCIÓN A NUMPY
print("\n" + "=" * 60)
print("PARTE 4 — INTRODUCCIÓN A NUMPY")

# Convertir columnas numéricas a array de NumPy
data = df_limpio[["precio", "cantidad"]].to_numpy()
print(f"\nArray NumPy 'data' — shape: {data.shape}, dtype: {data.dtype}")
print(f"   Primeras 3 filas:\n{data[:3]}")

# Separar columnas usando indexación
precios    = data[:, 0]   # Columna 0 → precios
cantidades = data[:, 1]   # Columna 1 → cantidades

print(f"\n→ precios    (primeros 5): {precios[:5]}")
print(f"→ cantidades (primeros 5): {cantidades[:5]}")

# Vectorización para calcular ventas totales
totales = precios * cantidades
print(f"\n→ totales    (primeros 5): {totales[:5]}")

# PARTE 5 — ANÁLISIS CON NUMPY
# =============================================================================
print("\n" + "=" * 60)
print("PARTE 5 — ANÁLISIS CON NUMPY")
suma_total_np     = np.sum(totales)
promedio_np       = np.mean(totales)
venta_maxima_np   = np.max(totales)
ventas_sup_1000   = np.sum(totales > 1000)

print(f"\n Suma total de ventas (NumPy):          ${suma_total_np:>15,.2f}")
print(f"Promedio de ventas (NumPy):            ${promedio_np:>15,.2f}")
print(f"Venta máxima (NumPy):                  ${venta_maxima_np:>15,.2f}")
print(f"Cantidad de ventas superiores a 1000:  {ventas_sup_1000} registros")

print("""
RESPUESTAS PARTE 5:

1. ¿Qué ventajas observas al usar NumPy?
   → NumPy opera sobre arrays en memoria de manera muy eficiente usando 
     operaciones vectorizadas implementadas en C. Es significativamente más 
     rápido que iterar con bucles for en Python puro, especialmente con 
     miles o millones de datos.

2. ¿Qué significa "vectorización"?
   → Es la capacidad de aplicar una operación a TODOS los elementos de un 
     array al mismo tiempo, sin necesidad de escribir un bucle explícito. 
     Por ejemplo: `precios * cantidades` multiplica cada par de elementos 
     de ambos arrays en una sola instrucción, delegando el ciclo al motor 
     interno de NumPy (en C), lo que lo hace mucho más rápido.

3. ¿Qué hace la expresión data[:, 0]?
   → Selecciona TODAS las filas (:) de la columna con índice 0 del array 2D.
     En otras palabras, extrae la primera columna completa del array,
     que en nuestro caso corresponde a los precios.
""")

# PARTE 6 — INTERPRETACIÓN DE RESULTADOS
# =============================================================================
print("=" * 60)
print("PARTE 6 — INTERPRETACIÓN DE RESULTADOS")
print(f"""
Análisis e interpretación:

1. ¿Los resultados tienen sentido?
   → En términos generales, sí. El total vendido de ${suma_total_np:,.2f} 
     y el promedio de ${promedio_np:,.2f} por transacción son valores 
     coherentes para un dataset de ventas de electrónica con precios 
     entre $20 y $2000 y cantidades de 1 a 9 unidades.

2. ¿Detectaste valores sospechosos?
   → Sí. Se detectó y eliminó un precio de 999999.0 que era claramente 
     un error de digitación (posible valor de relleno o prueba).
   → Algunos precios bajos (ej. $16, $21) para productos como Monitor 
     o Laptop podrían ser sospechosos, aunque se conservaron al no 
     poder confirmar que son errores.
   → Se encontraron registros con cantidades escritas como texto ('three') 
     que fueron tratados como datos faltantes.

3. ¿El promedio representa correctamente los datos?
   → El promedio puede estar influenciado por los extremos (precio muy bajo 
     o muy alto). Sería más representativo acompañarlo con la mediana y 
     la desviación estándar para entender la dispersión real.
   → Mediana del total: ${np.median(totales):,.2f} vs Promedio: ${promedio_np:,.2f}
     {"→ Distribución relativamente simétrica." if abs(np.median(totales) - promedio_np) < promedio_np * 0.1 else "→ Existe cierta asimetría en los datos."}

4. ¿Qué decisiones tomarías si esta fuera información real de negocio?
   → Investigar precios extremadamente bajos (< $50) para productos de alto 
     valor: podrían ser errores de captura o descuentos excepcionales.
   → Establecer reglas de validación en el sistema fuente para evitar 
     valores como 'three' en campos numéricos.
   → Revisar las fechas: existen fechas atípicas (ej. 2024-05-01) dentro 
     de una secuencia que va principalmente de enero a julio.
   → Estandarizar los campos 'pais' y 'metodo_pago' directamente en el 
     sistema de captura para evitar limpiezas manuales recurrentes.
   → Analizar la rentabilidad por país y producto para tomar decisiones 
     de inventario y marketing.
""")