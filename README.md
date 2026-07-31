# Segmentación de usuarios de Steam mediante Clustering

Análisis de comportamiento de usuarios a partir del dataset `steam-200k.csv`
(interacciones usuario-videojuego), aplicando K-Means para construir
segmentos accionables de negocio.

## Estructura del repositorio

```
├── analisis_steam.py          # Script principal (ejecutar con: python analisis_steam.py)
├── requirements.txt           # Dependencias
├── data/
│   └── steam-200k.csv         # Dataset original
└── outputs/
    ├── 01_top_juegos_comprados.png
    ├── 02_top_juegos_horas.png
    ├── 03_distribucion_horas_log.png
    ├── 04_juegos_por_usuario.png
    ├── 05_boxplot_features_log.png
    ├── 06_seleccion_k.png
    ├── 07_clusters_pca.png
    ├── 08_boxplots_por_cluster.png
    ├── 09_distribucion_clusters.png
    ├── usuarios_segmentados.csv   # Cada usuario con su cluster/segmento asignado
    └── perfil_clusters.csv        # Perfil promedio de cada cluster
```

## Cómo ejecutar

```bash
pip install -r requirements.txt
python analisis_steam.py
```

El script imprime en consola cada etapa del análisis (EDA, preparación,
modelado, interpretación) y guarda todas las figuras y tablas en `outputs/`.

## 1. Datos y exploración

El dataset original tiene 200,000 filas en formato "largo": cada fila es un
evento `usuario–juego–tipo de evento` (`purchase` o `play`), sin encabezado.
No tiene valores nulos, pero sí 707 filas duplicadas exactas que se eliminan.
Contiene 12,393 usuarios únicos y 5,155 juegos únicos. La columna extra
(siempre 0) se descarta por no aportar información.

Las horas jugadas y la cantidad de juegos comprados por usuario están muy
sesgadas a la derecha: la mayoría de usuarios tiene pocos juegos/horas,
mientras un grupo pequeño concentra volúmenes muy altos (usuarios "power").

## 2. Preparación de los datos

El dataset se transforma de formato de interacciones a una tabla **por
usuario**, agregando las siguientes variables:

| Variable | Descripción | Justificación |
|---|---|---|
| `n_games_purchased` | Nº de juegos distintos comprados | Tamaño de la biblioteca/colección |
| `n_games_played` | Nº de juegos distintos jugados | Cuántos de los comprados realmente usa |
| `total_hours` | Suma de horas jugadas | Volumen total de consumo |
| `avg_hours_per_game` | Promedio de horas por juego jugado | Profundidad de juego |
| `max_hours_single_game` | Horas del juego más jugado | Presencia de un juego "ancla" |
| `engagement_ratio` | `n_games_played / n_games_purchased` (máx. 1) | % de la biblioteca efectivamente usada |

Se aplica **transformación logarítmica (`log1p`)** a las variables de
conteo/horas por su fuerte asimetría, y luego **estandarización
(`StandardScaler`)** para que ninguna variable domine el cálculo de
distancia euclidiana usado por K-Means.

## 3. Modelado

Se evaluó K-Means para *k* entre 2 y 8 usando el **método del codo** y el
**coeficiente de silueta**. El mejor equilibrio se obtiene en **k = 4**
(silhouette ≈ 0.44), que además permite diferenciar matices de negocio
relevantes (comprador vs. jugador, casual vs. intensivo) que k=2 no
capturaría.

**Por qué K-Means:** el espacio de variables es continuo y de dimensión
moderada, los clusters resultantes son razonablemente esféricos tras el
escalamiento, y los centroides son directamente interpretables como
"perfil promedio" de cada segmento — ideal para conclusiones de negocio.

Como validación se probó **DBSCAN**, que con datos de densidad muy variable
(usuarios casuales vs. power users) dejó prácticamente todos los usuarios en
un único cluster grande, sin aportar segmentación útil para marketing. El
clustering jerárquico se descartó como algoritmo principal por su costo
computacional en +12,000 registros.

## 4. Interpretación de los clusters

| Cluster | Segmento | # Usuarios | % | Compras prom. | Horas totales prom. | Engagement |
|---|---|---|---|---|---|---|
| 0 | **Jugadores Casuales** | 5,049 | 40.7% | 1.5 | 6.3 | 0.97 |
| 1 | **Coleccionistas Muy Activos** | 1,765 | 14.2% | 56.1 | 1,168.4 | 0.63 |
| 2 | **Usuarios Muy Activos (Hardcore)** | 3,486 | 28.1% | 3.9 | 386.3 | 0.77 |
| 3 | **Compradores Ocasionales / Pasivos** | 2,093 | 16.9% | 4.1 | 2.3 | 0.16 |

**Cluster 0 – Jugadores Casuales (40.7%, el más numeroso).**
Compran muy pocos juegos (≈1-2) y prácticamente todos los que compran los
llegan a jugar (engagement 0.97), pero con pocas horas totales. Son usuarios
livianos: probablemente compraron un solo juego puntual o son relativamente
nuevos en la plataforma.

**Cluster 1 – Coleccionistas Muy Activos (14.2%).**
Compran en promedio 56 juegos y acumulan más de 1,100 horas jugadas en
total. Son compradores frecuentes que además consumen buena parte de lo que
compran. Es el segmento más valioso en términos de gasto potencial y
volumen de uso de la plataforma (los "power users"/ballenas).

**Cluster 2 – Usuarios Muy Activos / Hardcore (28.1%).**
Compran relativamente pocos juegos (≈4) pero acumulan muchísimas horas
(promedio 386, con hasta 228 horas por juego en promedio). Son jugadores muy
frecuentes, leales a pocos títulos "ancla" en los que invierten mucho
tiempo — perfil de jugador dedicado más que coleccionista.

**Cluster 3 – Compradores Ocasionales / Pasivos (16.9%).**
Compran una cantidad moderada de juegos (≈4) pero casi no los juegan
(engagement 0.16, horas totales casi nulas). Es probable que hayan adquirido
juegos en bundles/ofertas o por impulso sin llegar a activarlos: representan
"compra sin uso" y riesgo de abandono/reembolso.

## 5. Recomendaciones de negocio

**Cluster 0 – Jugadores Casuales**
- *Estrategia:* activación y aumento de frecuencia de uso, no venta agresiva.
- *Promociones:* recomendaciones de juegos similares a su única compra,
  descuentos de entrada baja (bundles pequeños, DLCs económicos) para
  incentivar una segunda compra.
- *Fidelización:* prioridad media-baja; útil para campañas de "onboarding"
  (tutoriales, logros, recordatorios de reanudar el juego).

**Cluster 1 – Coleccionistas Muy Activos**
- *Estrategia:* retención VIP. Es el segmento de mayor valor: ya compran
  mucho y juegan mucho.
- *Promociones:* acceso anticipado a lanzamientos, ediciones premium/DLC
  exclusivos, programas de puntos o recompensas por lealtad.
- *Fidelización:* **máxima prioridad** para campañas de fidelización y
  cross-selling (nuevos títulos de géneros que ya consumen).

**Cluster 2 – Usuarios Muy Activos (Hardcore)**
- *Estrategia:* profundizar el compromiso con sus juegos "ancla" y ampliar
  su catálogo hacia títulos similares.
- *Promociones:* contenido adicional (DLC, expansiones, cosméticos) de los
  juegos que ya dominan; recomendaciones basadas en similitud de género con
  su juego principal.
- *Fidelización:* alta prioridad — son usuarios comprometidos con alto
  potencial de gasto adicional dentro de su(s) juego(s) favorito(s).

**Cluster 3 – Compradores Ocasionales / Pasivos**
- *Estrategia:* reactivación. El riesgo es que dejen de comprar al no
  percibir valor de lo ya adquirido.
- *Promociones:* recordatorios y notificaciones de "juegos sin estrenar en
  tu biblioteca", ofertas de bajo compromiso, contenido introductorio corto
  para bajar la barrera de entrada.
- *Fidelización:* prioridad baja para campañas de fidelización tradicional,
  pero prioridad alta para campañas de **reactivación/win-back** antes de
  perderlos definitivamente.

**Información adicional que mejoraría la segmentación**
- Fecha/antigüedad de las compras y de la cuenta (permitiría medir
  recencia y tendencia, no solo volumen acumulado).
- Género/categoría de cada juego (para segmentar también por preferencias
  temáticas, no solo por intensidad de uso).
- Gasto real en dinero (precio pagado, si hubo descuento), ya que el
  dataset actual no tiene precios, solo eventos de compra.
- Uso de funciones sociales (amigos, logros, reseñas) para medir
  engagement con la comunidad.
- Plataforma/dispositivo y ubicación geográfica, útiles para campañas
  segmentadas por región o para campañas cross-platform.
