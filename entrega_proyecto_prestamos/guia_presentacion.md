# Guion de presentación — Predicción de aprobación de préstamos

## Diapositiva 1. Portada

**Predicción de aprobación de préstamos mediante aprendizaje supervisado**  
Tu nombre — Curso — Fecha

## Diapositiva 2. Problema y objetivo

- Una entidad financiera debe evaluar solicitudes de crédito.
- Objetivo: predecir si una solicitud será aprobada o rechazada.
- Pregunta: ¿qué tan bien pueden las variables financieras y socioeconómicas predecir esa decisión?

## Diapositiva 3. Datos

- 4.269 solicitudes y 13 variables.
- Variable objetivo: `loan_status` (Approved / Rejected).
- Se eligió porque contiene variables numéricas y categóricas relevantes para clasificación.
- No tiene valores nulos ni filas duplicadas.

## Diapositiva 4. EDA: hallazgos

- 62,2% de solicitudes aprobadas; 37,8% rechazadas.
- El puntaje CIBIL separa claramente ambos grupos.
- Promedio CIBIL: 703 en aprobados frente a 429 en rechazados.
- Educación y trabajo independiente muestran proporciones de aprobación parecidas.

Muestra el gráfico de caja e histograma del notebook.

## Diapositiva 5. Preparación

- Eliminé `loan_id` porque es un identificador.
- Limpié espacios en texto y encabezados.
- Corregí 28 activos residenciales negativos a 0.
- Codifiqué categorías, escalé variables numéricas y dividí 80% entrenamiento / 20% prueba.

## Diapositiva 6. Modelos y métricas

- Regresión logística: línea base simple e interpretable.
- Bosque aleatorio: captura relaciones no lineales.
- Métricas: accuracy, precision, recall, F1 y ROC-AUC.

| Modelo | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| Regresión logística | 91,5% | 0,932 | 0,973 |
| Bosque aleatorio | 98,0% | 0,984 | 0,999 |

## Diapositiva 7. Resultado principal

- El bosque aleatorio fue el modelo seleccionado.
- Acertó 837 de 854 casos de prueba.
- La variable más importante fue el puntaje CIBIL (aprox. 80,6% de importancia en el modelo).
- Muestra la matriz de confusión y el gráfico de importancia.

## Diapositiva 8. Conclusión y límites

- Sí se cumplió el objetivo de predecir y comparar la aprobación.
- Valor: ayuda a priorizar revisiones y estandarizar una primera evaluación.
- No reemplaza la decisión humana.
- Antes de usarlo: validar en datos recientes, revisar sesgos y definir el costo de los errores.

## Preguntas que te pueden hacer

**¿Por qué no usaste `loan_id`?** Porque identifica el registro, no describe al solicitante; incluirlo puede introducir patrones artificiales.

**¿Por qué F1 y no solo accuracy?** F1 equilibra precisión y recall; la exactitud sola puede ocultar errores relevantes por clase.

**¿Por qué el bosque gana?** Puede representar relaciones no lineales e interacciones entre variables que la regresión logística no captura de la misma manera.

**¿El modelo toma la decisión final?** No. Es una herramienta de apoyo; se necesita revisión humana, política de riesgo y controles de equidad.
