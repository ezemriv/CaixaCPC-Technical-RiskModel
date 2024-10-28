# Informe de Modelado de Riesgo de Crédito

## 1. Análisis Exploratorio de Datos (EDA)
   - Estadísticas descriptivas y visualización de variables.
   - Distribución de la variable objetivo (`CreditStatus`) y análisis de su relación con otras características.

## 2. Preprocesamiento
   - Manejo de valores faltantes.
   - Codificación de variables categóricas (`PaymentHistory`).
   - Normalización o estandarización de variables numéricas.

## 3. Entrenamiento y Evaluación de Modelos
   - Modelos utilizados: Ej. regresión logística, bosque aleatorio, red neuronal.
   - Comparación de métricas de evaluación (ej. precisión, AUC-ROC) para los diferentes modelos.

## 4. Optimización
   - Búsqueda de hiperparámetros mediante GridSearch o RandomizedSearch.
   - Validación cruzada (cross-validation) en el conjunto de entrenamiento.

## 5. Interpretación de Resultados
   - Técnicas de explicabilidad aplicadas (SHAP, LIME).
   - Análisis de las características más influyentes en la predicción.

## 6. Monitorización
   - Estrategia de monitorización post-despliegue:
     - Detección de drift en datos y rendimiento.
     - Métricas clave para asegurar consistencia del modelo en producción.
