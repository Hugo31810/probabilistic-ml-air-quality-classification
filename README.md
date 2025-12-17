# Air Quality Classification & Clustering with Probabilistic Models

Este repositorio contiene el desarrollo de una **práctica individual de Machine Learning**, centrada en el **entrenamiento, evaluación e implantación de modelos probabilísticos de clasificación**, así como en el uso de **técnicas de clustering** para el análisis exploratorio y la comparación con métodos supervisados.

La práctica ha sido realizada como parte de la asignatura **Aprendizaje Automático I** del **Grado en Inteligencia Artificial** (URJC).

---

## 👤 Autor

- **Hugo Salvador Aizpún**  
- Grado en Inteligencia Artificial  
- Universidad Rey Juan Carlos  

---

## 🎯 Objetivo del proyecto

El objetivo principal de esta práctica es:

- Diseñar un **pipeline completo de preprocesado**
- Implementar y comparar **modelos probabilísticos de clasificación**
- Evaluar el rendimiento mediante métricas estándar
- Realizar **inferencia sobre un conjunto de test**
- Analizar el comportamiento de **clustering no supervisado**
- Estudiar si el clustering puede emplearse como método de clasificación

Todo el desarrollo está documentado en la memoria asociada al proyecto :contentReference[oaicite:0]{index=0}.

---

## 🧩 Dataset y problema

El problema consiste en **clasificar la calidad del aire** en cuatro categorías:

- `Good`
- `Moderate`
- `Poor`
- `Hazardous`

Las etiquetas originales se presentan como un par `[Score, Air_Quality]`, por lo que ha sido necesario **codificar la variable categórica** para su uso en modelos de aprendizaje automático.

---

## 🔧 Pipeline de preprocesado

Siguiendo las buenas prácticas vistas en la asignatura, se ha diseñado un **pipeline de preprocesado único**, que incluye:

1. **Codificación de etiquetas**  
   - Conversión de las clases a valores enteros mediante `LabelEncoder`.

2. **Estandarización inicial**  
   - Normalización de las variables para evitar dominancias numéricas.

3. **Ingeniería de características**  
   - Generación de interacciones mediante `PolynomialFeatures`.

4. **Segunda estandarización**  
   - Ajuste de la varianza tras la creación de nuevas variables.

5. **PCA (Análisis de Componentes Principales)**  
   - Reducción de dimensionalidad conservando el **95% de la varianza**.

Este pipeline se integra directamente en los modelos de clasificación.

---

## 🤖 Modelos de clasificación implementados

Se han implementado y comparado tres sistemas probabilísticos:

### 1️⃣ Naive Bayes (GaussianNB)
- Modela cada variable de forma independiente.
- Uso explícito de **priors calculados a partir de los datos**.

### 2️⃣ MVN completo (QDA)
- Modelado multivariante normal con **covarianza completa por clase**.
- Captura correlaciones entre variables.
- Implementado mediante `QuadraticDiscriminantAnalysis`.

### 3️⃣ GMM por clase
- Un **Gaussian Mixture Model por clase**.
- Dos componentes por clase.
- Cálculo de la verosimilitud y suma con el log-prior para la predicción final.

---

## 📊 Evaluación y comparación

Se ha seguido un protocolo de validación con:

- División estratificada de los datos
- Métricas utilizadas:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Macro-F1 y Weighted-F1

### Resultados principales

| Modelo | Accuracy |
|------|----------|
| Naive Bayes | 0.857 |
| QDA (MVN) | **0.909** |
| GMM | 0.903 |

El modelo **MVN completo (QDA)** obtiene el mejor equilibrio entre rendimiento global y detección de clases minoritarias, siendo el modelo finalmente seleccionado para la fase de implantación.

---

## 🚀 Implantación e inferencia

El sistema final permite:

- Cargar los modelos entrenados desde ficheros `.pkl`
- Leer un conjunto de test
- Aplicar inferencia
- Convertir las predicciones a etiquetas originales
- Generar automáticamente el fichero de salida requerido (`*_Y_test.csv`)

---

## 🔍 Clustering

En la última parte del proyecto se analiza el uso de **clustering no supervisado**:

- Algoritmo utilizado: **K-Means**
- Número de clusters: **4**, alineado con el número de clases reales
- Análisis mediante:
  - Matriz de contingencia
  - Pureza de clusters
  - Mapeo cluster → clase

### Conclusión sobre clustering

- Clustering separa muy bien las clases `Good` y `Hazardous`
- Existe solapamiento significativo entre `Moderate` y `Poor`
- El clustering es útil para **análisis exploratorio**, pero **no sustituye a un modelo supervisado** cuando las clases se solapan

---

## 🧠 Conclusiones

- El modelo **QDA (MVN completo)** es el más adecuado para este problema
- El preprocesado y la ingeniería de características son claves para el rendimiento
- Los modelos probabilísticos permiten una interpretación clara del problema
- El clustering aporta información estructural, pero tiene limitaciones como clasificador

---

## 📚 Tecnologías utilizadas

- Python
- NumPy
- scikit-learn
- PCA
- Naive Bayes
- QDA
- Gaussian Mixture Models
- K-Means

---

## 📄 Documentación

La memoria completa del proyecto se encuentra disponible en el repositorio y recoge en detalle:

- Diseño del pipeline
- Fundamentos teóricos
- Resultados experimentales
- Conclusiones
