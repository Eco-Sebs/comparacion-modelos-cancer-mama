# Predicción probabilística de cáncer de mama con CatBoost

Proyecto de aprendizaje sobre el dataset **Breast Cancer Wisconsin (Diagnostic)**. En vez de responder solo "maligno" o "benigno", el modelo entrega un **porcentaje de probabilidad** de que un tumor sea maligno, explica **por qué** da ese porcentaje y compara CatBoost contra otros algoritmos.

> **Aviso:** es un proyecto educativo. No es una herramienta de diagnóstico y no debe usarse para tomar decisiones médicas, ADEMAS LOS RESULTADOS SON ASI POR QUE ES UN DATASET FACIL.

## Idea del proyecto

1. Construir un modelo **probabilístico** (un score de riesgo en %) y no solo un clasificador de etiquetas.
2. Buscar los mejores hiperparámetros de CatBoost con validación cruzada.
3. Comparar CatBoost contra regresión logística, Random Forest y XGBoost.
4. Explicar cada predicción con valores SHAP.
5. Entender **por qué** gana el modelo que gana y en qué casos ganarían los otros.

## Dataset

**Breast Cancer Wisconsin (Diagnostic)**, también conocido como WDBC.

| | |
|---|---|
| Origen | UCI Machine Learning Repository (donado el 31/10/1995) |
| Creadores | William Wolberg, Olvi Mangasarian, Nick Street |
| Casos | 569 (357 benignos y 212 malignos) |
| Variables | 30 numéricas: 10 características de los núcleos celulares, cada una como media, error estándar y "peor" valor |
| Variable objetivo | `diagnosis`: `M` (maligno) o `B` (benigno) |
| Valores faltantes | Ninguno |
| Licencia | CC BY 4.0 (según UCI) |

Las variables se calculan a partir de una imagen digitalizada de una punción con aguja fina (FNA) de una masa mamaria y describen los núcleos celulares (radio, textura, perímetro, área, suavidad, compacidad, concavidad, puntos cóncavos, simetría y dimensión fractal). El modelo trabaja con esas variables ya calculadas, no con las imágenes.

### De dónde descargarlo

- **Kaggle (el que usa el cuaderno):** https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data. Descarga `data.csv`. Trae las columnas `id`, `diagnosis`, las 30 variables y una columna vacía `Unnamed: 32` que el cuaderno elimina.
- **UCI (fuente original):** https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic
- **Con Python desde UCI:**
  ```python
  # pip install ucimlrepo
  from ucimlrepo import fetch_ucirepo
  wdbc = fetch_ucirepo(id=17)
  X = wdbc.data.features
  y = wdbc.data.targets
  ```
- **Con scikit-learn:** `sklearn.datasets.load_breast_cancer()` trae el mismo dataset. Al probarlo, los resultados coincidieron con los del CSV de Kaggle.

El cuaderno está escrito para el CSV de Kaggle. Con las otras dos opciones los nombres de las columnas son distintos (por ejemplo `radius1` en UCI, o `mean radius` en scikit-learn), y en scikit-learn la etiqueta viene invertida (0 = maligno), así que habría que adaptar la preparación de datos.

El archivo de datos no se incluye en este repositorio: descárgalo con alguna de las opciones anteriores y colócalo como `data.csv` junto al cuaderno.

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `breast_cancer_catboost_completo.ipynb` | Cuaderno principal (pensado para Google Colab) |
| `experimentos_adicionales.py` | Script que reproduce los experimentos de la sección "Experimentos adicionales" |
| `demo_fronteras.png` | Gráfico que genera el script |

## Cómo ejecutarlo

**En Google Colab:** sube el cuaderno (Archivo → Subir cuaderno) y ejecuta las celdas en orden. En la sección 2 te pedirá subir `data.csv`.

**En local:**

```bash
pip install catboost xgboost scikit-learn pandas numpy matplotlib jupyter
jupyter notebook breast_cancer_catboost_completo.ipynb
```

Para el script adicional (con `data.csv` en la misma carpeta):

```bash
python experimentos_adicionales.py
```

La semilla aleatoria es 42. Los resultados pueden variar levemente según las versiones de las librerías.

## Metodología

El cuaderno sigue estos pasos:

1. **Análisis de la información:** tamaño, tipos, nulos, duplicados y distribución de clases (62.7 % benignos y 37.3 % malignos).
2. **Preparación:** se eliminan `id` y `Unnamed: 32`; `y = 1` si es maligno y `0` si es benigno.
3. **Separación:** 80 % entrenamiento (455 casos) y 20 % prueba (114 casos), estratificada.
4. **Búsqueda de hiperparámetros de CatBoost:** `RandomizedSearchCV` con 15 combinaciones y validación cruzada de 5 particiones. Se puntúa con **log-loss** porque interesa que las probabilidades sean buenas, no solo que ordenen bien.
5. **Comparación de algoritmos:** regresión logística, Random Forest, XGBoost y CatBoost con la misma validación cruzada.
6. **Evaluación en el set de prueba:** AUC, log-loss, Brier score, reporte de clasificación, curva ROC y curva de calibración.
7. **Explicabilidad:** importancia de variables y valores SHAP de un caso individual.

Un detalle de método: el set de prueba nunca se usa para entrenar ni para elegir hiperparámetros. Al principio el entrenamiento pasaba el set de prueba como `eval_set` de CatBoost, lo que le permitía elegir el número de iteraciones mirando esos datos; se corrigió.

## Resultados

### Búsqueda de hiperparámetros

Mejor combinación: `depth=3`, `learning_rate=0.05`, `iterations=300`, `l2_leaf_reg=5` (log-loss de 0.0839 y AUC de 0.9931 en validación cruzada).

### Comparación de modelos (validación cruzada sobre entrenamiento)

| Modelo | Log-loss | AUC | Brier |
|---|---|---|---|
| **Regresión logística** | **0.0724** ± 0.023 | 0.9958 | 0.0191 |
| CatBoost (afinado) | 0.0839 ± 0.021 | 0.9931 | 0.0217 |
| XGBoost | 0.0859 ± 0.030 | 0.9936 | 0.0230 |
| Random Forest | 0.1243 ± 0.027 | 0.9880 | 0.0319 |

Log-loss y Brier: más bajo es mejor. El `±` es la desviación entre particiones.

### Set de prueba

| Modelo | AUC | Log-loss | Brier |
|---|---|---|---|
| Regresión logística | 0.996 | 0.0773 | 0.0213 |
| CatBoost (afinado) | 0.998 | 0.0835 | 0.0246 |

Con CatBoost y umbral de 50 %: exactitud de 0.96; para los benignos, precisión de 0.95 y recall de 1.00; para los malignos, precisión de 1.00 y recall de 0.90.

```
Matriz de confusión (filas = real, columnas = predicho) [Benigno, Maligno]
[[72  0]
 [ 4 38]]
```

En un contexto médico importa el recall de los malignos: con este umbral el modelo dejó pasar 4 de 42 tumores malignos. Bajar el umbral aumentaría el recall a costa de más falsas alarmas.

## Cómo explica el modelo cada predicción

CatBoost calcula el aporte de cada variable a la predicción (valores SHAP). La probabilidad se arma así:

`punto de partida + suma de los aportes de las 30 variables = predicción final` (en log-odds, luego convertida a porcentaje con la sigmoide).

- **El punto de partida** es el promedio de las puntuaciones internas del modelo sobre los datos de entrenamiento. En el cuaderno salió 19.8 %, y no el 37 % de malignos que hay en los datos, porque el promedio se hace en log-odds y no en porcentajes.
- **Aporte negativo** empuja hacia benigno; **aporte positivo**, hacia maligno.
- **Ejemplo (caso 0 del set de prueba):** el modelo parte de 19.8 % y termina en 0.2 % de ser maligno. Las variables `texture_worst`, `radius_worst`, `area_se`, `texture_mean` y `perimeter_worst` restan más, porque el caso tiene valores bajos, más cercanos al promedio de los benignos que al de los malignos.
- **Por qué parece un salto:** casi todas las variables empujan al mismo lado, muchas miden cosas parecidas (área, perímetro y radio miden el tamaño), y la sigmoide comprime los extremos, así que los mismos pasos quitan muchos puntos porcentuales al inicio y muy pocos al final.

## Conclusiones

### Por qué gana la regresión logística

- **Los datos son casi linealmente separables.** Las clases se distinguen por valores claramente distintos en variables de tamaño e irregularidad celular (por ejemplo, `area_worst` promedia 562 en benignos y 1442 en malignos). Un modelo lineal ya resuelve casi todo el problema, y los árboles necesitan muchos cortes para aproximar una diagonal.
- **Hay pocos datos.** Con 455 casos de entrenamiento y 30 variables, un ensamble de 300 árboles tiene más flexibilidad de la necesaria y puede ajustar ruido. La regresión logística tiene 31 parámetros y viene regularizada.
- **Sus probabilidades son más suaves.** El log-loss premia probabilidades bien calibradas, algo natural en la regresión logística y menos en Random Forest, que quedó último.

Nota sobre los modelos: CatBoost, XGBoost y Random Forest usan árboles de decisión, no regresión lineal. La regresión logística, a pesar del nombre, es un modelo de clasificación con frontera lineal.

### Cómo funciona cada modelo

| Modelo | Idea básica | Fortaleza | Límite |
|---|---|---|---|
| Regresión logística | Suma ponderada de las variables pasada por la sigmoide; frontera recta | Simple, rápida y explicable con coeficientes | No captura curvas ni interacciones por sí sola |
| Random Forest | Muchos árboles independientes que votan; la probabilidad es el promedio | Robusto y no lineal | Probabilidades comprimidas y peor calibradas |
| XGBoost | *Boosting*: árboles pequeños en secuencia, cada uno corrige los errores del anterior | Muy potente en datos tabulares | Sensible a sus parámetros |
| CatBoost | *Boosting* con árboles simétricos y técnicas contra el sobreajuste; maneja variables categóricas | Estable, buenas probabilidades y explicable con SHAP | Más lento y más difícil de explicar que uno lineal |

### Cuándo ganarían los modelos de árboles

Cuando los datos no se separan con una línea recta: fronteras curvas, interacciones entre variables (el efecto de una depende del valor de otra), umbrales bruscos, variables categóricas, valores faltantes, escalas muy distintas o muchos datos con patrones complejos. La regresión logística puede cubrir parte de esto si se le agregan variables a mano (términos cuadráticos o productos), pero eso exige saber de antemano qué relaciones buscar; los árboles las descubren solos.

### Precauciones

- Las diferencias entre regresión logística, CatBoost y XGBoost son pequeñas (aprox. 0.01 de log-loss), comparables a la variación entre particiones.
- Solo CatBoost se afinó con búsqueda de hiperparámetros; los demás usaron valores por defecto o fijos, así que la comparación no es del todo pareja.
- El set de prueba tiene solo 114 casos, por lo que sus métricas son ruidosas. CatBoost tuvo mejor AUC en prueba (0.998 vs 0.996) y peor log-loss (0.0835 vs 0.0773).
- Este dataset es fácil, limpio y muy estudiado. Las conclusiones no se generalizan a datos más ruidosos o complejos.

### Decisión del proyecto

Con diferencias tan pequeñas se suele preferir el modelo más simple, la regresión logística, que además se explica con sus coeficientes. En el cuaderno se mantiene CatBoost para las secciones de predicción y explicación porque permite el análisis caso por caso con valores SHAP.

## Experimentos adicionales

Estos experimentos no están en el cuaderno; se reproducen con `experimentos_adicionales.py`.

**1. ¿La ventaja de la regresión logística es real o ruido?** Con validación cruzada repetida (5 particiones × 4 repeticiones = 20), la regresión logística ganó a CatBoost en 14 de 20 particiones (diferencia media de 0.012 en log-loss) y a XGBoost en 15 de 20 (diferencia media de 0.018). Es una ventaja pequeña pero bastante consistente.

**2. Curva de aprendizaje.** Log-loss en validación según el número de casos de entrenamiento:

| Casos | Regresión logística | CatBoost | XGBoost | CatBoost − regresión |
|---|---|---|---|---|
| 54 | 0.115 | 0.236 | 0.219 | 0.121 |
| 109 | 0.092 | 0.169 | 0.168 | 0.077 |
| 182 | 0.083 | 0.114 | 0.131 | 0.031 |
| 273 | 0.076 | 0.093 | 0.098 | 0.016 |
| 364 | 0.072 | 0.084 | 0.086 | 0.012 |

La regresión logística gana con todos los tamaños, y con más datos la diferencia se reduce (de 0.121 a 0.012). Los últimos pasos se aplanan, así que con estos datos no se puede afirmar que CatBoost llegaría a superarla.

**3. Demostración con problemas no lineales.** Dos problemas artificiales de 600 casos: unas *lunas* entrelazadas (frontera curva) y un problema tipo *XOR* (la clase depende de la interacción entre dos variables).

| Modelo | Lunas: log-loss / AUC | XOR: log-loss / AUC |
|---|---|---|
| Regresión logística | 0.324 / 0.939 | 0.689 / 0.549 |
| Random Forest | 0.356 / 0.970 | 0.238 / 0.957 |
| XGBoost | 0.195 / 0.976 | 0.249 / 0.955 |
| CatBoost | 0.182 / 0.979 | 0.237 / 0.954 |

![Fronteras de decisión: regresión logística vs CatBoost](demo_fronteras.png)

- En las lunas, CatBoost y XGBoost superan claramente a la regresión logística, porque la frontera es una curva.
- En el XOR, la regresión logística queda al nivel del azar (AUC 0.55 y log-loss 0.69), porque ninguna recta separa las clases. Los tres modelos de árboles lo resuelven.
- Random Forest ordena bien los casos (buen AUC), pero en las lunas su log-loss es peor que el de la regresión logística: un ejemplo de sus probabilidades menos calibradas.

## Limitaciones

- Son datos de un solo origen (la Universidad de Wisconsin, Madison) y de una época concreta (años noventa); no hay validación externa.
- Las variables vienen calculadas por un software de análisis de imagen, no de las imágenes originales.
- Con 569 casos, los resultados tienen incertidumbre. La comparación entre modelos se apoya en validación cruzada, no en el set de prueba.

## Posibles mejoras

- Afinar también la regresión logística, XGBoost y Random Forest para una comparación pareja.
- Calibrar las probabilidades (por ejemplo con `CalibratedClassifierCV`) y revisar la curva de calibración.
- Ajustar el umbral de decisión para priorizar el recall de los malignos.
- Explicar la regresión logística con sus coeficientes y compararla con SHAP.
- Repetir el estudio con un dataset médico más ruidoso o con relaciones no lineales, donde los árboles tengan más oportunidad de ganar.

## Créditos y referencias

- **Dataset:** Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B. Licencia CC BY 4.0.
- **Artículo de origen:** Street, W., Wolberg, W., & Mangasarian, O. (1993). *Nuclear feature extraction for breast tumor diagnosis*. Electronic Imaging.
- **Copia en Kaggle:** https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
- **Librerías:** CatBoost, XGBoost, scikit-learn, pandas, NumPy y Matplotlib.
