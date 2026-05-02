# Referencia de Configuración — config.json

Toda la configuración del proyecto se gestiona desde un único archivo en la raíz: `config.json`.  
Ningún script tiene valores hardcodeados; todo lo que necesites ajustar pasa por aquí.

---

## Estructura general

```json
{
    "general":       { ... },   ← columnas, rutas de datos, idioma
    "preprocessing": { ... },   ← split, texto, escalado, balanceo
    "train":         { ... },   ← modelos, hiperparámetros, CV
    "test":          { ... },   ← rutas de salida de métricas
    "clustering":    { ... },   ← LDA: plataforma, tópicos, coherencia
    "generative":    { ... }    ← número de generaciones del modelo Ollama
}
```

---

## Sección `general`

Controla qué columnas usar y dónde están los datos.

```json
"general": {
    "random_state": 42,
    "column": "sentiment",
    "text_features": ["review"],
    "drop_features": ["reviewId", "score", "location", "date", "App", "gender"],
    "language": "english",
    "data": {
        "train_dev": "data/train.csv",
        "dev_file": null,
        "test": "data/test.csv"
    }
}
```

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `random_state` | `int` | Semilla aleatoria. Usar siempre el mismo valor garantiza reproducibilidad. |
| `column` | `string` | Nombre de la columna objetivo (la que el modelo tiene que predecir). |
| `text_features` | `list[string]` | Columnas de texto libre. Se procesan con TF-IDF o BoW. |
| `drop_features` | `list[string]` | Columnas que se eliminan antes de entrenar (IDs, fechas, etc.). |
| `language` | `string` | Idioma para las stopwords de NLTK. Valores: `"english"`, `"spanish"`, `"french"`, etc. |
| `data.train_dev` | `string` | Ruta al CSV de entrenamiento + validación. |
| `data.dev_file` | `string\|null` | Ruta a un CSV de validación separado. `null` si no existe (se crea la división automáticamente). |
| `data.test` | `string` | Ruta al CSV de test final. |

---

## Sección `preprocessing`

Controla cómo se transforman los datos antes de entrenar.

```json
"preprocessing": {
    "test_size": 0.25,
    "text_process": "tf_idf",
    "sampling": "oversampling",
    "impute_strategy_numeric": "mean",
    "impute_strategy_categorical": "most_frequent",
    "scaling": "minmax"
}
```

| Campo | Valores posibles | Descripción |
|-------|-----------------|-------------|
| `test_size` | `0.0` – `1.0` | Proporción del dataset dedicada a validación (dev). `0.25` = 25%. |
| `text_process` | `"tf_idf"` · `"bow"` | Vectorización del texto. TF-IDF penaliza términos muy frecuentes; BoW cuenta ocurrencias directamente. Para clasificación de sentimientos, TF-IDF suele dar mejores resultados. |
| `sampling` | `"oversampling"` · `"undersampling"` · `null` | Estrategia para clases desbalanceadas. `null` desactiva el balanceo. |
| `impute_strategy_numeric` | `"mean"` · `"median"` · `"most_frequent"` | Cómo rellenar valores nulos en columnas numéricas. |
| `impute_strategy_categorical` | `"most_frequent"` · `"constant"` | Cómo rellenar valores nulos en columnas categóricas. |
| `scaling` | `"minmax"` · `"standard"` | Normalización de columnas numéricas. MinMax escala a [0,1]; Standard a media 0 y desviación 1. |

**Nota sobre `sampling`:** el balanceo se aplica **dentro del pipeline**, solo sobre el conjunto de train, nunca sobre dev ni test. Esto garantiza que no hay fuga de información.

---

## Sección `train`

Define los modelos a entrenar y sus hiperparámetros.

```json
"train": {
    "cpu": 6,
    "scoring": "f1_macro",
    "cv_folds": 5,
    "modelos": [ ... ]
}
```

| Campo | Descripción |
|-------|-------------|
| `cpu` | Número de núcleos para GridSearchCV. `-1` usa todos los disponibles. |
| `scoring` | Métrica que GridSearchCV optimiza. Recomendado `"f1_macro"` para clases desbalanceadas. |
| `cv_folds` | Número de folds en la validación cruzada interna. |

### Configuración de modelos

Cada entrada en `modelos` sigue este esquema:

```json
{
    "<nombre_modelo>": true,
    "modelo_output": "modelos/<nombre>_BestModel.pkl",
    "parametros": {
        "<nombre_parametro>": <valor>
    }
}
```

Cambia el valor a `false` para desactivar un modelo sin borrarlo:

```json
{ "random_forest": false, ... }   ← se ignora completamente
```

### Formato de hiperparámetros

**Rango automático** (recomendado): define `min`, `max` y `step`, y el script genera la lista automáticamente.

```json
"clasificador__n_neighbors": { "min": 1, "max": 15, "step": 2 }
// Equivale a: [1, 3, 5, 7, 9, 11, 13, 15]
```

**Lista explícita**: útil para valores no numéricos o rangos irregulares.

```json
"clasificador__weights": ["uniform", "distance"]
```

Ambos formatos pueden mezclarse en el mismo modelo.

### Modelos disponibles

#### KNN

```json
{
    "knn": true,
    "modelo_output": "modelos/knn_BestModel.pkl",
    "parametros": {
        "clasificador__n_neighbors": { "min": 1, "max": 15, "step": 2 },
        "clasificador__weights": ["uniform", "distance"],
        "clasificador__metric": ["euclidean", "manhattan"]
    }
}
```

#### Decision Tree

```json
{
    "decision_tree": true,
    "modelo_output": "modelos/decision_tree_BestModel.pkl",
    "parametros": {
        "clasificador__criterion": ["gini", "entropy"],
        "clasificador__max_depth": { "min": 5, "max": 20, "step": 5 },
        "clasificador__min_samples_split": { "min": 2, "max": 10, "step": 2 },
        "clasificador__min_samples_leaf": { "min": 1, "max": 5, "step": 1 }
    }
}
```

#### Random Forest

```json
{
    "random_forest": true,
    "modelo_output": "modelos/random_forest_BestModel.pkl",
    "parametros": {
        "clasificador__n_estimators": { "min": 50, "max": 250, "step": 50 },
        "clasificador__max_depth": [null, 10, 20],
        "clasificador__min_samples_split": { "min": 2, "max": 10, "step": 2 },
        "clasificador__min_samples_leaf": { "min": 1, "max": 5, "step": 1 },
        "clasificador__bootstrap": [true, false]
    }
}
```

#### Multinomial Naive Bayes

Recomendado para texto con TF-IDF. No funciona con valores negativos, por lo que requiere `scaling: null` o `scaling: "minmax"`.

```json
{
    "multinomial_nb": true,
    "modelo_output": "modelos/multinomial_nb_BestModel.pkl",
    "parametros": {
        "clasificador__alpha": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
        "clasificador__fit_prior": [true, false]
    }
}
```

#### Categorical Naive Bayes

Requiere que las features numéricas estén discretizadas (el pipeline lo hace automáticamente con `KBinsDiscretizer`).

```json
{
    "categorical_nb": true,
    "modelo_output": "modelos/categorical_nb_BestModel.pkl",
    "parametros": {
        "preprocessor__num__n_bins": [3, 5, 8],
        "preprocessor__num__strategy": ["uniform", "quantile"],
        "clasificador__alpha": [0.1, 0.5, 1.0, 2.0],
        "clasificador__min_categories": [100]
    }
}
```

---

## Sección `test`

Define dónde se guardan los resultados de evaluación.

```json
"test": {
    "metricas_output": "resultados/metricas_modelos.csv",
    "predicciones_output": "resultados/predicciones_test.csv"
}
```

| Campo | Descripción |
|-------|-------------|
| `metricas_output` | CSV con accuracy, F1 macro/micro/weighted, precision y recall de cada modelo. |
| `predicciones_output` | CSV con las predicciones de cada modelo sobre el conjunto de test, junto con los datos originales. |

---

## Sección `clustering`

Configura el análisis de tópicos LDA ejecutado por `src/analysis/clustering.py`.

```json
"clustering": {
    "name": "AppleMusic",
    "passes": 6,
    "coherence_metric": "c_v",
    "output_dir": "resultados/clustering",
    "text_col": "review",
    "num_topics_range": {
        "min": 2,
        "max_positivo": 7,
        "max_negativo": 7
    },
    "data": {
        "train_dev": "data/AppleMusic.csv",
        "test": "data/test.csv"
    }
}
```

| Campo | Descripción |
|-------|-------------|
| `name` | Nombre de la plataforma. Se usa para nombrar las carpetas de salida (`resultados/clustering_AppleMusic/`). Cambia a `"Spotify"`, `"SoundCloud"` o `"Tidal"` para cada plataforma. |
| `passes` | Número de pasadas del algoritmo LDA sobre el corpus. Más pasadas = convergencia mejor pero más lento. |
| `coherence_metric` | Métrica para evaluar la calidad de los tópicos. `"c_v"` es la más estable en la práctica. |
| `output_dir` | Carpeta raíz donde se guardan los resultados. El script añade automáticamente `_<name>` al final. |
| `text_col` | Nombre de la columna de texto en el CSV de clustering. |
| `num_topics_range.min` | Número mínimo de tópicos a probar. |
| `num_topics_range.max_positivo` | Número máximo de tópicos para las reseñas positivas. |
| `num_topics_range.max_negativo` | Número máximo de tópicos para las reseñas negativas. |
| `data.train_dev` | CSV específico de la plataforma (ej. `"data/AppleMusic.csv"`). |

Para ejecutar el clustering de las 4 plataformas, cambia `name` y `data.train_dev`, y ejecuta `python -m src.analysis.clustering` una vez por plataforma.

---

## Sección `generative`

```json
"generative": {
    "n_generations": 200,
    "eval_dev_limit": 150,
    "eval_test_limit": null    
}
```

| Campo             | Descripción                                                                                                                                                                                                                                                       |
|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `n_generations`   | Número de instancias que se quieren generar con Ollama para hacer oversampling, en `balancear_con_ia.py`.                                                                                                                                                         |
| `eval_dev_limit`  | Número de reseñas del conjunto `dev` que se evaluarán para decidir cuál es el mejor prompt en `generativo.py`. Si se define como null, se evaluará todo el conjunto dev. Un número bajo (ej. 20) acelera las pruebas, aunque no obtendremos un resultado fidedigno. |
| `eval_test_limit` | Número de reseñas del conjunto `test` que se usarán en la evaluación final de `test_generativo.py`. Si se define como null (recomendado para la entrega final), se evalúa todo el test.                                                                                                                                                       |

---

## Ejemplos de configuraciones completas

### Configuración mínima para empezar rápido

```json
{
    "general": {
        "random_state": 42,
        "column": "sentiment",
        "text_features": ["review"],
        "drop_features": [],
        "language": "english",
        "data": {
            "train_dev": "data/train.csv",
            "dev_file": null,
            "test": "data/test.csv"
        }
    },
    "preprocessing": {
        "test_size": 0.25,
        "text_process": "tf_idf",
        "sampling": null,
        "impute_strategy_numeric": "mean",
        "impute_strategy_categorical": "most_frequent",
        "scaling": "minmax"
    },
    "train": {
        "cpu": -1,
        "scoring": "f1_macro",
        "cv_folds": 3,
        "modelos": [
            {
                "knn": true,
                "modelo_output": "modelos/knn_BestModel.pkl",
                "parametros": {
                    "clasificador__n_neighbors": { "min": 1, "max": 9, "step": 2 }
                }
            }
        ]
    },
    "test": {
        "metricas_output": "resultados/metricas_modelos.csv",
        "predicciones_output": "resultados/predicciones_test.csv"
    }
}
```

### Cómo añadir un modelo nuevo (Logistic Regression)

1. Añade la entrada en `config.json`:

```json
{
    "logistic_regression": true,
    "modelo_output": "modelos/logistic_BestModel.pkl",
    "parametros": {
        "clasificador__C": { "min": 1, "max": 100, "step": 10 },
        "clasificador__penalty": ["l1", "l2"]
    }
}
```

2. Añade el caso en `src/models/train.py`, dentro de la función `crear_pipeline`:

```python
elif modelo_nombre == 'logistic_regression':
    from sklearn.linear_model import LogisticRegression
    modelo = LogisticRegression(solver='liblinear', random_state=random_state)
```