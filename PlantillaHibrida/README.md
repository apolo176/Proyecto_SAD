# 🚀 Plantilla Híbrida de Machine Learning

**La mejor arquitectura para proyectos de clasificación en equipo**

Combina lo mejor de dos mundos:
- ✅ **Arquitectura modular** (fácil trabajo en equipo)
- ✅ **Validación matemática robusta** (sin data leakage)
- ✅ **OneHotEncoder correcto** (matemáticamente riguroso)
- ✅ **Pipelines de sklearn** (prevención automática de errores)
- ✅ **Automatización de hiperparámetros** (min/max/step)

---

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Instalación](#-instalación)
- [Uso Rápido](#-uso-rápido)
- [Configuración](#️-configuración)
- [Arquitectura Técnica](#-arquitectura-técnica)
- [Ventajas Técnicas](#-ventajas-técnicas)
- [Trabajo en Equipo](#-trabajo-en-equipo)
- [FAQ](#-faq)

---

## ✨ Características

### Corrección Matemática
- ✅ **División train/dev ANTES del preprocesamiento** → Evita data leakage
- ✅ **OneHotEncoder para categóricas** → Distancias euclidianas correctas
- ✅ **Pipelines de sklearn** → Flujo de datos garantizado
- ✅ **GridSearchCV con CV interna** → Validación cruzada dentro de train

### Facilidad de Uso
- 📦 **Arquitectura modular** → train.py, test.py, funciones.py
- 🎯 **Batch training** → Entrena múltiples modelos de una vez
- ⚙️ **Configuración JSON** → Todo controlado desde un archivo
- 📊 **Métricas automáticas** → Comparación de modelos facilitada

### Procesamiento Avanzado
- 🇪🇸 **Stopwords en español** → NLP optimizado para el idioma
- 📝 **TF-IDF y BoW** → Procesamiento de texto configurable
- ⚖️ **Oversampling/Undersampling** → Balanceo de clases integrado
- 🔢 **Automatización de rangos** → `{"min": 1, "max": 15, "step": 2}`

---

## 📦 Instalación

### 1. Requisitos
- Python 3.12+
- pip

### 2. Crear entorno virtual (recomendado)

```bash
# Con conda
conda create -n hibrida python=3.12
conda activate hibrida

# O con venv
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Descargar recursos de NLTK (automático en primera ejecución)

Los recursos de NLTK se descargan automáticamente al ejecutar `train.py` por primera vez.

---

## 🚀 Uso Rápido

### Paso 1: Preparar tus datos

Coloca tus archivos CSV en una carpeta `data/`:
```
data/
  ├── train.csv      # Datos de entrenamiento
  └── test.csv       # Datos de prueba
```

### Paso 2: Configurar

Edita `config.json`:
```json
{
    "general": {
        "column": "tu_columna_objetivo",
        "text_features": ["columna_de_texto"],
        "data": {
            "train_dev": "data/train.csv",
            "test": "data/test.csv"
        }
    }
}
```

### Paso 3: Entrenar

```bash
python train.py
```

Esto:
1. Carga los datos
2. Divide train/dev (75%/25% por defecto)
3. Crea pipelines automáticos
4. Entrena todos los modelos activos
5. Guarda los mejores modelos en `modelos/`

### Paso 4: Evaluar

```bash
python test.py
```

Esto:
1. Carga los modelos entrenados
2. Evalúa en el conjunto de test
3. Genera métricas comparativas
4. Guarda resultados en `resultados/`

---

## ⚙️ Configuración

### Estructura del config.json

```json
{
    "general": {
        "random_state": 42,
        "column": "sentiment",                    // Columna a predecir
        "text_features": ["review_text"],         // Columnas de texto
        "drop_features": [],                       // Columnas a eliminar
        "language": "spanish",                     // Idioma para stopwords
        "data": {
            "train_dev": "data/train.csv",
            "test": "data/test.csv"
        }
    },
    
    "preprocessing": {
        "test_size": 0.25,                        // % para dev (0.25 = 25%)
        "text_process": "tf_idf",                 // "tf_idf" o "bow"
        "sampling": "oversampling",               // "oversampling", "undersampling", null
        "impute_strategy_numeric": "mean",        // "mean", "median", "most_frequent"
        "impute_strategy_categorical": "most_frequent",
        "scaling": "minmax"                       // "minmax" o "standard"
    },
    
    "train": {
        "cpu": -1,                                // -1 = usar todos los núcleos
        "scoring": "f1_macro",                    // Métrica para GridSearchCV
        "cv_folds": 5,                            // Número de folds en CV
        
        "modelos": [
            {
                "knn": true,                      // true = activar, false = desactivar
                "modelo_output": "modelos/knn_BestModel.pkl",
                "parametros": {
                    // Formato automático: min/max/step
                    "clasificador__n_neighbors": {"min": 1, "max": 15, "step": 2},
                    
                    // O formato manual: lista explícita
                    "clasificador__weights": ["uniform", "distance"]
                }
            }
        ]
    }
}
```

### Automatización de Hiperparámetros

**Formato min/max/step (recomendado):**
```json
"clasificador__n_neighbors": {"min": 1, "max": 15, "step": 2}
// Se expande a: [1, 3, 5, 7, 9, 11, 13, 15]
```

**Formato manual (tradicional):**
```json
"clasificador__n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15]
```

Ambos son equivalentes, pero el primero es más compacto y menos propenso a errores.

---

## 🏗️ Arquitectura Técnica

### Flujo de Datos (SIN DATA LEAKAGE)

```
1. Cargar datos completos
        ↓
2. División train/dev (train_test_split)
        ↓
3. Crear Pipeline (mirando SOLO train)
   ├─ ColumnTransformer
   │  ├─ Numéricas:    Imputer → Scaler
   │  ├─ Categóricas:  Imputer → OneHotEncoder  ✅ CORRECTO
   │  └─ Texto:        TextCleaner → TF-IDF
   ├─ Balanceo (opcional)
   └─ Modelo
        ↓
4. GridSearchCV (CV solo dentro de train)
        ↓
5. Evaluar en dev
        ↓
6. Guardar mejor modelo
```

**Punto crítico:** El pipeline se crea **DESPUÉS** de dividir los datos, usando **SOLO** el conjunto de train. Esto garantiza que dev nunca contamina el preprocesamiento.

### Diferencia vs Plantillas Originales

| Aspecto | PlantillaEder | PlantillaLiviu-Aitor | **PlantillaHíbrida** |
|---------|---------------|----------------------|----------------------|
| **Data Leakage** | ✅ Sin leakage | ❌ Con leakage | ✅ Sin leakage |
| **Codificación Categórica** | ✅ OneHot | ❌ Label | ✅ OneHot |
| **Modularidad** | ❌ Monolítica | ✅ Modular | ✅ Modular |
| **Batch Training** | ❌ No | ✅ Sí | ✅ Sí |
| **Automatización JSON** | ✅ min/max/step | ❌ Manual | ✅ min/max/step |
| **Stopwords Español** | ✅ Sí | ❌ Inglés | ✅ Sí |
| **MultinomialNB** | ✅ Incluido | ❌ No | ✅ Incluido |

---

## 🎯 Ventajas Técnicas

### 1. Sin Data Leakage

**Problema común (PlantillaLiviu-Aitor):**
```python
# ❌ INCORRECTO
data = load("train_dev.csv")          # train+dev mezclados
data = preprocesar(data)               # ← Dev contamina las estadísticas
train, dev = split(data)               # ← Demasiado tarde
```

**Solución (PlantillaHíbrida):**
```python
# ✅ CORRECTO
data = load("train_dev.csv")
train, dev = split(data)               # ← División PRIMERO
pipeline = crear_pipeline(train)       # ← Pipeline solo mira train
modelo = GridSearchCV(pipeline, ...)
modelo.fit(train, y_train)             # ← CV solo usa train
```

### 2. OneHotEncoder vs LabelEncoder

**Problema (LabelEncoder para features):**
```python
# ❌ INCORRECTO
Color: Rojo → 0, Verde → 1, Azul → 2

# KNN calcula:
distancia(Verde, Rojo) = |1-0| = 1
distancia(Azul, Rojo) = |2-0| = 2

# ⚠️ Implica que Verde está "más cerca" de Rojo que Azul
```

**Solución (OneHotEncoder):**
```python
# ✅ CORRECTO
Color: Rojo  → [1, 0, 0]
       Verde → [0, 1, 0]
       Azul  → [0, 0, 1]

# Todas las distancias son iguales: √2
# No hay orden artificial
```

### 3. Pipelines de Sklearn

**Ventaja:** El pipeline garantiza que el preprocesamiento se aplica correctamente en cada fold de CV.

```python
# El Pipeline AUTOMÁTICAMENTE:
# - Aprende estadísticas solo del training fold
# - Aplica transformaciones al validation fold
# - Repite esto para cada fold del CV
# - Nunca hay fuga de información
```

---

## 👥 Trabajo en Equipo

### División de Responsabilidades

Para un equipo de 4 personas:

#### **Persona 1: Líder de Clasificación**
- **Tarea:** Experimentar con diferentes modelos y preprocesos
- **Archivos:** `train.py`, `config.json`
- **Actividades:**
  - Probar diferentes combinaciones de modelos
  - Ajustar rangos de hiperparámetros
  - Experimentar con diferentes estrategias de balanceo
  - Analizar resultados de CV

#### **Persona 2: Líder Generativo**
- **Tarea:** Integración con modelos generativos (Ollama)
- **Archivos:** Crear `generativo.py` (nuevo)
- **Actividades:**
  - Prompt engineering para clasificación
  - Generación de datos sintéticos para balanceo
  - Comparación generativo vs tradicional

#### **Persona 3: Líder de Clustering**
- **Tarea:** Análisis de tópicos y palabras clave
- **Archivos:** Crear `clustering.py` (nuevo)
- **Actividades:**
  - Clustering de comentarios por sentimiento
  - Extracción de palabras significativas
  - Visualización de clusters

#### **Persona 4: Líder de Visualización (Tableau)**
- **Tarea:** Data storytelling
- **Archivos:** Tableau, scripts de exportación
- **Actividades:**
  - Importar resultados a Tableau
  - Crear historia contada con datos
  - Generar visualizaciones de clusters
  - Preparar dashboard interactivo

### Flujo de Trabajo Colaborativo

```
Semana 1: Setup inicial
├─ Todos: Instalar dependencias, probar train.py
└─ Líder Clasificación: Primera versión de config.json

Semana 2: Desarrollo paralelo
├─ Líder Clasificación: Experimentos con modelos
├─ Líder Generativo: Setup Ollama + primeros prompts
├─ Líder Clustering: Investigar librerías (gensim, sklearn)
└─ Líder Visualización: Diseñar estructura de Tableau

Semana 3: Integración
├─ Líder Clasificación: Mejor modelo tradicional
├─ Líder Generativo: Mejor prompt + balanceo generativo
├─ Líder Clustering: Clusters + palabras clave
└─ Líder Visualización: Primera versión de dashboard

Semana 4: Refinamiento
└─ Todos: Integrar, documentar, preparar presentación
```

---

## 📊 Outputs Generados

### Durante Entrenamiento

```
modelos/
├── knn_BestModel.pkl                    # Modelo entrenado
├── knn_BestModel_cv_results.csv         # Resultados de CV
├── decision_tree_BestModel.pkl
├── decision_tree_BestModel_cv_results.csv
├── random_forest_BestModel.pkl
├── random_forest_BestModel_cv_results.csv
├── multinomial_nb_BestModel.pkl
├── multinomial_nb_BestModel_cv_results.csv
└── label_encoder_y.pkl                  # Codificador del target
```

### Durante Evaluación

```
resultados/
├── metricas_modelos.csv                 # Comparación de todos los modelos
└── predicciones_test.csv                # Predicciones con datos originales
```

### Ejemplo: metricas_modelos.csv

| modelo | accuracy | f1_macro | f1_micro | f1_weighted | precision_macro | recall_macro |
|--------|----------|----------|----------|-------------|-----------------|--------------|
| random_forest | 0.8542 | 0.8401 | 0.8542 | 0.8523 | 0.8456 | 0.8398 |
| knn | 0.8123 | 0.7989 | 0.8123 | 0.8105 | 0.8045 | 0.8012 |
| multinomial_nb | 0.7856 | 0.7701 | 0.7856 | 0.7834 | 0.7723 | 0.7689 |

---

## ❓ FAQ

### ¿Por qué no hay process.py?

En la plantilla original de Liviu-Aitor, `process.py` causaba **data leakage** al preprocesar train y dev juntos. En la plantilla híbrida, el preprocesamiento está **dentro del pipeline**, garantizando que se aprende solo del train.

### ¿Puedo seguir usando LabelEncoder?

**Solo para la variable objetivo (y)**, no para las features. LabelEncoder es correcto para codificar el target (positivo/negativo/neutro → 0/1/2), pero **incorrecto** para features categóricas con algoritmos basados en distancia.

### ¿Qué modelos están incluidos?

Por defecto:
- KNN
- Decision Tree
- Random Forest
- MultinomialNB (óptimo para texto con TF-IDF)
- CategoricalNB (requiere discretización)

### ¿Cómo añado un modelo nuevo?

1. Edita `config.json`:
```json
{
    "logistic_regression": true,
    "modelo_output": "modelos/logistic_BestModel.pkl",
    "parametros": {
        "clasificador__C": {"min": 1, "max": 100, "step": 10},
        "clasificador__penalty": ["l1", "l2"]
    }
}
```

2. Edita `train.py` en la función `crear_pipeline`:
```python
elif modelo_nombre == 'logistic_regression':
    from sklearn.linear_model import LogisticRegression
    modelo = LogisticRegression(solver='liblinear', random_state=42)
```

### ¿Cómo cambio el idioma de stopwords?

En `config.json`:
```json
{
    "general": {
        "language": "english"  // o "french", "german", etc.
    }
}
```

### ¿Puedo desactivar el balanceo?

Sí, en `config.json`:
```json
{
    "preprocessing": {
        "sampling": null  // o "oversampling", "undersampling"
    }
}
```

### ¿Cómo interpreto los resultados de CV?

El archivo `*_cv_results.csv` contiene todas las combinaciones probadas. Columnas importantes:
- `mean_test_score`: Score promedio en CV
- `params`: Hiperparámetros de esa combinación
- `rank_test_score`: Ranking (1 = mejor)

---

## 🎓 Conceptos Clave

### Data Leakage

**Definición:** Cuando información del conjunto de validación/test contamina el entrenamiento.

**Consecuencia:** Métricas infladas que no reflejan el rendimiento real.

**Cómo evitarlo:** Dividir los datos ANTES de cualquier preprocesamiento.

### Pipeline de Sklearn

**Definición:** Secuencia de transformaciones + modelo empaquetada como un único objeto.

**Ventaja:** Garantiza que las transformaciones se aplican en el orden correcto en cada fold de CV.

### OneHotEncoder vs LabelEncoder

**LabelEncoder:**
- Convierte categorías a números (Rojo→0, Verde→1, Azul→2)
- ❌ Incorrecto para features con KNN, SVM, etc.
- ✅ Correcto para la variable objetivo (y)

**OneHotEncoder:**
- Convierte categorías a vectores binarios (Rojo→[1,0,0])
- ✅ Correcto para features categóricas
- No introduce orden artificial

---

## 📚 Referencias

- [Sklearn Pipelines](https://scikit-learn.org/stable/modules/compose.html)
- [Imbalanced-learn](https://imbalanced-learn.org/)
- [NLTK Stopwords](https://www.nltk.org/howto/corpus.html)

---

## 🤝 Contribuciones

Esta plantilla es el resultado de combinar lo mejor de:
- **PlantillaEder**: Pipelines robustos y procesamiento de texto avanzado
- **PlantillaLiviu-Aitor**: Arquitectura modular y batch training

**Mejoras adicionales:**
- Automatización de hiperparámetros (min/max/step)
- MultinomialNB para clasificación de texto
- Documentación exhaustiva
- Manejo de errores mejorado

---

## 📝 Licencia

Este proyecto es de uso educativo para la asignatura de Sistemas de Ayuda a la Decisión.

---

## 💡 Consejos Finales

1. **Comienza simple:** Activa solo un modelo para probar que todo funciona
2. **Incrementa gradualmente:** Añade más modelos cuando el flujo esté claro
3. **Documenta tus experimentos:** Usa nombres descriptivos en `modelo_output`
4. **Compara métricas:** El archivo `metricas_modelos.csv` es tu mejor amigo
5. **No temas experimentar:** Cambia parámetros, prueba cosas nuevas

**¡Éxito en tu proyecto!** 🚀
