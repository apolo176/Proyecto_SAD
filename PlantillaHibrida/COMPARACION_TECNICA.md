# 📊 Comparación Técnica Detallada

## PlantillaEder vs PlantillaLiviu-Aitor vs PlantillaHíbrida

---

## 🎯 Resumen Ejecutivo

| Criterio | PlantillaEder | PlantillaLiviu-Aitor | **PlantillaHíbrida** |
|----------|---------------|----------------------|----------------------|
| **Validez Matemática** | ✅ Excelente | ❌ Comprometida | ✅ Excelente |
| **Facilidad de Uso** | ⚠️ Media | ✅ Excelente | ✅ Excelente |
| **Trabajo en Equipo** | ⚠️ Limitado | ✅ Excelente | ✅ Excelente |
| **Recomendación** | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 🔬 Análisis Detallado

### 1. PREVENCIÓN DE DATA LEAKAGE

#### PlantillaEder ✅
```python
# Flujo correcto
data = load_data("train.csv")
X_train, X_dev, y_train, y_dev = train_test_split(data)  # ← División PRIMERO
pipeline = crear_pipeline(X_train)                        # ← Pipeline mira solo train
gs = GridSearchCV(pipeline, params, cv=5)
gs.fit(X_train, y_train)                                 # ← CV solo usa train
```
**Resultado:** ✅ Sin leakage

#### PlantillaLiviu-Aitor ❌
```python
# Flujo con leakage
data = load_data("train_dev.csv")      # train+dev juntos
data = process.py(data)                # ← Calcula estadísticas de TODO
                                       # ← Dev contamina medias, TF-IDF, etc.
train, dev = train_test_split(data)    # ← Demasiado tarde
```
**Resultado:** ❌ Data leakage severo → métricas infladas

#### PlantillaHíbrida ✅
```python
# Flujo correcto + modular
data = load_data("train.csv")
X_train, X_dev, y_train, y_dev = train_test_split(data)  # ← División PRIMERO
pipeline = crear_pipeline(X_train, config)                # ← Pipeline mira solo train
gs = GridSearchCV(pipeline, params, cv=5)
gs.fit(X_train, y_train)                                 # ← CV solo usa train
```
**Resultado:** ✅ Sin leakage + modular

---

### 2. CODIFICACIÓN DE VARIABLES CATEGÓRICAS

#### PlantillaEder ✅
```python
OneHotEncoder(handle_unknown='ignore')

# Color → OneHot
Rojo  → [1, 0, 0]
Verde → [0, 1, 0]
Azul  → [0, 0, 1]

# Distancias euclidianas
dist(Rojo, Verde) = √2
dist(Rojo, Azul)  = √2
dist(Verde, Azul) = √2
# ✅ Todas iguales (correcto)
```

#### PlantillaLiviu-Aitor ❌
```python
LabelEncoder()

# Color → Label
Rojo  → 0
Verde → 1
Azul  → 2

# Distancias euclidianas
dist(Rojo, Verde) = 1
dist(Rojo, Azul)  = 2
dist(Verde, Azul) = 1
# ❌ Verde "más cerca" de Rojo (incorrecto)
```

#### PlantillaHíbrida ✅
```python
OneHotEncoder(handle_unknown='ignore')

# Igual que PlantillaEder
# ✅ Matemáticamente correcto
```

---

### 3. ARQUITECTURA DE SOFTWARE

#### PlantillaEder ⚠️
```
clasificador.py (503 líneas)
├─ parse_args()
├─ load_data()
├─ preparar_y_dividir()
├─ crear_pipeline()
├─ ejecutar_modelo()
├─ save_model()
└─ main()

clasificador.json
requirements.txt
README.md
```
**Pros:**
- Todo en un lugar
- Fácil de entender el flujo

**Contras:**
- Difícil de modificar
- Difícil de dividir trabajo
- Menos reutilizable

#### PlantillaLiviu-Aitor ✅
```
process.py     → Preprocesamiento
train.py       → Entrenamiento
test.py        → Evaluación
funciones.py   → Utilidades

config.json
requirements.txt
README.md
```
**Pros:**
- Muy modular
- Fácil dividir trabajo
- Batch training

**Contras:**
- process.py causa leakage
- LabelEncoder incorrecto

#### PlantillaHíbrida ✅
```
train.py       → Entrenamiento (con pipelines)
test.py        → Evaluación
funciones.py   → Utilidades

config.json
requirements.txt
README.md
```
**Pros:**
- Modular (sin process.py problemático)
- Pipelines robustos dentro de train.py
- Batch training
- Reutilizable

**Contras:**
- Ninguno significativo

---

### 4. CONFIGURACIÓN JSON

#### PlantillaEder ✅
```json
{
  "kNN": {
    "clasificador__n_neighbors": {"min": 1, "max": 15, "step": 2}
  }
}
```
**Resultado:** Se expande automáticamente a `[1, 3, 5, 7, 9, 11, 13, 15]`

**Pros:**
- Compacto
- Menos errores
- Fácil de modificar

#### PlantillaLiviu-Aitor ❌
```json
{
  "parametros": {
    "n_neighbors": [1,2,3,4,5,6,7,8,9,10]
  }
}
```
**Pros:**
- Control total

**Contras:**
- Verboso
- Propenso a errores
- Difícil de leer

#### PlantillaHíbrida ✅
```json
{
  "parametros": {
    "clasificador__n_neighbors": {"min": 1, "max": 15, "step": 2}
  }
}
```
**Resultado:** Igual que Eder + estructura de Liviu-Aitor

---

### 5. PROCESAMIENTO DE TEXTO

#### PlantillaEder ✅
```python
class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, language='spanish'):
        self.stop_words = set(stopwords.words(self.language))
    
    def transform(self, X):
        # Limpieza, tokenización, stopwords
        return X.apply(self._clean_text)
```
**Ventajas:**
- Stopwords configurables por idioma
- Integrado en el pipeline
- Limpieza automática

#### PlantillaLiviu-Aitor ⚠️
```python
stop_words = set(stopwords.words('english'))  # ← Hardcodeado
stemmer = PorterStemmer()                     # ← Solo inglés
```
**Problema:** Hardcodeado para inglés

#### PlantillaHíbrida ✅
```python
class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, language='spanish'):  # ← Configurable desde JSON
        self.stop_words = set(stopwords.words(self.language))
```
**Ventajas:**
- Igual que Eder
- Configurable desde JSON

---

### 6. MODELOS INCLUIDOS

#### PlantillaEder ✅
- KNN
- Decision Tree
- Random Forest
- GaussianNB
- **MultinomialNB** ← Óptimo para texto

#### PlantillaLiviu-Aitor ⚠️
- KNN
- Decision Tree
- Random Forest
- CategoricalNB

**Falta:** MultinomialNB (el mejor para TF-IDF)

#### PlantillaHíbrida ✅
- KNN
- Decision Tree
- Random Forest
- **MultinomialNB**
- CategoricalNB

**Incluye:** Todos los modelos relevantes

---

## 📊 Matriz de Decisión

| Situación | PlantillaEder | PlantillaLiviu-Aitor | PlantillaHíbrida |
|-----------|---------------|----------------------|------------------|
| **Proyecto individual** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Equipo 2-3 personas** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Equipo 4+ personas** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Evaluación académica** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Producción real** | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| **Aprendizaje ML** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 🎓 Conclusiones

### PlantillaEder
**Cuándo usarla:**
- Proyecto individual
- Quieres máxima corrección matemática
- No te importa la modularidad

**Cuándo NO usarla:**
- Trabajo en equipo grande
- Necesitas experimentar rápidamente

### PlantillaLiviu-Aitor
**Cuándo usarla:**
- Solo como referencia de arquitectura modular
- **NUNCA sin corregir el data leakage**

**Cuándo NO usarla:**
- Evaluación académica
- Producción
- Sin modificaciones previas

### PlantillaHíbrida ⭐
**Cuándo usarla:**
- **SIEMPRE que sea posible**
- Trabajo en equipo
- Evaluación académica
- Producción
- Aprendizaje

**Cuándo NO usarla:**
- Nunca (es la mejor opción en todos los casos)

---

## 💡 Recomendación Final

```
Si tienes tiempo: PlantillaHíbrida
Si no tienes tiempo: PlantillaHíbrida
Si trabajas solo: PlantillaHíbrida
Si trabajas en equipo: PlantillaHíbrida

Respuesta: SIEMPRE PlantillaHíbrida
```

**Razón:** Combina lo mejor de ambos mundos sin ningún compromiso.

---

## 🚀 Migración desde otras plantillas

### Desde PlantillaEder
1. Mantener `clasificador.py` como está
2. Opcionalmente, separar en módulos
3. Listo

### Desde PlantillaLiviu-Aitor
1. ❌ **ELIMINAR** `process.py` completamente
2. ✅ **REEMPLAZAR** train.py con la versión híbrida
3. ✅ **CAMBIAR** LabelEncoder → OneHotEncoder
4. ✅ **VERIFICAR** que división ocurre antes del preprocesamiento

---

## 📚 Referencias Técnicas

- **Data Leakage:** https://machinelearningmastery.com/data-leakage-machine-learning/
- **Sklearn Pipelines:** https://scikit-learn.org/stable/modules/compose.html
- **OneHot vs Label:** https://towardsdatascience.com/one-hot-encoding-vs-label-encoding-70f7d1f6a72d

---

Documento creado por PlantillaHíbrida © 2026
