# Inicio Rápido — Proyecto SAD

Guía para levantar el proyecto, entrenar modelos y generar resultados en el menor tiempo posible.

---

## Requisitos previos

- Python 3.12+
- [Ollama](https://ollama.com/) instalado y corriendo (solo para los módulos generativo y balanceo con IA)

---

## 1. Instalación

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

Los recursos de NLTK (stopwords, tokenizador) se descargan automáticamente la primera vez que ejecutas cualquier script.

---

## 2. Estructura de carpetas esperada

Antes de ejecutar nada, coloca tus datos en la carpeta `data/`:

```
Proyecto_SAD/
├── data/
│   ├── train.csv       ← datos de entrenamiento + validación
│   └── test.csv        ← datos de prueba final
```

Formato mínimo esperado en los CSV:

```
review,sentiment
"This app is great for discovering music",positivo
"Crashes every time I open it",negativo
"It's okay, nothing special",neutro
```

La columna de texto y la columna objetivo se configuran en `config.json` (ver documento de Configuración).

---

## 3. Prueba rápida con datos sintéticos

Si aún no tienes datos reales, genera un dataset de ejemplo:

```bash
python examples/ejemplo_generar_datos.py
```

Esto crea `data/train.csv` (1200 muestras) y `data/test.csv` (300 muestras) con reviews sintéticas de películas clasificadas como positivo / negativo / neutro. Sirve para verificar que todo el pipeline funciona antes de usar tus datos reales.

---

## 4. Flujo completo paso a paso

Todos los comandos se ejecutan desde la **raíz del proyecto** (`Proyecto_SAD/`).

### Paso 1 — Preparar los datos

Convierte las puntuaciones numéricas (ej. estrellas 1-5) a etiquetas de sentimiento:

```bash
python -m src.data.score_to_sentiment
```

Genera `data/train.csv` y `data/test.csv` con la columna `sentiment` lista para entrenar.

### Paso 2 — (Opcional) Balancear clases con IA

Si tus clases están desbalanceadas y quieres usar Ollama para generar muestras sintéticas adicionales:

```bash
python -m src.data.balancear_con_ia
```

Requiere que Ollama esté activo (`ollama serve`).

### Paso 3 — Entrenar modelos clásicos

```bash
python -m src.models.train
```

Lee `config.json`, entrena todos los modelos marcados como `true` y guarda los resultados en `modelos/`:

```
modelos/
├── knn_BestModel.pkl
├── knn_BestModel_cv_results.csv
├── random_forest_BestModel.pkl
├── random_forest_BestModel_cv_results.csv
└── label_encoder_y.pkl
```

### Paso 4 — Evaluar modelos clásicos

```bash
python -m src.models.test
```

Carga los modelos entrenados, los evalúa en el conjunto de test y guarda en `resultados/`:

```
resultados/
├── metricas_modelos.csv        ← comparativa de todos los modelos
└── predicciones_test.csv       ← predicciones con los datos originales
```

### Paso 5 — Análisis de tópicos (LDA)

```bash
python -m src.analysis.clustering
```

Ejecuta modelado de tópicos LDA con Gensim sobre los datos de la plataforma configurada en `config.json > clustering > name`. Guarda gráficas de coherencia y CSVs para Tableau en `resultados/clustering_<Plataforma>/`.

### Paso 6 — Modelo generativo (Ollama)

```bash
python -m src.analysis.generativo
```

Prueba distintas estrategias de prompt (zero-shot, one-shot, few-shot) y guarda los resultados de cada prompt en `resultados/`.

Requiere Ollama activo.

### Paso 7 — Evaluar modelo generativo

```bash
python -m src.models.test_generativo
```

Carga el mejor prompt encontrado en el paso anterior y lo evalúa sobre el conjunto de test.

---

## 5. Activar o desactivar modelos

En `config.json`, dentro de `train > modelos`, cambia `true` / `false`:

```json
"modelos": [
    { "knn": true,           ... },   ← se entrena
    { "decision_tree": false, ... },  ← se salta
    { "random_forest": true,  ... }   ← se entrena
]
```

Empieza con un único modelo activo para verificar que el pipeline funciona, y luego activa los demás.

---

## 6. Solución de problemas frecuentes

**`ModuleNotFoundError: No module named 'src'`**
Asegúrate de ejecutar los comandos desde la raíz del proyecto, no desde dentro de `src/`.

**`FileNotFoundError: data/train.csv`**
Ejecuta primero `python examples/ejemplo_generar_datos.py` para generar datos de prueba, o coloca tus propios CSVs en `data/`.

**Error de NLTK (stopwords / punkt)**
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

**Ollama no responde**
Arranca el servidor con `ollama serve` en otra terminal y confirma que el modelo requerido está descargado (`ollama pull llama3`).

**`FutureWarning` de scikit-learn**
Son avisos informativos, no errores. No afectan al resultado.

---

## 7. Checklist antes de entregar

- [ ] `data/train.csv` y `data/test.csv` existen y tienen el formato correcto
- [ ] `config.json` apunta a las columnas correctas (`column`, `text_features`)
- [ ] `python -m src.models.train` termina sin errores
- [ ] `python -m src.models.test` genera `resultados/metricas_modelos.csv`
- [ ] Al menos 2 modelos clásicos entrenados y evaluados
- [ ] Clustering ejecutado para las 4 plataformas
- [ ] `resultados/` contiene los CSVs para Tableau