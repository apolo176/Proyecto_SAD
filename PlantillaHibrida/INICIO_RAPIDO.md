# 🎉 PlantillaHíbrida - Guía de Inicio Rápido

## 📦 Contenido del Paquete

```
PlantillaHibrida/
├── 📘 README.md                      # Documentación completa
├── 📊 COMPARACION_TECNICA.md         # Análisis vs otras plantillas
├── ⚙️ config.json                    # Configuración del proyecto
├── 🔧 funciones.py                   # Utilidades compartidas
├── 🚂 train.py                       # Script de entrenamiento
├── 📈 test.py                        # Script de evaluación
├── 🎬 ejemplo_generar_datos.py       # Generador de datos de prueba
├── 📋 requirements.txt               # Dependencias
└── 🙈 .gitignore                     # Archivos a ignorar en Git
```

---

## 🚀 Instalación en 3 Pasos

### 1️⃣ Descomprimir
```bash
unzip PlantillaHibrida.zip
cd PlantillaHibrida
```

### 2️⃣ Crear entorno virtual
```bash
# Con conda (recomendado)
conda create -n hibrida python=3.12
conda activate hibrida

# O con venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3️⃣ Instalar dependencias
```bash
pip install -r requirements.txt
```

---

## ⚡ Prueba Rápida (5 minutos)

### Generar datos de ejemplo
```bash
python ejemplo_generar_datos.py
```
**Salida:**
- `data/train.csv` (1200 muestras)
- `data/test.csv` (300 muestras)

### Entrenar modelos
```bash
python train.py
```
**Salida:**
- `modelos/knn_BestModel.pkl`
- `modelos/decision_tree_BestModel.pkl`
- `modelos/random_forest_BestModel.pkl`
- `modelos/multinomial_nb_BestModel.pkl`

### Evaluar modelos
```bash
python test.py
```
**Salida:**
- `resultados/metricas_modelos.csv`
- `resultados/predicciones_test.csv`

---

## 🎯 Para tu Proyecto Real

### 1. Preparar tus datos

Formato requerido:
```csv
review_id,review_text,sentiment,genre,year,duration_min,imdb_rating
REV_0001,"La película fue excelente...",positivo,drama,2020,120,8.5
REV_0002,"Muy aburrida y lenta",negativo,terror,2019,95,4.2
```

**Columnas:**
- Una columna de **texto** (reviews, comentarios, etc.)
- Una columna **objetivo** (sentiment, clasificación, etc.)
- Otras columnas opcionales (numéricas, categóricas)

### 2. Configurar config.json

```json
{
    "general": {
        "column": "sentiment",              // ← Tu columna objetivo
        "text_features": ["review_text"],   // ← Tus columnas de texto
        "data": {
            "train_dev": "data/train.csv",  // ← Tus archivos
            "test": "data/test.csv"
        }
    }
}
```

### 3. Ejecutar
```bash
python train.py
python test.py
```

---

## 🏆 Ventajas vs Plantillas Originales

| Característica | Eder | Liviu-Aitor | **Híbrida** |
|----------------|------|-------------|-------------|
| Sin Data Leakage | ✅ | ❌ | ✅ |
| OneHotEncoder | ✅ | ❌ | ✅ |
| Modular | ❌ | ✅ | ✅ |
| Batch Training | ❌ | ✅ | ✅ |
| min/max/step | ✅ | ❌ | ✅ |
| Español | ✅ | ❌ | ✅ |
| MultinomialNB | ✅ | ❌ | ✅ |

**Resultado:** ✅ Todas las ventajas, ningún compromiso

---

## 📚 Documentación

### README.md (16 KB)
- Características detalladas
- Guía completa de uso
- Configuración avanzada
- Trabajo en equipo
- FAQ
- Conceptos técnicos

### COMPARACION_TECNICA.md (8 KB)
- Análisis línea por línea del código
- Comparación con plantillas originales
- Evidencia de data leakage
- Matriz de decisión
- Guía de migración

---

## 🎓 Para el Proyecto de SAD

### Entregables Cubiertos

✅ **Código de clasificación**
- `train.py` con pipelines robustos
- `test.py` con métricas completas
- `requirements.txt`
- `README.md` con instrucciones

✅ **Múltiples modelos**
- KNN ✅
- Decision Tree ✅
- Random Forest ✅
- MultinomialNB ✅

✅ **Balanceo de datos**
- Oversampling ✅
- Undersampling ✅
- Generativo (añadir aparte) 🔄

✅ **Métricas**
- F1 Macro ✅
- Accuracy ✅
- Precision/Recall ✅
- Matriz de confusión ✅

### Pendientes (fuera de esta plantilla)
- 🔄 Clustering (crear `clustering.py`)
- 🔄 Generativo (crear `generativo.py`)
- 🔄 Tableau (usar `resultados/metricas_modelos.csv`)
- 🔄 Poster (usar métricas generadas)

---

## 💡 Consejos Pro

### 1. Empieza simple
```json
{
    "modelos": [
        {"knn": true, ...},           // ← Solo uno primero
        {"decision_tree": false, ...},
        {"random_forest": false, ...}
    ]
}
```

### 2. Incrementa gradualmente
Una vez que funcione con un modelo, activa los demás.

### 3. Usa las métricas de CV
El archivo `*_cv_results.csv` te dice qué combinaciones probó GridSearchCV.

### 4. Compara modelos
`resultados/metricas_modelos.csv` tiene la comparación automática.

### 5. Experimenta sin miedo
Todos los parámetros están en `config.json`. Modifica y prueba.

---

## 🐛 Solución de Problemas

### Error: "No module named 'imbalanced-learn'"
```bash
pip install imbalanced-learn
```

### Error: "No se encontró el archivo data/train.csv"
```bash
python ejemplo_generar_datos.py  # Genera datos de prueba
```

### Error: NLTK stopwords
Los recursos se descargan automáticamente. Si falla:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### Warning: "FutureWarning"
Son avisos de versiones futuras de sklearn. Ignora o actualiza:
```bash
pip install --upgrade scikit-learn
```

---

## 📞 Soporte

### Documentación incluida
- `README.md` → Guía completa
- `COMPARACION_TECNICA.md` → Análisis técnico
- Comentarios en el código → Cada función documentada

### Recursos externos
- [Sklearn Pipelines](https://scikit-learn.org/stable/modules/compose.html)
- [Imbalanced-learn](https://imbalanced-learn.org/)
- [NLTK](https://www.nltk.org/)

---

## ✅ Checklist Final

Antes de entregar tu proyecto, verifica:

- [ ] Los datos están en `data/train.csv` y `data/test.csv`
- [ ] `config.json` apunta a tus columnas correctas
- [ ] `python train.py` ejecuta sin errores
- [ ] `python test.py` genera métricas
- [ ] Los modelos están en `modelos/*.pkl`
- [ ] Las métricas están en `resultados/metricas_modelos.csv`
- [ ] El `README.md` está actualizado con tu información
- [ ] Has probado al menos 2-3 modelos diferentes
- [ ] Las métricas son razonables (F1 > 0.5)

---

## 🎉 ¡Listo!

Ya tienes todo lo necesario para:
1. ✅ Entrenar modelos robustos
2. ✅ Evitar data leakage
3. ✅ Comparar algoritmos
4. ✅ Trabajar en equipo
5. ✅ Aprobar el proyecto con nota alta

**¡Éxito en tu proyecto de SAD!** 🚀

---

_PlantillaHíbrida v1.0 - Abril 2026_
_Combina lo mejor de PlantillaEder y PlantillaLiviu-Aitor_
