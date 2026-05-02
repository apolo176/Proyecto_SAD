# 🚀 Proyecto SAD Grupal

**Análisis de sentimientos de reseñas de plataformas musicales (Spotify, Apple Music, Tidal, SoundCloud)**

---

## 📁 Estructura del Proyecto

```
Proyecto_SAD/
│
├── src/                         # Código fuente principal
│   ├── utils/
│   │   └── funciones.py         # Funciones auxiliares compartidas
│   ├── data/
│   │   ├── score_to_sentiment.py  # Conversión de puntuaciones a sentimiento
│   │   └── balancear_con_ia.py    # Balanceo de clases con IA (Ollama)
│   ├── models/
│   │   ├── train.py               # Entrenamiento de modelos clásicos
│   │   └── test.py                # Evaluación de modelos entrenados
│   │
│   └── analysis/
│       ├── clustering.py          # Modelado de tópicos con LDA (Gensim)
│       ├── generativo.py          # Clasificación con Ollama (prompt engineering)
│       └── test_generativo.py     # Evaluación del modelo generativo
│
├── data/                        # Datasets (no versionados)
├── modelos/                     # Modelos entrenados .pkl (auto-generados)
├── resultados/                  # Outputs generados
│   ├── clustering_AppleMusic/
│   ├── clustering_SoundCloud/
│   ├── clustering_Spotify/
│   └── clustering_Tidal/
│
├── docs/                        # Documentación extra
│   ├── INICIO_RAPIDO.md
│   └── COMPARACION_TECNICA.md
│   └── CLUSTERING.md
│
├── examples/
│   └── ejemplo_generar_datos.py # Script de prueba con datos sintéticos
│
├── config.json                  # Configuración central del proyecto
├── requirements.txt
└── README.md
```

---

## 📦 Instalación

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Uso

Todos los scripts se ejecutan desde la **raíz del proyecto**:

```bash
# 1. Preparar datos
python -m src.data.score_to_sentiment

# 2. (Opcional) Balancear con IA
python -m src.data.balancear_con_ia

# 3. Entrenar modelos clásicos
python -m src.models.train

# 4. Evaluar modelos clásicos
python -m src.models.test

# 5. Análisis de tópicos LDA
python -m src.analysis.clustering

# 6. Modelo generativo
python -m src.analysis.generativo

# 7. Evaluar modelo generativo
python -m src.analysis.test_generativo
```

---

## 👥 División de trabajo

| Persona | Responsabilidad | Ficheros clave                                                                                  |
|---------|----------------|-------------------------------------------------------------------------------------------------|
| Líder Clasificación | Modelos clásicos | `src/models/train.py`, `src/models/test.py`, `config.json`                                      |
| Líder Generativo | Ollama + prompts | `src/analysis/generativo.py`, `src/analysis/test_generativo.py`, `src/data/balancear_con_ia.py` |
| Líder Clustering | Análisis LDA | `src/analysis/clustering.py`                                                                    |
| Líder Visualización | Tableau | `resultados/*.csv`                                                                              |

---

Consulta `docs/INICIO_RAPIDO.md` para detalles de configuración y `docs/COMPARACION_TECNICA.md` para comparativas técnicas.