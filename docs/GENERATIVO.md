# IA Generativa — Qué hace, por qué y cómo

---

## El problema que resuelve

En este proyecto nos enfrentamos a dos grandes retos relacionados con el análisis de sentimientos de las reseñas de apps de música:

### 1. Clasificación sin entrenamiento previo
Los algoritmos clásicos (KNN, Random Forest, etc.) necesitan miles de ejemplos etiquetados para aprender a clasificar si una reseña es positiva, negativa o neutra. ¿Podría un **Modelo de Lenguaje Grande (LLM)** entender el sarcasmo, el contexto y el sentimiento directamente *"de fábrica"*, superando a los modelos clásicos?

### 2. Desbalanceo de clases
Algunas clases (como la neutra o negativa) suelen tener muchos menos ejemplos. Las técnicas tradicionales de oversampling (como duplicar filas) no aportan vocabulario nuevo y pueden causar sobreajuste.

> **La pregunta central:** ¿Podemos usar la IA generativa para clasificar sentimientos mediante estrategias de *Prompt Engineering* y para generar reseñas sintéticas que enriquezcan nuestro dataset?

---

## La solución: Ollama y Prompt Engineering

Utilizamos **Ollama**, una herramienta que permite ejecutar modelos de lenguaje (específicamente **Llama 3**) de manera local. Esto garantiza:

- **Privacidad** de los datos
- **Coste cero** de API
- **Control total** sobre el modelo

### 1. Clasificación mediante Prompts

En lugar de entrenar pesos matemáticos, "programamos" al modelo con instrucciones precisas. Probamos la evolución de tres estrategias:

| Estrategia | Descripción |
|---|---|
| **Zero-shot** | Clasificar la reseña sin proporcionar ningún ejemplo previo |
| **One-shot** | Proporcionar un único ejemplo resuelto por clase para que entienda el formato esperado |
| **Few-shot** | Dar múltiples ejemplos que cubran casos dudosos (sarcasmo, quejas técnicas vs. de catálogo) para refinar la precisión |

### 2. Oversampling Generativo (Data Augmentation)

Para el problema del desbalanceo, usamos el LLM como un *"redactor"*. Le pasamos reseñas reales de la clase minoritaria y le pedimos que genere **paráfrasis**: el mismo sentimiento, pero con diferentes palabras y estructuras. Así logramos un oversampling con **riqueza semántica real**.

---

## El pipeline completo, paso a paso

El flujo de trabajo está diseñado en **tres fases independientes pero conectadas**:

### Fase 1: Búsqueda del mejor prompt — `generativo.py`

No asumimos qué prompt funciona mejor; lo demostramos con datos.

1. **División de datos:** Se separa el dataset crudo en *Train* y *Dev* (validación) para evitar contaminación.
2. **Muestreo (opcional):** Se lee el parámetro `eval_dev_limit` de `config.json`. Para pruebas rápidas, se evalúa solo una muestra estratificada del Dev.
3. **Inferencia estructurada:** Los textos se envían al modelo Llama 3. Se usa un truco clave: restringir la salida a **1 solo token** (`num_predict: 1`), obligando al modelo a responder únicamente `"A"`, `"B"` o `"C"` (Positivo, Neutro, Negativo) sin explicaciones. Esto acelera masivamente la inferencia.
4. **Benchmarking:** Se calcula el **F1-Macro** de cada prompt sobre las predicciones válidas.
5. **Selección:** El script elige automáticamente el prompt con el F1-Macro más alto, lo etiqueta como `[MEJOR RESULTADO]` y guarda todo en `resultados_generativos.csv`.

---

### Fase 2: Inferencia final contra los clásicos — `test_generativo.py`

1. **Extracción dinámica:** El script lee `resultados_generativos.csv`, extrae el prompt ganador y lo reconstruye.
2. **Inferencia en Test:** Se aplica el prompt ganador al conjunto de Test (controlado por `eval_test_limit` en `config.json`).
3. **Integración:** Calcula la matriz de confusión, Recall, Precision y F1-Macro. Finalmente, anexa estos resultados a `metricas_modelos.csv` para comparar directamente si **Llama 3 ha derrotado a Random Forest o KNN**.

---

### Fase 3: Oversampling Generativo — `balancear_con_ia.py`

1. **Separación estricta:** Se separan Train y Dev *antes* de generar nada. El oversampling solo se aplica al Train para evitar **Data Leakage** hacia la validación.
2. **Cálculo de brecha:** Se identifica la clase minoritaria y cuántas muestras necesita para igualar a la mayoritaria.
3. **Generación con límite:** Usando el parámetro `n_generations` de `config.json` (ej. `300`), se le pide al LLM que parafrasee textos de la clase minoritaria hasta alcanzar el límite o equilibrar la clase. Este parámetro es muy útil, sobre todo para no extender la ejecución en caso de tener un ordenador con una potencia de cómputo limitado.
4. **Exportación:** Las nuevas reseñas se guardan en `parafrases_generadas_ollama.csv` y se genera un nuevo dataset de entrenamiento (`train_SOLO_balanceado_ia.csv`), sacando también el dataset dev sin modificar, obtenido al realizar la separación del train al principio (`dev_intacto.csv`).

---

## Herramientas y por qué cada una

| Herramienta | Rol en el proyecto |
|---|---|
| **Ollama (Llama 3 8B)** | LLM local: suficientemente capaz para razonamiento lingüístico, pero ligero para correr en CPU/GPU doméstica |
| **Scikit-learn** (`f1_score`, `classification_report`) | Marco de evaluación idéntico al de los modelos clásicos para comparativas justas y rigurosas |
| **Pandas** | Manipulación de datos, guardado de métricas y concatenación segura de DataFrames sintéticos con los originales |

---

## Qué se obtiene al finalizar

Al ejecutar este módulo generativo se producen tres outputs clave:

### 📄 `resultados_generativos.csv`
Registro auditable de la evolución de los prompts (del peor al mejor), junto con las entradas y salidas crudas del LLM.

### 🧪 `parafrases_generadas_ollama.csv`
Corpus de texto sintético de alta calidad para enriquecer los modelos de Machine Learning tradicionales.

### 📊 Actualización de `metricas_modelos.csv`
El ranking definitivo donde el mejor modelo generativo se compara directamente contra los algoritmos clásicos.

---

> Toda esta información permite contar una **historia muy rica en Tableau** sobre hasta qué punto la IA moderna entiende las quejas de los usuarios de plataformas musicales, frente a los enfoques tradicionales.