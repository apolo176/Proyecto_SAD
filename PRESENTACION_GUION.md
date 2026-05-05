# Guion de Presentación — Proyecto SAD
## Análisis de Sentimientos de Reseñas de Plataformas Musicales
**Duración estimada: 25-30 minutos**

---

## INTRODUCCIÓN (2 min)

> "Buenas tardes. El proyecto que presentamos hoy es un análisis de sentimientos sobre reseñas de usuarios de cuatro plataformas musicales: Spotify, Apple Music, Tidal y SoundCloud."

> "La motivación es sencilla: las plataformas reciben miles de reseñas diarias en las tiendas de aplicaciones. Procesarlas manualmente es inviable, pero contienen información valiosísima sobre qué falla y qué funciona. Nosotros construimos un sistema que las analiza automáticamente."

> "El proyecto se divide en cuatro bloques: visualización con Tableau, clasificación supervisada con modelos clásicos, un módulo generativo con LLM, y análisis de tópicos con clustering. Cada bloque tiene un responsable, y yo os explicaré el bloque de clustering."

**Datos del dataset:**
> "Trabajamos con 15.000 reseñas de Apple Music y 15.000 de Spotify, más datos adicionales de Tidal y SoundCloud. Cada reseña tiene: texto, puntuación del 1 al 5, género, localización y fecha. Convertimos las puntuaciones en tres etiquetas: 1-2 es NEGATIVO, 3 es NEUTRO, 4-5 es POSITIVO."

---

## BLOQUE 1: TABLEAU (5 min)

> "Empezamos con la parte de visualización. Tableau nos permite explorar los datos de forma interactiva y comunicar los hallazgos a cualquier stakeholder sin que necesite entender el código."

### Visualización 1: Evolución anual de puntuaciones
> "Este gráfico muestra cómo ha evolucionado la valoración media de cada plataforma desde 2008 hasta 2025. Usamos una media bayesiana en vez de una media aritmética simple, porque hay años con muy pocas reseñas y una media simple daría valores engañosos. La tendencia es clara: ambas plataformas han empeorado en valoración desde 2015."

### Visualización 2: Comparativa demográfica
> "Analizamos si el género del usuario influye en las valoraciones. El resultado es que hombres y mujeres muestran tendencias negativas similares. Esto nos dice que el problema no es demográfico, sino estructural: los productos están fallando a todo el mundo por igual."

### Visualización 3: Evolución por países
> "Este mapa interactivo muestra, país por país, si los usuarios prefieren Spotify o Apple Music. Calculamos un coeficiente que va de -5 a +5: negativo significa preferencia por Apple Music, positivo por Spotify."

### Visualización 4: Análisis semántico — nubes de palabras
> "Por último, usamos nubes de palabras para ver qué términos aparecen más en reseñas positivas y negativas de cada plataforma. Las conclusiones son muy concretas:"
> - "Apple Music tiene problemas de compatibilidad multiplataforma, bugs técnicos, y no ofrece nivel gratuito."
> - "Spotify falla en las restricciones del modo premium: reproducción aleatoria forzada y sin poder saltar anuncios."
> - "Las fortalezas son complementarias: Apple destaca en calidad de audio, Spotify en recomendaciones y accesibilidad."

---

## BLOQUE 2: CLASIFICACIÓN (6 min)

> "El segundo bloque es la clasificación supervisada. El objetivo es entrenar un modelo que, dado el texto de una reseña, prediga automáticamente si es positiva, neutra o negativa."

### Pipeline de preprocesamiento
> "El texto pasa por un pipeline: primero TF-IDF, que convierte las palabras en vectores numéricos penalizando términos que aparecen en casi todas las reseñas. Eliminamos stopwords en inglés y español según la plataforma."

### Modelos entrenados
> "Entrenamos y comparamos cuatro familias de modelos:"
> - "**K-Nearest Neighbors:** clasifica por similitud con los vecinos más cercanos en el espacio TF-IDF."
> - "**Árbol de Decisión:** crea reglas de clasificación jerárquicas."
> - "**Random Forest:** ensemble de múltiples árboles para reducir el sobreajuste."
> - "**Naive Bayes Multinomial:** modelo probabilístico especialmente eficaz con texto."

### Búsqueda de hiperparámetros
> "Usamos GridSearchCV con validación cruzada de 5 folds para encontrar los mejores hiperparámetros de cada modelo. El conjunto de datos se divide en train, dev y test: el dev se usa durante el desarrollo para comparar modelos, el test solo se toca al final para la evaluación definitiva."

### Resultados
> "La métrica que usamos es F1-Macro, que trata igual a todas las clases independientemente de su frecuencia. Los resultados en el conjunto de dev son:"
> - "Naive Bayes Multinomial: **0.60** — el mejor modelo clásico."
> - "Random Forest: 0.55"
> - "Árbol de Decisión: 0.48"
> - "KNN: 0.44"

> "El Naive Bayes destaca porque es un modelo probabilístico que se ajusta muy bien a la distribución de palabras en texto. El KNN rinde peor porque en espacios de alta dimensión como TF-IDF, la noción de 'distancia' entre vectores pierde significado."

---

## BLOQUE 3: GENERATIVA (7 min)

> "El tercer bloque usa inteligencia artificial generativa. Aquí no entrenamos un modelo desde cero: usamos Llama 3 de 8 mil millones de parámetros ejecutado localmente con Ollama."

### Parte 3a: Clasificación por prompt engineering
> "La idea es: ¿puede un LLM, sin ningún entrenamiento adicional, clasificar sentimientos igual de bien que los modelos clásicos? Para responder esto, diseñamos y evaluamos 9 prompts distintos organizados en tres estrategias:"

> "**Zero-shot:** solo describimos la tarea. La versión 'buena' asigna un rol al modelo ('eres un experto en análisis de sentimientos'), define el formato de respuesta como A/B/C, y limita la salida a un solo token con `num_predict=1`. Esto reduce el tiempo de inferencia de 2-3 segundos a 0.2-0.5 segundos por reseña."

> "**One-shot:** añadimos un ejemplo. En la versión 'mejorada' ponemos un ejemplo de cada clase: positivo, neutro y negativo."

> "**Few-shot:** varios ejemplos. La versión 'mejorada' incluye 5 ejemplos cubriendo casos difíciles: sentimiento mixto, quejas técnicas, problemas de facturación."

> "Evaluamos los 9 prompts en el conjunto de dev y seleccionamos automáticamente el que obtiene mejor F1-Macro."

### Parte 3b: Comparación con modelos clásicos
> "El mejor prompt se aplica al conjunto de test y se añade a la tabla de métricas junto con los modelos clásicos. Así comparamos directamente: modelos entrenados versus LLM zero-shot."

### Parte 3c: Aumentación de datos con IA
> "También usamos el LLM para balancear el dataset. Si la clase 'neutro' tiene pocas reseñas, pedimos al modelo que parafrasee reseñas existentes manteniendo el mismo sentimiento. Es importante hacer esta generación ANTES del split train/dev para no contaminar el conjunto de evaluación."

---

## BLOQUE 4: CLUSTERING — MI PARTE (8 min)

> "Y ahora paso al bloque que yo he desarrollado: el análisis de tópicos mediante clustering."

### Motivación
> "La clasificación nos dice *si* una reseña es negativa, pero no nos dice *por qué*. El clustering de tópicos responde esa pregunta. Un usuario puede estar insatisfecho por los anuncios, por bugs técnicos, por el precio, o por la experiencia de usuario. Son problemas muy distintos que requieren soluciones distintas."

### Algoritmo: LDA (Latent Dirichlet Allocation)
> "Usamos LDA, un modelo probabilístico que asume que cada documento es una mezcla de tópicos, y cada tópico es una distribución de palabras. El modelo infiere estas distribuciones a partir del corpus sin ninguna etiqueta."

> "Separamos las reseñas positivas de las negativas antes de aplicar LDA. La razón es que si las mezclamos, los tópicos resultantes son incoherentes: una queja sobre anuncios y un elogio sobre recomendaciones tienen vocabulario completamente diferente. Al separarlas, obtenemos tópicos accionables: 'esto es lo que falla' y 'esto es lo que gusta'."

### Preprocesamiento específico para clustering
> "El preprocesamiento para LDA es más agresivo que para clasificación, porque no queremos capturar sentimiento sino tema:"

> 1. "Eliminamos reseñas de menos de 8 palabras: demasiado cortas para aportar información temática."
> 2. "POS tagging con NLTK: nos quedamos SOLO con sustantivos y adjetivos de más de 3 caracteres. Los verbos y adverbios tienden a ser genéricos y no distinguen tópicos."
> 3. "Tres capas de stopwords: genéricas ('the', 'is'), específicas de sentimiento ('amazing', 'terrible') y de dominio ('app', 'music', 'spotify'). Si no eliminamos las palabras de dominio, todos los tópicos hablarían de 'música' y 'aplicación'."

### Construcción del corpus y selección de K
> "Construimos el vocabulario, convertimos las reseñas a Bag-of-Words, y aplicamos filtrado TF-IDF para penalizar palabras que aparecen en casi todos los documentos."

> "Para seleccionar el número óptimo de tópicos K, entrenamos LDA para cada K en un rango definido en la configuración y calculamos la **coherencia c_v** de cada modelo. La coherencia mide si las palabras de cada tópico suelen aparecer juntas en el corpus, lo que indica que el tópico tiene sentido semántico. Elegimos el K con mayor coherencia."

> "Generamos gráficas de coherencia para reseñas positivas y negativas de cada plataforma, lo que permite justificar la elección de K."

### Resultados y exportación
> "El modelo LDA asigna cada reseña a su tópico dominante. Exportamos los resultados en CSV con la reseña original, la polaridad, el ID del tópico y las palabras clave. Estos ficheros son exactamente los que consume Tableau para generar las visualizaciones del bloque 1."

> "Por ejemplo, en reseñas negativas podemos encontrar tópicos como: 'offline, download, cache, connection' que agrupa usuarios que quieren escuchar sin conexión; o 'ads, premium, free, paid' que agrupa quejas sobre publicidad y precios; o 'crash, freeze, update, broken' para problemas técnicos."

---

## CONCLUSIONES (3 min)

> "Para cerrar, ¿qué hemos conseguido con este proyecto?"

> "Primero: un pipeline completo de análisis de opinión, desde la reseña cruda hasta visualizaciones interactivas en Tableau, pasando por modelos de ML, LLM y clustering."

> "Segundo: hemos comparado enfoques clásicos con IA generativa. El mejor modelo clásico, Naive Bayes, alcanza F1-Macro de 0.60. El LLM, sin entrenamiento, ofrece una alternativa flexible cuando no se tienen datos etiquetados."

> "Tercero: el clustering añade una capa interpretable que los modelos de clasificación no dan. Sabemos que los usuarios de Spotify se quejan del modo premium, y que los de Apple Music se quejan de la falta de versión gratuita. Eso es información accionable para el negocio."

> "El proyecto demuestra que es posible construir, con herramientas open-source, un sistema de análisis de opinión comparable al de grandes empresas, y que la combinación de ML clásico, LLM y modelos no supervisados se complementan mejor que cualquiera de ellos solo."

---

---

# PREGUNTAS FRECUENTES DE LA PROFESORA

---

## PREGUNTAS SOBRE TABLEAU

**P: ¿Por qué usáis media bayesiana y no media aritmética para la evolución temporal?**
> R: Porque la media aritmética es inestable cuando hay pocas muestras. Si en 2009 solo hay 3 reseñas y las tres son negativas, la media da 1.0, pero eso no representa la opinión real de los usuarios de ese año, sino simplemente que la muestra es demasiado pequeña. La media bayesiana "pesa" la estimación hacia la media global cuando hay pocas observaciones, y se va ajustando a los datos reales conforme aumenta el número de reseñas.

**P: ¿Qué os dice el mapa por países para el negocio?**
> R: Permite hacer estrategias de marketing localizado. Si en ciertos países Spotify domina claramente pero Apple Music tiene mejor valoración en audio, Apple podría hacer campañas específicas en esos mercados destacando su calidad de sonido. Y viceversa.

**P: ¿Las nubes de palabras son análisis cualitativo o cuantitativo?**
> R: Son cuantitativas: el tamaño de cada palabra es proporcional a su frecuencia en el corpus filtrado. Lo que las hace útiles es que aplicamos preprocesamiento (eliminar stopwords, palabras de dominio) para que las palabras que aparecen sean las realmente discriminativas del tema.

---

## PREGUNTAS SOBRE CLASIFICACIÓN

**P: ¿Por qué TF-IDF y no embeddings como Word2Vec o BERT?**
> R: TF-IDF es interpretable, eficiente y funciona bien con vocabulario específico de dominio. Word2Vec y BERT capturan mejor el contexto semántico, pero requieren más recursos y son más difíciles de explicar. Para este proyecto, la comparativa con el LLM era más interesante que usar embeddings en el modelo clásico.

**P: ¿Cómo manejáis el desbalanceo de clases?**
> R: El pipeline incluye opciones de oversampling con RandomOverSampler (de la librería imbalanced-learn) o undersampling. También tenemos el módulo generativo que puede generar reseñas sintéticas de la clase minoritaria con Ollama.

**P: ¿Por qué Naive Bayes supera a Random Forest?**
> R: Porque el texto en representación TF-IDF cumple aproximadamente las hipótesis de independencia de Naive Bayes. Además, Random Forest puede sobreajustarse más fácilmente en espacios de alta dimensionalidad como TF-IDF (miles de features), mientras Naive Bayes es más regularizado por naturaleza.

**P: ¿Qué significa F1-Macro y por qué lo elegís frente a accuracy?**
> R: F1-Macro calcula el F1 de cada clase por separado y luego hace la media, sin ponderar por tamaño de clase. Con datos desbalanceados, accuracy es engañosa: un modelo que siempre predice 'positivo' tendría accuracy alta si el 80% de las reseñas son positivas, pero F1-Macro de la clase negativa sería 0. F1-Macro nos obliga a que el modelo funcione bien en todas las clases.

**P: ¿Qué es GridSearchCV?**
> R: Es una búsqueda exhaustiva sobre una malla de hiperparámetros con validación cruzada. Probamos todas las combinaciones posibles de los hiperparámetros definidos y para cada combinación calculamos el F1-Macro promedio sobre 5 particiones del conjunto de entrenamiento. Elegimos la combinación con mejor F1-Macro promedio.

---

## PREGUNTAS SOBRE GENERATIVA

**P: ¿Por qué Llama 3 local y no la API de OpenAI o Claude?**
> R: Dos razones. Primera: privacidad, las reseñas de usuarios no salen del entorno local. Segunda: coste, ejecutar miles de inferencias con una API de pago sería caro para un proyecto universitario. Ollama permite ejecutar el modelo en la GPU o CPU del equipo.

**P: ¿Qué es `num_predict=1` y por qué lo usáis?**
> R: Es un parámetro de Ollama que limita el número máximo de tokens generados. Al limitar a 1, el modelo solo genera la primera letra de su respuesta: A, B o C, correspondiente a positivo, neutro o negativo. Esto reduce el tiempo de inferencia de 2-3 segundos a 0.2-0.5 segundos por reseña, lo que hace viable procesar miles de reseñas.

**P: ¿Qué pasa si el modelo no genera A, B ni C?**
> R: El código extrae el primer carácter de la respuesta y lo mapea. Si no es A, B ni C, la reseña se clasifica como no mapeable y se excluye del cálculo de métricas. En la práctica, con el prompt 'bueno' bien diseñado, el ratio de fallos es muy bajo porque el prompt especifica explícitamente el formato.

**P: ¿La aumentación con IA no introduce sesgo en el dataset?**
> R: Es un riesgo real. Por eso hay dos salvaguardas. Primero, la generación se hace antes del split, lo que significa que las paráfrasis no 'contaminan' el dev set. Segundo, usamos temperatura 0.8 para que las reseñas generadas tengan variabilidad léxica real y no sean copias casi idénticas. Aun así, es una limitación: las reseñas generadas siguen los patrones del modelo, no los patrones reales de usuarios.

**P: ¿Cómo comparáis el LLM con los modelos clásicos?**
> R: Aplicamos el mejor prompt al conjunto de test (el mismo que usan los modelos clásicos) y calculamos exactamente las mismas métricas: F1-Macro, accuracy, precisión y recall. Todo se añade a la misma tabla de resultados para una comparación directa.

**P: ¿El LLM supera a los modelos clásicos?**
> R: No necesariamente en F1-Macro, pero ofrece ventajas distintas. Los modelos clásicos requieren datos etiquetados para entrenarse; el LLM no. Si tuviéramos que analizar reseñas de una nueva plataforma sin datos etiquetados, el LLM es usable de inmediato mientras que el modelo clásico necesitaría un proceso de reentrenamiento.

---

## PREGUNTAS SOBRE CLUSTERING

**P: ¿Por qué LDA y no K-Means o DBSCAN?**
> R: K-Means trabaja con distancias euclidianas en el espacio vectorial, pero los documentos de texto no se distribuyen de forma esférica. DBSCAN requiere ajustar parámetros de densidad que son difíciles de intuir en texto. LDA es un modelo probabilístico diseñado específicamente para documentos: captura que una reseña puede hablar de varios temas a la vez, que es lo que ocurre en la realidad.

**P: ¿Qué es la coherencia c_v y cómo se interpreta?**
> R: La coherencia c_v mide la co-ocurrencia entre las palabras más representativas de cada tópico. Si las top-10 palabras de un tópico suelen aparecer juntas en los documentos del corpus, el tópico tiene coherencia alta. Valores más altos significan tópicos más interpretables semánticamente. Se usa para seleccionar el número óptimo de K.

**P: ¿Los tópicos son interpretables automáticamente?**
> R: No completamente. LDA da las palabras más representativas de cada tópico, pero la etiqueta semántica (ej: "problemas con pagos") la asigna un humano revisando esas palabras. Es una limitación conocida de LDA. En proyectos industriales se suele complementar con interfaces de inspección manual.

**P: ¿Por qué separáis reseñas positivas y negativas antes del clustering?**
> R: Porque mezclarlas produce tópicos incoherentes. Imagina que el tópico 1 mezcla "great sound quality" con "terrible sound crash": el tópico no tiene sentido. Al separarlas, los tópicos negativos son problemas concretos y los positivos son fortalezas concretas. Esto hace los resultados directamente accionables para el negocio.

**P: ¿Cómo se conecta el clustering con Tableau?**
> R: El clustering exporta un CSV por plataforma que incluye la reseña original más tres columnas añadidas: polaridad, ID de tópico y palabras clave del tópico. Tableau carga ese CSV y puede filtrar por tópico, plataforma, fecha o género para crear visualizaciones interactivas. Es el puente entre el análisis estadístico y la presentación visual.

**P: ¿Cuántos tópicos encontrasteis en cada plataforma?**
> R: El número varía por plataforma y polaridad, porque lo seleccionamos según la coherencia. El rango y los valores concretos están en los archivos de salida del modelo. El punto clave es que no imponemos un número fijo: dejamos que los datos indiquen cuántos tópicos son estadísticamente coherentes.

---

## PREGUNTAS GENERALES DEL PROYECTO

**P: ¿Por qué estas cuatro plataformas?**
> R: Son las principales plataformas de streaming musical con reseñas públicas en tiendas de aplicaciones en cantidad suficiente para el análisis. Tener plataformas distintas permite comparar y encontrar patrones específicos de cada producto.

**P: ¿Cuál es la limitación más importante del proyecto?**
> R: El idioma. Las reseñas están en inglés y el preprocesamiento (stopwords, POS tagging) está calibrado para inglés. Un sistema multi-idioma requeriría detectar el idioma y aplicar modelos específicos. Otra limitación es que las puntuaciones de las tiendas pueden no reflejar fielmente el sentimiento del texto, lo que añade ruido al etiquetado.

**P: ¿Qué mejoraríais si tuvierais más tiempo?**
> R: Para clustering: usar embeddings de BERT para construir el corpus en lugar de Bag-of-Words, lo que capturaría mejor el contexto semántico. Para clasificación: probar fine-tuning de un modelo BERT pequeño. Para el módulo generativo: explorar si con few-shot prompting el LLM supera al Naive Bayes entrenado.

**P: ¿El sistema escala a millones de reseñas?**
> R: El módulo de clasificación y clustering sí, con optimizaciones de batch processing. El módulo generativo con Llama 3 local sería el cuello de botella: 0.5 segundos por reseña significa 138 horas para un millón de reseñas en una sola máquina. En producción se paralelizaría o se usaría una API con rate limits.
