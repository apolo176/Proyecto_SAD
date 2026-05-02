# Clustering — Qué hace, por qué y cómo

---

## El problema que resuelve

Tenemos miles de reseñas de apps de música. Sabemos si cada reseña es positiva o negativa (eso lo hace el clasificador), pero no sabemos **de qué hablan**. Un usuario puede dar 1 estrella porque la app se cuelga, otro porque no tiene la canción que busca, otro porque no funciona sin conexión. Todos son negativos, pero por razones completamente distintas.

El clustering responde a: **¿sobre qué temas hablan los usuarios cuando se quejan? ¿y cuando están contentos?**

---

## La solución: LDA

Usamos **LDA** (Latent Dirichlet Allocation), un algoritmo de modelado de tópicos. La idea es sencilla: dado un conjunto de textos, LDA descubre automáticamente los temas que aparecen en ellos, sin que tú le digas cuáles son de antemano.

**Analogía:** imagina que tienes 5.000 cartas mezcladas de distintos temas y las lees todas. Al final, empiezas a notar patrones: algunas cartas hablan siempre de dinero y facturas, otras de problemas técnicos, otras de viajes. LDA hace exactamente eso, pero con matemáticas.

Lo que devuelve LDA por cada tópico es un conjunto de palabras que tienden a aparecer juntas. Por ejemplo:

- Tópico 0: `offline, download, available, cache, connection` → usuarios que quieren escuchar sin internet
- Tópico 1: `ads, commercial, free, premium, paid` → usuarios que se quejan de la publicidad
- Tópico 2: `crash, freeze, bug, update, broken` → usuarios con problemas técnicos

Y a cada reseña le asigna el tópico dominante, es decir, el tema sobre el que más habla.

---

## Por qué separamos positivos y negativos

No mezclamos todas las reseñas en un mismo análisis. Las corremos por separado.

La razón es simple: los usuarios positivos y negativos hablan de cosas distintas. Un positivo habla de descubrimiento de música, recomendaciones, calidad de sonido. Un negativo habla de fallos, precios, bugs. Si los mezclas, los temas que saca LDA son un batiburrillo inútil.

Separando, obtenemos tópicos limpios y accionables para cada tipo de sentimiento.

---

## Cómo se elige el número de tópicos (K)

El problema con LDA es que tú decides cuántos temas quieres encontrar. Si pides 2, te da 2 muy genéricos; si pides 20, te da 20 demasiado específicos. Hay que encontrar el punto óptimo.

Para esto usamos la **métrica de coherencia** (`c_v`). Mide, básicamente, qué tan parecidas son las palabras dentro de un mismo tópico. Un tópico con palabras `crash, bug, freeze, error` es muy coherente; uno con `crash, song, premium, download` no lo es tanto.

El script prueba todos los valores de K entre un mínimo y un máximo (configurados en `config.json`), calcula la coherencia de cada uno, y elige el K con puntuación más alta. También genera una gráfica para que puedas verlo tú:

```
coherencia_positivo.png
coherencia_negativo.png
```

---

## El pipeline completo, paso a paso

### 1. Filtrado inicial

Antes de analizar nada, filtramos reseñas que no sirven:
- Reseñas con menos de 8 palabras (demasiado cortas para extraer temas)
- Reseñas con caracteres no ASCII (otros idiomas que no podemos analizar bien)

### 2. Limpieza del texto

La limpieza es más agresiva que en el clasificador porque aquí necesitamos palabras con significado semántico real. Se hace en varios pasos:

**Normalización básica:** todo a minúsculas, se eliminan signos de puntuación y contracciones (`don't` → `dont` → eliminado).

**POS tagging:** se analiza la función gramatical de cada palabra (sustantivo, adjetivo, verbo, etc.) y **solo se conservan sustantivos y adjetivos** de más de 3 caracteres. Los verbos y adverbios casi nunca identifican temas concretos.

> Analogía: si alguien te dice "esta app es absolutamente terrible y nunca funciona bien", lo que importa para identificar el tema es `app` y `terrible`, no `absolutamente`, `nunca` o `bien`.

**Eliminación de stopwords en tres capas:**
- Stopwords genéricas del idioma (`the`, `is`, `and`...)
- Stopwords de sentimiento: palabras que aparecen en todas las reseñas independientemente del tema (`great`, `amazing`, `horrible`, `love`...). Son útiles para clasificar sentimiento, pero no para distinguir temas.
- Stopwords de dominio: palabras propias del contexto que no aportan información (`song`, `music`, `app`, `spotify`, `playlist`...). Están en todas las reseñas, así que no distinguen nada.

El resultado es una lista de tokens por reseña, como `['download', 'offline', 'storage', 'cache']`.

### 3. Diccionario y corpus

Con los tokens limpios se construye un **diccionario**: un índice que asigna un número a cada palabra única del vocabulario. Luego se convierte cada reseña en un vector de frecuencias (cuántas veces aparece cada palabra del diccionario en ese texto). Esto es el corpus en formato BoW (Bag of Words).

Se filtran palabras que aparecen en menos de 2 documentos (son tan raras que no forman un tema) y las que aparecen en más del 50% (son tan comunes que tampoco distinguen temas).

### 4. Ponderación TF-IDF

Sobre el corpus BoW se aplica TF-IDF. Esto penaliza las palabras que aparecen mucho en todos los documentos y premia las que son específicas de pocos. El efecto es que las palabras que realmente distinguen un texto del resto tienen más peso en el análisis.

### 5. Búsqueda del K óptimo

Se entrena un modelo LDA para cada K posible, se calcula la coherencia, y se guarda la gráfica. El K con mayor coherencia se usa en el paso siguiente.

### 6. Modelo LDA final

Se entrena el LDA definitivo con el K óptimo y con más pasadas (`passes` en `config.json`) para que converja mejor.

### 7. Asignación de tópicos

Cada reseña recibe el ID del tópico al que pertenece más (el de mayor probabilidad según el modelo) y las palabras clave de ese tópico.

### 8. Exportación para Tableau

Se genera un CSV con estas columnas añadidas a los datos originales:

| Columna | Contenido |
|--------|-----------|
| `Polaridad_Clustering` | `positivo` o `negativo` |
| `Topic_ID` | número del tópico asignado |
| `Palabras_Clave` | palabras representativas del tópico |

Este CSV es el que se importa directamente en Tableau para visualizar qué temas concentran más quejas o más elogios, cruzado con plataforma, fecha, etc.

---

## Herramientas y por qué cada una

**Gensim** — librería especializada en modelado de tópicos y embeddings de texto. Es la referencia en LDA; más eficiente y con más opciones que la implementación de sklearn para este caso.

**NLTK** — para el POS tagging (etiquetar cada token con su función gramatical). Usamos el tagger `averaged_perceptron_tagger`, que es rápido y suficientemente preciso para inglés.

**Matplotlib** — para generar las gráficas de coherencia. No hay más misterio.

**langdetect** — importado pero usado en versiones previas para filtrar idioma. Los filtros actuales de ASCII cumplen esa función.

---

## Qué se obtiene al final

Por cada plataforma (Spotify, Apple Music, Tidal, SoundCloud) y cada sentimiento (positivo, negativo):

- Una **gráfica de coherencia** que muestra cómo varía la calidad de los tópicos según K
- Un **CSV** con cada reseña etiquetada con su tópico y sus palabras clave, listo para Tableau

Eso permite responder preguntas como: *"¿Los usuarios de Tidal que se quejan, de qué se quejan exactamente? ¿Es lo mismo que los de Spotify?"* o *"¿Qué valoran los usuarios satisfechos de Apple Music que no valoran los de SoundCloud?"*