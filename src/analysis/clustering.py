#!/usr/bin/env python3
"""
Script de Modelado de Tópicos con Gensim (LDA)
Configuración dinámica desde la sección 'clustering' del JSON.
Líder: Liviu Deleanu
"""

import os
import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
import nltk
import gensim
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from sklearn.base import BaseEstimator, TransformerMixin
from langdetect import detect, LangDetectException
import unicodedata
import argparse

from src.utils.funciones import load_config, load_data, print_section_header

# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

try:
    STOP_WORDS = {
        'spanish': set(stopwords.words('spanish')),
        'english': set(stopwords.words('english')),
    }
except Exception as e:
    print(f"Error cargando stopwords NLTK: {e}")
    STOP_WORDS = {'spanish': set(), 'english': set()}

# Centralised stop-word lists — defined once, used everywhere
SENTIMENT_STOPWORDS = {
    'great', 'amazing', 'good', 'nice', 'awesome', 'perfect', 'excellent',
    'love', 'best', 'worst', 'super', 'like', 'much', 'many', 'ever',
    'just', 'with', 'from', 'every', 'have', 'than', 'more', 'better',
    'thanks', 'thank', 'appreciate','wonderful', 'fantastic', 'brilliant', 'superb', 'outstanding',
    'incredible', 'exceptional', 'cool', 'beautiful', 'amazing',
    'happy', 'satisfied', 'liked', 'feel', 'vibe', 'opinion',
    'sucks', 'stupid', 'garbage','terrible', 'awful', 'horrible', 'disappointing', 'worst','cheaper'
}
DOMAIN_STOPWORDS = {
    'song', 'songs', 'music', 'app', 'apps','aplication','application','soundcloud', 'spotify', 'apple',
    'tidal', 'listen', 'listening', 'playlist', 'track', 'tracks', 'artist', 'artists',
    'stars', 'star','goood', 'quran', 'commercials',
    'primium','premium', 'musics','screen'

}
CONTRACTION_NOISE = {
    'dont', 'cant', 'thats', 'doesnt', 'isnt', 'ive', 'youre',
    'everything', 'anything', 'nothing', 'something', 'thing', 'things',
    'version', 'versions', 'really', 'would', 'could', 'should',
    'experience', 'overall', 'works', 'simple', 'years','such', 'lots',
    'everybody', 'everywhere', 'everyone', 'everybody', 'everywhere',
    'everyday', 'work', 
    'only', 'other', 'single', 'same', 'whats', 'stuff', 'place',
    'kind', 'words', 'everyone', 'world', 'life', 'part', 'right',
    'able', 'need', 'please', 'guys', 'ones', 'someone', 'people',
    'today', 'daily', 'point', 'reason', 'school', 
    'huge', 'whole', 'high', 'easy', 'most', 'long', 'full', 'basic',
    'different', 'last', 'next', 'latest', 'first',
    'superrrrr', 'goto', 'truth', 'wouldnt','calming', 'software','previus', 'number',

}
DOMAIN_GARBAGE = {
    'music', 'song', 'songs', 'app', 'soundcloud', 'listen', 'listening',
    'really', 'would', 'could', 'should', 'experience', 'favorite',
    'overall', 'works', 'simple', 'years', 'ever', 'week', 'previous', 'collections',
    'mins', 'minutes', 'months', 'days','secs','seconds','second','hours', 'times','free', 'while', 
    'available', 'order','youll', 'cant', 'dont', 'thats', 'doesnt', 'isnt', 'ive', 'youre','sure'
}
SOUNDCLOUD_STOPS = {
    'dope', 'sick', 'fire', 'yeah', 'theres', 'okay',
    'awsome', 'goooood', 'communitydriven',
    'luck', 'wish', 'little', 'others', 'guess','feeling','youtube',
    'communitydriven',  'sound', 'cloud',  'awsome',
    'theres', 'alot',  'luck', 'vibes',    'greatest',   
    'havent', 'fabulous', 'black', 'real', 'home',
    'year', 'less', 'hard', 'share',    
}
TIDAL_STOPS = {
    'quality','serato', 'phenomenal', 'sluggish', 'arent', 
    'glad', 'period','main', 'feels', 'harder',
    'excelente','google', 'month','sounds','highest',
    'possible', 'amazon','superior', 'hands', 'switch', 'videos', 'social'     
}
APPLE_MUSIC_STOPS = {
    'apple', 'itunes', 'siri', 'iphone', 'ipad', 'applemusic', 'music',
    'streaming', 'ios', 'airpods', 'homepod', 'icloud', 'macbook', 'watch',
    'store', 'subscription', 'library', 'songs', 'song', 'artist', 'artists',
    'album', 'albums', 'playlist', 'playlists', 'radio', 'station', 'stations'
}
GAMING_NOISE = {'game', 'bomb', 'games', 'gaming', 'level', 'play'}

ALL_CUSTOM_STOPS = SENTIMENT_STOPWORDS | DOMAIN_STOPWORDS | CONTRACTION_NOISE | DOMAIN_GARBAGE | SOUNDCLOUD_STOPS | TIDAL_STOPS | APPLE_MUSIC_STOPS | GAMING_NOISE


# ---------------------------------------------------------------------------
# TEXT CLEANER
# ---------------------------------------------------------------------------
def contiene_no_latino(texto):
    for char in str(texto):
        if unicodedata.category(char) in ('Lo',) and ord(char) > 0x04FF:
            return True
    return False

class TextCleaner(BaseEstimator, TransformerMixin):
    """Keeps only nouns and adjectives, strips contractions and stopwords."""

    def __init__(self, language: str = 'english'):
        self.language = language

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        nltk_stops = STOP_WORDS.get(self.language, set())
        combined_stops = nltk_stops | ALL_CUSTOM_STOPS
        return [self._clean(str(text), combined_stops) for text in X]

    @staticmethod
    def _clean(text: str, stop_words: set) -> list:
        
        text = text.lower()
        # Remove contracted forms
        text = re.sub(
            r"\b(dont|cant|thats|doesnt|isnt|ive|youre|won't|can't|it's)\b",
            '', text,
        )
        text = re.sub(r'[^\w\s]', '', text)

        tokens = text.split()
        tokens = [t for t in tokens if t.isascii()]  # elimina tokens no-ASCII

        # POS tag and keep only nouns (NN*) and adjectives (JJ*) > 3 chars
        pos_tags = nltk.pos_tag(tokens)
        filtered = [
            word for word, tag in pos_tags
            if (tag.startswith('NN') or tag.startswith('JJ'))
            and len(word) > 3
            and word not in stop_words
        ]
        return filtered


# ---------------------------------------------------------------------------
# COHERENCE HELPER
# ---------------------------------------------------------------------------
def calcular_coherencia(corpus, dictionary, texts, cluster_cfg: dict):
    """
    Trains one LDA per K in the configured range and returns
    (k_range, coherence_scores).
    The K with the highest score is the suggested optimum.
    """
    k_min = cluster_cfg['num_topics_range']['min']
    k_max = cluster_cfg['num_topics_range']['max']
    metric = cluster_cfg.get('coherence_metric', 'c_v')

    print(f"  Buscando K óptimo con métrica '{metric}' ({k_min}–{k_max})…")
    scores = []
    for k in range(k_min, k_max + 1):
        model = gensim.models.LdaModel(
            corpus=corpus, id2word=dictionary,
            num_topics=k, random_state=42, passes=3,
        )
        cm = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence=metric)
        scores.append(cm.get_coherence())
        print(f"    K={k:2d}  {metric}={scores[-1]:.4f}")

    k_range = range(k_min, k_max + 1)
    best_k = k_min + scores.index(max(scores))
    print(f"  → K óptimo sugerido: {best_k}  (score={max(scores):.4f})")
    return k_range, scores, best_k


# ---------------------------------------------------------------------------
# TOPIC MODELLING
# ---------------------------------------------------------------------------
def get_topic_words(model, topic_id, min_words=5, topn=10):
        words_weights = model.show_topic(topic_id, topn=topn)
        # Intenta filtrar por peso primero
        filtered = [w for w, weight in words_weights if weight > 0.01]
        # Si quedan menos de min_words, toma las top min_words sin filtro
        if len(filtered) < min_words:
            filtered = [w for w, _ in words_weights[:min_words]]
        return ", ".join(filtered)

def _limpiar_palabras_topico(topic_string: str) -> str:
    """Strips numeric weights from a Gensim topic string."""
    return ", ".join(re.findall(r'"([^"]*)"', topic_string))


def analizar_topics_gensim(df, col_texto: str, polaridad: str,
                           idioma: str, cluster_cfg: dict):
    print_section_header(f"ANALIZANDO {polaridad.upper()}", char="-")

    # 1. Clean text — single, unified filtering pass
    cleaner = TextCleaner(language=idioma)
    texts = cleaner.transform(df[col_texto])

    # 2. Dictionary & corpus
    id2word = corpora.Dictionary(texts)
    # Relaxed no_below so small datasets keep more vocabulary
    no_below = 2 if polaridad == 'positivo' else max(2, cluster_cfg.get('no_below', 3))
    id2word.filter_extremes(no_below=no_below, no_above=0.5)
    corpus_bow = [id2word.doc2bow(doc) for doc in texts]

    # 3. TF-IDF weighting (downweights very frequent tokens)
    print("  Aplicando TF-IDF…")
    tfidf = gensim.models.TfidfModel(corpus_bow)
    corpus_tfidf = tfidf[corpus_bow]

    # 4. Coherence search + plot
    output_dir = cluster_cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    k_max_key = f"max_{polaridad}"  # "max_positivo" o "max_negativo"
    cluster_cfg_local = cluster_cfg.copy()
    cluster_cfg_local['num_topics_range']['max'] = cluster_cfg['num_topics_range'].get(k_max_key, 8)

    ks, scores, best_k = calcular_coherencia(corpus_tfidf, id2word, texts, cluster_cfg_local)

    plt.figure(figsize=(8, 4))
    plt.plot(list(ks), scores, marker='o', color='teal')
    plt.axvline(best_k, color='red', linestyle='--', label=f'K óptimo={best_k}')
    plt.title(f"Coherencia ({cluster_cfg['coherence_metric']}) — {polaridad}")
    plt.xlabel('Número de Tópicos')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/coherencia_{polaridad}.png", dpi=150)
    plt.close()

    # 5. Final LDA with auto-selected optimal K
    passes = cluster_cfg.get('passes', 10)
    print(f"  Entrenando LDA final (K={best_k}, passes={passes})…")
    lda_model = gensim.models.LdaModel(
        corpus=corpus_tfidf, id2word=id2word,
        num_topics=best_k, random_state=42, passes=passes,
    )

    # Print topics for inspection
    print("\n  Tópicos encontrados:")
    for i in range(best_k):

        print(f"    Tópico {i}: {get_topic_words(lda_model, i)}")

    # 6. Assign dominant topic to each document
    def get_topic_data(bow):
        topic_dist = lda_model.get_document_topics(bow)
        if not topic_dist:
            return -1, ""
        t_id = max(topic_dist, key=lambda x: x[1])[0]
        
        # Solo palabras con peso > 0.02 (ajusta según resultados)
        top_words = [
            word for word, weight in lda_model.show_topic(t_id, topn=10)
            if weight > 0.01
        ]
        top_words = get_topic_words(lda_model, t_id)
        return t_id, top_words
    
    results = [get_topic_data(bow) for bow in corpus_tfidf]

    df_out = df.copy()
    df_out['Polaridad_Clustering'] = polaridad
    df_out['Topic_ID'] = [r[0] for r in results]
    df_out['Palabras_Clave'] = [r[1] for r in results]
    return df_out


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print_section_header("LÍDER CLUSTERING: GENSIM LDA", char="═")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c','--config',help='Archivo de configuración', default='config.json', type=str)
    args = parser.parse_args()
    
    # 1. Config
    full_config = load_config(args.config)
    cluster_cfg = full_config['clustering']
    gen_cfg = full_config['general']

    idioma = gen_cfg.get('language', 'english')
    col_texto = cluster_cfg['text_col']
    ruta_datos = cluster_cfg['data']['train_dev']

    # 2. Load data
    print(f"Cargando archivo: {ruta_datos}")
    df = load_data(ruta_datos)

    # 3. Ensure sentiment column exists
    sent_col = gen_cfg.get('column', 'sentiment')
    if sent_col not in df.columns:
        if 'score' in df.columns:
            print(f"  Columna '{sent_col}' no encontrada → mapeando desde 'score'…")
            def map_sentiment(s):
                try:
                    val = int(s)
                    if val <= 2:  return 'negativo'
                    if val == 3:  return 'neutro'
                    if val == 4:  return 'mixto'    # ← nueva categoría
                    return 'positivo'  # solo score 5
                except: return 'neutro'
            df[sent_col] = df['score'].apply(map_sentiment)
        else:
            print(f"Error: columna '{sent_col}' ni 'score' encontradas.")
            sys.exit(1)

    # 4. Aplicar filtros ANTES de dividir
    df = df[df[col_texto].str.split().str.len() >= 8] 
    df = df[~df[col_texto].str.contains(r'[^\x00-\x7F]', regex=True, na=False)]
    print(f"Reviews tras filtro de idioma y longitud: {len(df)}")  

    # 5. Split by sentiment
    df_pos = df[df[sent_col] == 'positivo']
    df_neg = df[df[sent_col] == 'negativo']
    
    sample = df_pos[df_pos[col_texto].str.contains('problem', case=False)]
    print(f"Reviews score 5 con 'problem': {len(sample)}")
    if not sample.empty:
        print(sample[col_texto].head(3).to_string())

    # 6. Analizar topics si no están vacíos
    if not df_pos.empty:
        res_pos = analizar_topics_gensim(df_pos, col_texto, 'positivo', idioma, cluster_cfg)
    else:
        print("No hay reviews positivas")
        res_pos = pd.DataFrame()  # Evita que el pd.concat posterior falle

    if not df_neg.empty:
        res_neg = analizar_topics_gensim(df_neg, col_texto, 'negativo', idioma, cluster_cfg)
    else:
        print("No hay reviews negativas")
        res_neg = pd.DataFrame()  # Evita que el pd.concat posterior falle

    # 7. Export for Tableau
    final_df = pd.concat([res_pos, res_neg], ignore_index=True)
    output_path = f"{cluster_cfg['output_dir']}/resultados_tableau_{cluster_cfg['name']}.csv"

    cols_export = [col_texto, sent_col, 'Polaridad_Clustering', 'Topic_ID', 'Palabras_Clave']
    for c in ('location', 'gender', 'date'):
        if c in final_df.columns:
            cols_export.append(c)
    
    # Extraemos un csv con los tópicos
    topic_path = f"{cluster_cfg['output_dir']}/topicos_{cluster_cfg['name']}.csv"
    df_topics = final_df[['Polaridad_Clustering', 'Topic_ID', 'Palabras_Clave']].drop_duplicates()
    df_topics = df_topics.sort_values(by=['Polaridad_Clustering', 'Topic_ID']).reset_index(drop=True)
    df_topics.to_csv(topic_path, index=False)

    os.makedirs(cluster_cfg['output_dir'], exist_ok=True)
    final_df[cols_export].to_csv(output_path, index=False)

    print_section_header("PROCESO COMPLETADO", char="═")
    print(f"Resultados del clustering guardados en: {output_path}")
    print(f"Resumen de tópicos guardado en: {topic_path}")


if __name__ == '__main__':
    main()
