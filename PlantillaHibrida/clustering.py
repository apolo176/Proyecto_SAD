#!/usr/bin/env python3
"""
Script de Modelado de Tópicos con Gensim (LDA)"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Gensim imports
import gensim
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

# Sklearn (solo para la estructura de limpieza)
from sklearn.base import BaseEstimator, TransformerMixin

import re
import nltk
from nltk.corpus import stopwords

# Configuración de NLTK
try:
    STOP_WORDS_DICT = {
        'spanish': set(stopwords.words('spanish')),
        'english': set(stopwords.words('english'))
    }
except Exception:
    STOP_WORDS_DICT = {'spanish': set(), 'english': set()}

# ---------------------------------------------------------
# 1. PREPARACIÓN Y LIMPIEZA
# ---------------------------------------------------------

def preparar_datos(ruta_archivo):
    """Carga y prepara el sentimiento a partir del score."""
    df = pd.read_csv(ruta_archivo)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    if 'score' in df.columns and 'sentiment' not in df.columns:
        def map_sentiment(score):
            try:
                s = int(score)
                return 'negativo' if s <= 2 else ('neutro' if s == 3 else 'positivo')
            except: return 'neutro'
        df['sentiment'] = df['score'].apply(map_sentiment)
    return df

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, language='english'):
        self.language = language
    
    def transform(self, X):
        stop_words = STOP_WORDS_DICT.get(self.language, set())
        return [self._clean_text(str(t), stop_words).split() for t in X]
    
    def _clean_text(self, text, stop_words):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return " ".join([w for w in text.split() if w not in stop_words and len(w) > 3])

# ---------------------------------------------------------
#  MODELADO DE TÓPICOS
# ---------------------------------------------------------

def calcular_coherencia(corpus, dictionary, texts, max_k):

    print(f"Calculando Coherence Scores (K=2 hasta {max_k})...")
    coherence_values = []
    model_list = []
    for num_topics in range(2, max_k + 1):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=num_topics, 
                                                id2word=dictionary, random_state=42, passes=10)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    
    return coherence_values

def analizar_topics_gensim(df, columna_texto, polaridad, k_optimo=4):
    print(f"\n{'-'*50}\nANALIZANDO CON GENSIM: {polaridad.upper()}\n{'-'*50}")
    
    # Limpieza
    cleaner = TextCleaner(language='english')
    texts = cleaner.transform(df[columna_texto])
    
    # Crear Diccionario y BOW
    id2word = corpora.Dictionary(texts)
    corpus = [id2word.doc2bow(text) for text in texts]
    
    # Grafico
    max_k = 10
    scores = calcular_coherencia(corpus, id2word, texts, max_k)
    
    plt.plot(range(2, max_k + 1), scores)
    plt.title(f'Coherencia de Tópicos - {polaridad}')
    plt.xlabel('Num Topics')
    plt.ylabel('Coherence score')
    os.makedirs('resultados', exist_ok=True)
    plt.savefig(f"resultados/coherencia_{polaridad}.png")
    plt.close()

    # Entrenar modelo final
    print(f"Entrenando LDA con K={k_optimo}...")
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, num_topics=k_optimo, 
                                                id2word=id2word, random_state=42, passes=15)
    
    # Palabras clave
    print("\n TÓPICOS IDENTIFICADOS:")
    for idx, topic in lda_model.print_topics(-1):
        print(f"      • Tópico {idx}: {topic}")

    # Asignar el tópico dominante
    def get_main_topic(bow):
        topics = lda_model.get_document_topics(bow)
        return max(topics, key=lambda x: x[1])[0]

    df_res = df.copy()
    df_res['Tópico_Dominante'] = [get_main_topic(c) for c in corpus]
    return df_res

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-k_pos', type=int, default=3)
    parser.add_argument('-k_neg', type=int, default=3)
    args = parser.parse_args()
    
    df = preparar_datos(args.input)
    
    df_pos = df[df['sentiment'] == 'positivo']
    df_neg = df[df['sentiment'] == 'negativo']
    
    res_pos = analizar_topics_gensim(df_pos, 'review', 'positivos', k_optimo=args.k_pos)
    res_neg = analizar_topics_gensim(df_neg, 'review', 'negativos', k_optimo=args.k_neg)
    
    final_df = pd.concat([res_pos, res_neg])
    final_df.to_csv('resultados/datos_clustering_gensim.csv', index=False)
    print(f"\nProceso completado.")

if __name__ == '__main__':
    main()