#!/usr/bin/env python3
"""
Script de Clustering para Análisis de Tópicos.
Objetivo: Descubrir POR QUÉ los usuarios dejan reviews positivas o negativas.
Líder: Liviu Deleanu
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sklearn imports para Clustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin

# NLTK para procesamiento de texto
import re
import nltk
from nltk.corpus import stopwords

# Diccionario de stopwords
try:
    STOP_WORDS_DICT = {
        'spanish': set(stopwords.words('spanish')),
        'english': set(stopwords.words('english'))
    }
except Exception as e:
    STOP_WORDS_DICT = {'spanish': set(), 'english': set()}

# ---------------------------------------------------------
# 1. PREPARACIÓN DE DATOS (NUEVO)
# ---------------------------------------------------------

def preparar_datos(ruta_archivo):
    """
    Carga el CSV crudo, limpia columnas basuras y convierte 'score' a 'sentiment'.
    """
    print(f"📂 Cargando datos desde: {ruta_archivo}")
    try:
        df = pd.read_csv(ruta_archivo)
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {ruta_archivo}")
        sys.exit(1)

    # Limpiar columnas 'Unnamed' que se cuelan al exportar
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Si existe 'score' pero no 'sentiment', hacemos la conversión automática
    if 'score' in df.columns and 'sentiment' not in df.columns:
        print("   ⚙️ Convirtiendo columna 'score' a 'sentiment'...")
        def map_sentiment(score):
            try:
                s = int(score)
                if s <= 2: return 'negativo'
                elif s == 3: return 'neutro'
                else: return 'positivo'
            except:
                return 'neutro'
        
        df['sentiment'] = df['score'].apply(map_sentiment)
    
    return df

# ---------------------------------------------------------
# CLASES Y FUNCIONES DE CLUSTERING
# ---------------------------------------------------------

class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, language='english'): # Pongo english por defecto para SoundCloud
        self.language = language
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        stop_words = STOP_WORDS_DICT.get(self.language, set())
        if isinstance(X, pd.Series):
            return X.apply(lambda t: self._clean_text(t, stop_words))
        return pd.Series(X).apply(lambda t: self._clean_text(t, stop_words))
    
    def _clean_text(self, text, stop_words):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        return " ".join([w for w in tokens if w not in stop_words])

def metodo_del_codo(X_tfidf, max_k, output_filename, polaridad):
    print(f"   📈 Calculando inercias para {polaridad} (K=2 hasta {max_k})...")
    inercias = []
    rango_k = range(2, max_k + 1)
    
    for k in rango_k:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(X_tfidf)
        inercias.append(kmeans.inertia_)
        
    plt.figure(figsize=(8, 5))
    plt.plot(rango_k, inercias, marker='o', linestyle='--', color='b')
    plt.title(f'Método del Codo - Reviews {polaridad.capitalize()}')
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inercia')
    plt.grid(True)
    
    os.makedirs('resultados', exist_ok=True)
    plt.savefig(output_filename)
    plt.close()
    print(f"   💾 Gráfico del codo guardado en: {output_filename}")

def obtener_palabras_significativas(modelo_kmeans, vectorizador, num_palabras=10):
    centroides = modelo_kmeans.cluster_centers_
    palabras_vocabulario = vectorizador.get_feature_names_out()
    
    topicos = {}
    for i in range(modelo_kmeans.n_clusters):
        indices_top = centroides[i].argsort()[::-1][:num_palabras]
        palabras_top = [palabras_vocabulario[ind] for ind in indices_top]
        topicos[f"Cluster_{i}"] = palabras_top
        
    return topicos

def analizar_polaridad(df, columna_texto, polaridad, k_optimo=4):
    print(f"\n{'-'*50}")
    print(f"ANALIZANDO REVIEWS: {polaridad.upper()}")
    print(f"{'-'*50}")
    
    print("   🧹 Limpiando textos...")
    cleaner = TextCleaner(language='english') # SoundCloud está en inglés
    textos_limpios = cleaner.transform(df[columna_texto])
    
    print("   🔢 Vectorizando con TF-IDF...")
    vectorizador = TfidfVectorizer(max_features=1000)
    X_tfidf = vectorizador.fit_transform(textos_limpios)
    
    grafico_path = f"resultados/codo_{polaridad}.png"
    metodo_del_codo(X_tfidf, max_k=10, output_filename=grafico_path, polaridad=polaridad)
    
    print(f"\n   🤖 Entrenando KMeans final con K={k_optimo}...")
    kmeans_final = KMeans(n_clusters=k_optimo, random_state=42, n_init='auto')
    kmeans_final.fit(X_tfidf)
    
    print("\n   🎯 PALABRAS SIGNIFICATIVAS POR TÓPICO:")
    topicos = obtener_palabras_significativas(kmeans_final, vectorizador, num_palabras=10)
    
    for cluster, palabras in topicos.items():
        print(f"      • {cluster}: {', '.join(palabras)}")
        
    df_resultado = df.copy()
    df_resultado['Cluster_Asignado'] = kmeans_final.labels_
    
    return df_resultado, topicos

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Script de Clustering para Análisis de Tópicos")
    # AQUÍ ES DONDE LE DECIMOS QUÉ ARCHIVO USAR:
    parser.add_argument('-i', '--input', type=str, required=True, help='Ruta al archivo CSV (ej: SoundCloud.csv)')
    parser.add_argument('-c_texto', '--col_texto', type=str, default='review', help='Nombre de la columna con el texto')
    parser.add_argument('-k_pos', '--k_positivos', type=int, default=4, help='Número de clusters para positivas')
    parser.add_argument('-k_neg', '--k_negativos', type=int, default=4, help='Número de clusters para negativas')
    args = parser.parse_args()
    
    print("\n" + "═"*50)
    print("SAD - PIPELINE DE CLUSTERING")
    print("═"*50)
    
    # 1. Preparar los datos dinámicamente
    data = preparar_datos(args.input)
    columna_texto = args.col_texto
    columna_target = 'sentiment'
    
    # 2. Filtrar DataFrames por polaridad
    df_positivos = data[data[columna_target].astype(str).str.lower() == 'positivo']
    df_negativos = data[data[columna_target].astype(str).str.lower() == 'negativo']
    
    print(f"\n📊 Distribución para clustering:")
    print(f"   - Positivos: {len(df_positivos)} reviews")
    print(f"   - Negativos: {len(df_negativos)} reviews")
    
    # 3. Ejecutar análisis
    df_pos_resultado, topicos_pos = analizar_polaridad(df_positivos, columna_texto, 'positivos', k_optimo=args.k_positivos)
    df_neg_resultado, topicos_neg = analizar_polaridad(df_negativos, columna_texto, 'negativos', k_optimo=args.k_negativos)
    
    # 4. Guardar resultados para Tableau
    print("\n" + "─"*50)
    print("GUARDANDO DATOS PARA TABLEAU")
    print("─"*50)
    
    os.makedirs('resultados', exist_ok=True)
    df_final_tableau = pd.concat([df_pos_resultado, df_neg_resultado])
    output_csv = 'resultados/datos_clustering_tableau.csv'
    df_final_tableau.to_csv(output_csv, index=False)
    
    print(f"✅ ¡Clustering completado!")
    print(f"📁 Datos exportados para Tableau en: {output_csv}")
    print(f"📁 Gráficos del codo guardados en la carpeta 'resultados/'")

if __name__ == '__main__':
    main()