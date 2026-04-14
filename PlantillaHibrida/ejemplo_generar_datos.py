#!/usr/bin/env python3
"""
Script de ejemplo para probar la PlantillaHíbrida con datos sintéticos.
Genera un dataset de reviews de películas con sentimientos (positivo/negativo/neutro).
"""

import pandas as pd
import numpy as np
import os

# Configurar seed para reproducibilidad
np.random.seed(42)


def generar_datos_ejemplo(n_samples=1000):
    """
    Genera un dataset sintético de reviews de películas.
    
    Returns:
        pd.DataFrame: DataFrame con reviews y sentimientos
    """
    
    # Palabras positivas
    palabras_positivas = [
        "excelente", "increíble", "fantástico", "maravilloso", "genial",
        "brillante", "espectacular", "magnífico", "impresionante", "perfecto",
        "hermoso", "fascinante", "emocionante", "divertido", "entretenido"
    ]
    
    # Palabras negativas
    palabras_negativas = [
        "terrible", "horrible", "malo", "pésimo", "aburrido",
        "decepcionante", "mediocre", "insulso", "tedioso", "lento",
        "predecible", "confuso", "frustrante", "desastroso", "deficiente"
    ]
    
    # Palabras neutras
    palabras_neutras = [
        "regular", "normal", "aceptable", "promedio", "estándar",
        "correcto", "pasable", "común", "típico", "ordinario"
    ]
    
    # Géneros de películas
    generos = ["acción", "comedia", "drama", "terror", "ciencia ficción", 
               "romance", "thriller", "aventura"]
    
    # Años
    años = list(range(2015, 2024))
    
    datos = []
    
    for i in range(n_samples):
        # Determinar sentimiento (distribución desbalanceada a propósito)
        sentimiento = np.random.choice(
            ['positivo', 'negativo', 'neutro'],
            p=[0.5, 0.3, 0.2]  # 50% positivo, 30% negativo, 20% neutro
        )
        
        # Generar review basado en el sentimiento
        if sentimiento == 'positivo':
            palabras_base = np.random.choice(palabras_positivas, size=np.random.randint(3, 6))
            review = f"La película fue {' y '.join(palabras_base)}. Recomendada."
        elif sentimiento == 'negativo':
            palabras_base = np.random.choice(palabras_negativas, size=np.random.randint(3, 6))
            review = f"La película fue {' y '.join(palabras_base)}. No recomendada."
        else:
            palabras_base = np.random.choice(palabras_neutras, size=np.random.randint(2, 4))
            review = f"La película fue {' y '.join(palabras_base)}. Puede verse."
        
        # Generar características adicionales
        genero = np.random.choice(generos)
        año = np.random.choice(años)
        duracion = np.random.randint(80, 180)
        rating_imdb = np.random.uniform(3.0, 9.0)
        
        # Añadir algo de ruido en el rating correlacionado con sentimiento
        if sentimiento == 'positivo':
            rating_imdb = min(10.0, rating_imdb + np.random.uniform(0.5, 2.0))
        elif sentimiento == 'negativo':
            rating_imdb = max(1.0, rating_imdb - np.random.uniform(0.5, 2.0))
        
        datos.append({
            'review_id': f'REV_{i:04d}',
            'review_text': review,
            'sentiment': sentimiento,
            'genre': genero,
            'year': año,
            'duration_min': duracion,
            'imdb_rating': round(rating_imdb, 1)
        })
    
    return pd.DataFrame(datos)


def main():
    """Función principal."""
    
    print("🎬 Generando datos de ejemplo para PlantillaHíbrida...")
    print("="*70)
    
    # Crear carpeta data si no existe
    os.makedirs('data', exist_ok=True)
    
    # Generar dataset completo
    print("\n📊 Generando 1500 reviews...")
    df_completo = generar_datos_ejemplo(n_samples=1500)
    
    # Dividir en train y test (80/20)
    from sklearn.model_selection import train_test_split
    
    df_train, df_test = train_test_split(
        df_completo,
        test_size=0.2,
        stratify=df_completo['sentiment'],
        random_state=42
    )
    
    # Guardar
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    print(f"✅ Datos guardados:")
    print(f"   📁 {train_path}: {len(df_train)} muestras")
    print(f"   📁 {test_path}: {len(df_test)} muestras")
    
    # Mostrar distribución
    print("\n📊 Distribución de sentimientos (train):")
    print(df_train['sentiment'].value_counts())
    print(f"\n📊 Distribución de sentimientos (test):")
    print(df_test['sentiment'].value_counts())
    
    # Mostrar ejemplos
    print("\n📝 Ejemplos de reviews:")
    print("="*70)
    for sentiment in ['positivo', 'negativo', 'neutro']:
        ejemplo = df_train[df_train['sentiment'] == sentiment].iloc[0]
        print(f"\n{sentiment.upper()}:")
        print(f"  Review: {ejemplo['review_text']}")
        print(f"  Género: {ejemplo['genre']}, Año: {ejemplo['year']}")
        print(f"  IMDB: {ejemplo['imdb_rating']}, Duración: {ejemplo['duration_min']}min")
    
    print("\n" + "="*70)
    print("✅ Datos de ejemplo generados exitosamente!")
    print("\n💡 Próximos pasos:")
    print("   1. Revisar config.json (ya configurado para estos datos)")
    print("   2. Ejecutar: python train.py")
    print("   3. Ejecutar: python test.py")
    print("\n🚀 ¡Listo para empezar!")


if __name__ == '__main__':
    main()
