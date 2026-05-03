#!/usr/bin/env python3
"""
Script de evaluación de modelos entrenados.
Carga modelos desde archivos .pkl y los evalúa en el conjunto de test.
"""

import argparse
import pickle
import os
import sys
import pandas as pd
from pathlib import Path
from glob import glob

import re
import nltk
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin

# Sklearn imports
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

# Funciones propias
from src.utils.funciones import load_config, load_data, print_section_header


try:
    STOP_WORDS_DICT = {
        'spanish': set(stopwords.words('spanish')),
        'english': set(stopwords.words('english'))
    }
except Exception as e:
    STOP_WORDS_DICT = {'spanish': set(), 'english': set()}

# --- PEGAR LAS CLASES PERSONALIZADAS AQUÍ ---
class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, language='spanish'):
        self.language = language
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        stop_words = STOP_WORDS_DICT.get(self.language, set())
            
        if isinstance(X, pd.Series):
            return X.apply(lambda t: self._clean_text(t, stop_words))
        elif isinstance(X, pd.DataFrame):
            return X.iloc[:, 0].apply(lambda t: self._clean_text(t, stop_words))
        else:
            return pd.Series(X).apply(lambda t: self._clean_text(t, stop_words))
    
    def _clean_text(self, text, stop_words):
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        
        filtered_tokens = [word for word in tokens if word not in stop_words]
        return " ".join(filtered_tokens)

class DenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if hasattr(X, "toarray"):
            return X.toarray()
        return X

def cargar_modelo(ruta_modelo: str):
    """
    Carga un modelo desde un archivo pickle.
    
    Args:
        ruta_modelo: Ruta al archivo .pkl del modelo
    
    Returns:
        Modelo cargado o None si hay error
    """
    try:
        with open(ruta_modelo, 'rb') as f:
            modelo = pickle.load(f)
        return modelo
    except FileNotFoundError:
        print(f"❌ No se encontró el archivo: {ruta_modelo}")
        return None
    except Exception as e:
        print(f"❌ Error cargando {ruta_modelo}: {e}")
        return None


def evaluar_modelo(modelo, X_test, y_test, nombre_modelo: str):
    """
    Evalúa un modelo en el conjunto de test.
    
    Args:
        modelo: Modelo entrenado
        X_test: Features de test
        y_test: Labels de test
        nombre_modelo: Nombre del modelo para identificación
    
    Returns:
        dict: Diccionario con todas las métricas
    """
    print(f"\n{'─'*70}")
    print(f"📊 Evaluando: {nombre_modelo}")
    print(f"{'─'*70}")
    
    # Realizar predicciones
    try:
        y_pred = modelo.predict(X_test)
    except Exception as e:
        print(f"❌ Error al predecir: {e}")
        return None
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    
    # F1 scores
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_test, y_pred, average='micro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    # Precision y Recall
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Imprimir resultados
    print(f"\n📈 Métricas Globales:")
    print(f"   • Accuracy:        {accuracy:.4f}")
    print(f"   • F1 Macro:        {f1_macro:.4f}")
    print(f"   • F1 Micro:        {f1_micro:.4f}")
    print(f"   • F1 Weighted:     {f1_weighted:.4f}")
    print(f"   • Precision Macro: {precision_macro:.4f}")
    print(f"   • Recall Macro:    {recall_macro:.4f}")
    
    print(f"\n📊 F1 por Clase:")
    for i, f1 in enumerate(f1_per_class):
        print(f"   • Clase {i}: {f1:.4f}")
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🔢 Matriz de Confusión:")
    print(cm)
    
    # Reporte de clasificación
    print(f"\n📋 Reporte de Clasificación:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Retornar métricas en un diccionario
    metricas = {
        'modelo': nombre_modelo,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_per_class': f1_per_class.tolist()
    }
    
    return metricas, y_pred


def main():
    """Función principal."""
    
    parser = argparse.ArgumentParser(
        description="Evaluación de modelos en el conjunto de test"
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='config.json',
        help='Archivo de configuración JSON'
    )
    parser.add_argument(
        '-m', '--models',
        type=str,
        nargs='+',
        help='Rutas específicas a modelos .pkl (opcional, por defecto usa todos en modelos/)'
    )
    
    args = parser.parse_args()
    
    print_section_header("PLANTILLA HÍBRIDA - EVALUACIÓN", char="═")
    
    # Cargar configuración
    print("📖 Cargando configuración...")
    config = load_config(args.config, sections=['general', 'test'])
    
    # Cargar datos de test
    print("\n📂 Cargando datos de test...")
    test_file = config.get('test')
    
    if not test_file or not os.path.exists(test_file):
        print(f"❌ Error: archivo de test '{test_file}' no encontrado")
        sys.exit(1)
    
    data_test = load_data(test_file)
    
    # Separar X e y
    target_col = config.get('column')
    if target_col not in data_test.columns:
        print(f"❌ Error: columna '{target_col}' no encontrada en test")
        sys.exit(1)
    
    X_test = data_test.drop(columns=[target_col])
    y_test = data_test[target_col]
    
    # Eliminar columnas innecesarias
    drop_cols = config.get('drop_features', [])
    if drop_cols:
        drop_cols = [col for col in drop_cols if col in X_test.columns]
        X_test = X_test.drop(columns=drop_cols)
    
    # Decodificar variable objetivo si fue codificada en entrenamiento
    le_path = 'modelos/label_encoder_y.pkl'
    if os.path.exists(le_path):
        with open(le_path, 'rb') as f:
            le = pickle.load(f)
        
        if not pd.api.types.is_numeric_dtype(y_test):
            y_test_encoded = le.transform(y_test)
            print(f"✓ Variable objetivo decodificada usando LabelEncoder guardado")
        else:
            y_test_encoded = y_test
    else:
        y_test_encoded = y_test
    
    print(f"   - Muestras de test: {X_test.shape[0]}")
    print(f"   - Features: {X_test.shape[1]}")
    
    # Buscar modelos a evaluar
    if args.models:
        # Usar modelos específicos proporcionados
        modelos_paths = args.models
    else:
        # Buscar todos los .pkl en la carpeta modelos/
        modelos_paths = glob('modelos/*_BestModel.pkl')
        
        if not modelos_paths:
            print("❌ No se encontraron modelos en modelos/")
            print("💡 Asegúrate de haber ejecutado train.py primero")
            sys.exit(1)
    
    print(f"\n🤖 Modelos a evaluar: {len(modelos_paths)}")
    for path in modelos_paths:
        print(f"   • {path}")
    
    # Evaluar cada modelo
    resultados = []
    predicciones = {}
    
    for modelo_path in modelos_paths:
        nombre_modelo = Path(modelo_path).stem.replace('_BestModel', '')
        
        # Cargar modelo
        modelo = cargar_modelo(modelo_path)
        if modelo is None:
            continue
        
        # Evaluar
        try:
            metricas, y_pred = evaluar_modelo(modelo, X_test, y_test_encoded, nombre_modelo)
            if metricas:
                resultados.append(metricas)
                predicciones[nombre_modelo] = y_pred
        except Exception as e:
            print(f"❌ Error evaluando {nombre_modelo}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Guardar resultados
    if resultados:
        print_section_header("GUARDANDO RESULTADOS", char="─")
        
        # Crear DataFrame de resultados
        df_metricas = pd.DataFrame(resultados)
        
        # Ordenar por F1 Macro (descendente)
        df_metricas = df_metricas.sort_values('f1_macro', ascending=False)
        
        # Guardar métricas
        os.makedirs('resultados', exist_ok=True)
        metricas_output = config.get('metricas_output', 'resultados/metricas_modelos.csv')
        df_metricas.to_csv(metricas_output, index=False)
        print(f"💾 Métricas guardadas en: {metricas_output}")
        
        # Mostrar ranking
        print(f"\n🏆 RANKING DE MODELOS (por F1 Macro):")
        print(f"{'─'*70}")
        for i, row in df_metricas.iterrows():
            print(f"{row['modelo']:30s} | F1 Macro: {row['f1_macro']:.4f} | Accuracy: {row['accuracy']:.4f}")
        print(f"{'─'*70}")
        
        # Guardar predicciones
        if predicciones:
            # Decodificar predicciones si es necesario
            if os.path.exists(le_path):
                for modelo_nombre in predicciones:
                    predicciones[modelo_nombre] = le.inverse_transform(predicciones[modelo_nombre])
            
            df_pred = data_test.copy()
            for modelo_nombre, preds in predicciones.items():
                df_pred[f'pred_{modelo_nombre}'] = preds
            
            pred_output = config.get('predicciones_output', 'resultados/predicciones_test.csv')
            df_pred.to_csv(pred_output, index=False)
            print(f"💾 Predicciones guardadas en: {pred_output}")
    
    print_section_header("EVALUACIÓN COMPLETADA", char="═")
    print("✅ Evaluación finalizada")
    print(f"📁 Resultados guardados en la carpeta 'resultados/'")


if __name__ == '__main__':
    main()
