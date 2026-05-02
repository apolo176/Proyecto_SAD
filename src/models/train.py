#!/usr/bin/env python3
"""
Script de entrenamiento con arquitectura de pipelines robusta.
Combina lo mejor de ambas plantillas: modularidad + validación matemática correcta.

CARACTERÍSTICAS CLAVE:
- División train/dev ANTES del preprocesamiento (evita data leakage)
- Pipelines de sklearn para garantizar flujo correcto
- OneHotEncoder para variables categóricas (matemáticamente correcto)
- Procesamiento de texto con stopwords en español
- Entrenamiento en batch de múltiples modelos
- Automatización de rangos de hiperparámetros (min/max/step)
"""

import argparse
import pickle
import os
import sys
import pandas as pd

# Sklearn imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, CategoricalNB
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.base import BaseEstimator, TransformerMixin

# Imblearn imports
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# NLTK para procesamiento de texto
import nltk
from nltk.corpus import stopwords
import re

# Funciones propias
from src.utils.funciones import load_config, load_data, procesar_parametros, get_model_name, print_section_header

# Descargar recursos de NLTK si no están disponibles
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("📥 Descargando recursos de NLTK...")
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

# NUEVO: Extraer stopwords a un diccionario nativo de Python para evitar
# que joblib intente serializar el CorpusReader de NLTK
try:
    STOP_WORDS_DICT = {
        'spanish': set(stopwords.words('spanish')),
        'english': set(stopwords.words('english'))
    }
except Exception as e:
    print(f"⚠️ Error cargando stopwords: {e}")
    STOP_WORDS_DICT = {'spanish': set(), 'english': set()}

        
class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Limpiador de texto personalizado que se integra en el pipeline.
    Basado en la implementación de Eder, mejorado para español.
    """
    
    def __init__(self, language='spanish'):
        self.language = language
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """
        Limpia el texto: minúsculas, sin caracteres especiales, sin stopwords.
        """
        # Usamos el diccionario nativo global. 
        # ¡Cero llamadas a NLTK aquí adentro para evitar el crash de Loky!
        stop_words = STOP_WORDS_DICT.get(self.language, set())
            
        if isinstance(X, pd.Series):
            return X.apply(lambda t: self._clean_text(t, stop_words))
        elif isinstance(X, pd.DataFrame):
            return X.iloc[:, 0].apply(lambda t: self._clean_text(t, stop_words))
        else:
            return pd.Series(X).apply(lambda t: self._clean_text(t, stop_words))
    
    def _clean_text(self, text, stop_words):
        """Limpia un texto individual."""
        if not isinstance(text, str):
            return ""
        
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        
        filtered_tokens = [word for word in tokens if word not in stop_words]
        return " ".join(filtered_tokens)       

class DenseTransformer(BaseEstimator, TransformerMixin):
    """
    Convierte matrices sparse a dense.
    Necesario para algunos algoritmos como GaussianNB.
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if hasattr(X, "toarray"):
            return X.toarray()
        return X


def crear_pipeline(modelo_nombre: str, X_train: pd.DataFrame, config: dict):
    """
    Crea un pipeline completo de preprocesamiento + modelo.
    
    ARQUITECTURA DEL PIPELINE:
    1. ColumnTransformer que procesa cada tipo de columna:
       - Numéricas: Imputer + Scaler
       - Categóricas: Imputer + OneHotEncoder (CORRECTO, no LabelEncoder)
       - Texto: TextCleaner + TF-IDF/BoW
    2. Balanceo de clases (opcional)
    3. Modelo de clasificación
    
    Args:
        modelo_nombre: Nombre del modelo ('knn', 'random_forest', etc.)
        X_train: DataFrame de entrenamiento (solo features, sin target)
        config: Diccionario de configuración
    
    Returns:
        Pipeline completo listo para GridSearchCV
    """
    
    # 1. IDENTIFICAR TIPOS DE COLUMNAS
    text_cols = config.get('text_features', [])
    
    # Columnas numéricas (int/float) que NO son texto
    num_cols = [
        col for col in X_train.select_dtypes(include=['int64', 'float64']).columns
        if col not in text_cols
    ]
    
    # Columnas categóricas (object/category/str) que NO son texto
    cat_cols = [
        col for col in X_train.select_dtypes(include=['object', 'category', 'str']).columns
        if col not in text_cols
    ]

    print(f"📊 Tipos de columnas detectadas:")
    print(f"   - Numéricas: {len(num_cols)} {num_cols if num_cols else ''}")
    print(f"   - Categóricas: {len(cat_cols)} {cat_cols if cat_cols else ''}")
    print(f"   - Texto: {len(text_cols)} {text_cols if text_cols else ''}")
    
    # 2. CREAR TRANSFORMADORES PARA CADA TIPO
    transformers = []
    
    # Pipeline para columnas numéricas
    if num_cols:
        impute_strategy = config.get('impute_strategy_numeric', 'mean')
        scaling = config.get('scaling', 'minmax').lower()
        
        scaler = MinMaxScaler() if scaling == 'minmax' else StandardScaler()
        
        num_pipeline = ImbPipeline([
            ('imputer', SimpleImputer(strategy=impute_strategy)),
            ('scaler', scaler)
        ])
        
        transformers.append(('num', num_pipeline, num_cols))
    
    # Pipeline para columnas categóricas - USA ONEHOT, NO LABEL
    if cat_cols:
        impute_strategy = config.get('impute_strategy_categorical', 'most_frequent')
        
        cat_pipeline = ImbPipeline([
            ('imputer', SimpleImputer(strategy=impute_strategy)),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        transformers.append(('cat', cat_pipeline, cat_cols))
        print("   ✓ Usando OneHotEncoder para categóricas (correcto)")
    
    # Pipeline para cada columna de texto
    text_process = config.get('text_process', 'tf_idf')
    language = config.get('language', 'spanish')
    
    for col in text_cols:
        if col in X_train.columns:
            # Vectorizador según configuración
            if text_process == 'tf_idf':
                vectorizer = TfidfVectorizer(max_features=1000)
            else:
                vectorizer = CountVectorizer(max_features=1000)
            
            text_pipeline = ImbPipeline([
                ('cleaner', TextCleaner(language=language)),
                ('vectorizer', vectorizer)
            ])
            
            transformers.append((f'text_{col}', text_pipeline, col))
    
    if not transformers:
        raise ValueError("❌ No se detectaron columnas para procesar. Revisa los datos y la configuración.")
    
    # 3. CREAR PREPROCESSOR
    preprocessor = ColumnTransformer(transformers=transformers)
    
    # 4. SELECCIONAR MODELO
    if modelo_nombre == 'knn':
        modelo = KNeighborsClassifier()
    elif modelo_nombre == 'decision_tree':
        modelo = DecisionTreeClassifier(random_state=config.get('random_state', 42))
    elif modelo_nombre == 'random_forest':
        modelo = RandomForestClassifier(random_state=config.get('random_state', 42))
    elif modelo_nombre == 'multinomial_nb':
        modelo = MultinomialNB()
    elif modelo_nombre == 'categorical_nb':
        # CategoricalNB requiere discretización de variables numéricas
        # Modificar el preprocessor para este caso específico
        preprocessor = crear_preprocessor_categorical_nb(X_train, config)
        modelo = CategoricalNB()
    else:
        raise ValueError(f"❌ Modelo '{modelo_nombre}' no soportado")
    
    # 5. SELECCIONAR BALANCEO
    sampling = config.get('sampling', None)
    sampler = None
    
    if sampling == 'oversampling':
        sampler = RandomOverSampler(random_state=config.get('random_state', 42))
        print(f"   ✓ Oversampling activado")
    elif sampling == 'undersampling':
        sampler = RandomUnderSampler(random_state=config.get('random_state', 42))
        print(f"   ✓ Undersampling activado")
    
    # 6. CONSTRUIR PIPELINE FINAL
    pasos = [('preprocessor', preprocessor)]
    
    if sampler is not None:
        pasos.append(('sampler', sampler))
    
    # Para MultinomialNB, asegurar que la salida sea densa y no negativa
    if modelo_nombre == 'multinomial_nb':
        pasos.append(('to_dense', DenseTransformer()))
    
    pasos.append(('clasificador', modelo))
    
    pipeline = ImbPipeline(steps=pasos)
    
    return pipeline


def crear_preprocessor_categorical_nb(X_train: pd.DataFrame, config: dict):
    """
    Crea un preprocessor especial para CategoricalNB.
    Este modelo requiere que TODAS las features sean categóricas.
    """
    text_cols = config.get('text_features', [])
    
    num_cols = [
        col for col in X_train.select_dtypes(include=['int64', 'float64']).columns
        if col not in text_cols
    ]
    
    cat_cols = [
        col for col in X_train.select_dtypes(include=['object', 'category', 'str']).columns
        if col not in text_cols
    ]

    transformers = []
    
    # Discretizar columnas numéricas
    if num_cols:
        num_pipeline = ImbPipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('discretizer', KBinsDiscretizer(encode='ordinal', strategy='uniform'))
        ])
        transformers.append(('num', num_pipeline, num_cols))
    
    # LabelEncoder para categóricas (en este caso específico sí es correcto)
    if cat_cols:
        from sklearn.preprocessing import OrdinalEncoder
        cat_pipeline = ImbPipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        transformers.append(('cat', cat_pipeline, cat_cols))
    
    # Texto como categórico mediante hashing
    for col in text_cols:
        if col in X_train.columns:
            from sklearn.feature_extraction.text import HashingVectorizer
            text_pipeline = ImbPipeline([
                ('cleaner', TextCleaner(language=config.get('language', 'spanish'))),
                ('hash', HashingVectorizer(n_features=100, alternate_sign=False))
            ])
            transformers.append((f'text_{col}', text_pipeline, col))
    
    return ColumnTransformer(transformers=transformers)


def entrenar_modelo(modelo_config: dict, X_train, X_dev, y_train, y_dev, config: dict):
    """
    Entrena un modelo específico con GridSearchCV.
    
    Args:
        modelo_config: Configuración del modelo desde el JSON
        X_train, X_dev, y_train, y_dev: Conjuntos de datos
        config: Configuración general
    """
    # Obtener nombre del modelo
    modelo_nombre = get_model_name(modelo_config)
    if not modelo_nombre:
        print("⚠️ No se pudo determinar el nombre del modelo, saltando...")
        return
    
    print_section_header(f"ENTRENANDO: {modelo_nombre.upper()}")
    
    # Crear pipeline
    try:
        pipeline = crear_pipeline(modelo_nombre, X_train, config)
    except Exception as e:
        print(f"❌ Error creando pipeline para {modelo_nombre}: {e}")
        return
    
    # Procesar parámetros (expandir rangos min/max/step)
    params_raw = modelo_config.get('parametros', {})
    params = procesar_parametros(params_raw)
    
    print(f"🔍 Configuración de GridSearchCV:")
    print(f"   - Parámetros a probar: {len(list(params.values())[0]) if params else 0} combinaciones por parámetro")
    print(f"   - Cross-validation: {config.get('cv_folds', 5)} folds")
    print(f"   - Scoring: {config.get('scoring', 'f1_macro')}")
    print(f"   - CPUs: {config.get('cpu', -1)}")

    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        cv=config.get('cv_folds', 5),
        scoring=config.get('scoring', 'f1_macro'),
        n_jobs=config.get('cpu', -1),
        verbose=3
    )
    
    # Entrenar
    print("\n🚀 Iniciando búsqueda de hiperparámetros...")
    grid_search.fit(X_train, y_train)
    
    # Resultados
    print(f"\n✅ Entrenamiento completado!")
    print(f"   - Mejores parámetros: {grid_search.best_params_}")
    print(f"   - Mejor score (CV): {grid_search.best_score_:.4f}")
    
    # Evaluar en Dev
    y_pred = grid_search.predict(X_dev)
    from sklearn.metrics import f1_score, accuracy_score, classification_report
    
    f1_macro = f1_score(y_dev, y_pred, average='macro')
    f1_micro = f1_score(y_dev, y_pred, average='micro')
    accuracy = accuracy_score(y_dev, y_pred)
    
    print(f"\n📊 Resultados en Dev:")
    print(f"   - F1 Macro: {f1_macro:.4f}")
    print(f"   - F1 Micro: {f1_micro:.4f}")
    print(f"   - Accuracy: {accuracy:.4f}")
    
    # Guardar modelo
    output_path = modelo_config.get('modelo_output', f'modelos/{modelo_nombre}_BestModel.pkl')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(grid_search.best_estimator_, f)
    
    print(f"💾 Modelo guardado en: {output_path}")
    
    # Guardar resultados de CV
    resultados_cv = pd.DataFrame(grid_search.cv_results_)
    cv_output = output_path.replace('.pkl', '_cv_results.csv')
    resultados_cv.to_csv(cv_output, index=False)
    print(f"💾 Resultados de CV guardados en: {cv_output}")
    
    return grid_search.best_estimator_


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento de modelos con arquitectura híbrida (modular + pipelines robustos)"
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='config.json',
        help='Archivo de configuración JSON'
    )
    
    args = parser.parse_args()
    
    print_section_header("PLANTILLA HÍBRIDA - ENTRENAMIENTO", char="═")
    
    # Cargar configuración
    print("📖 Cargando configuración...")
    config = load_config(args.config, sections=['general', 'preprocessing', 'train'])
    
    # Cargar datos
    print("\n📂 Cargando datos...")
    data = load_data(config['train_dev'])
    
    # Separar X e y
    target_col = config['column']
    if target_col not in data.columns:
        print(f"❌ Error: columna '{target_col}' no encontrada en los datos")
        sys.exit(1)
    
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    # Eliminar columnas innecesarias
    drop_cols = config.get('drop_features', [])
    if drop_cols:
        drop_cols = [col for col in drop_cols if col in X.columns]
        X = X.drop(columns=drop_cols)
        print(f"🗑️ Columnas eliminadas: {drop_cols}")
    
    # Codificar variable objetivo si es categórica
    if not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        y = le.fit_transform(y)
        
        # Guardar encoder para usar en test
        os.makedirs('modelos', exist_ok=True)
        with open('modelos/label_encoder_y.pkl', 'wb') as f:
            pickle.dump(le, f)
        print(f"💾 LabelEncoder guardado para el target")
        print(f"   Clases: {le.classes_}")

        # --- NUEVA LÓGICA: DETECCIÓN DE DATASETS PRE-DIVIDIDOS ---
        # Comprobamos si el usuario ha definido un 'dev_file' explícito en el config.json
        dev_file = config.get('data', {}).get('dev_file', None)

        if dev_file and os.path.exists(dev_file):
            print(f"\n🧠 Modo IA Generativa Detectado: Usando splits pre-definidos")
            print(f"   - Train: Cargado desde {config['data']['train_dev']}")
            print(f"   - Dev: Cargando desde {dev_file}")

            data_dev = load_data(dev_file)

            # El archivo principal (X, y) ya es puramente Train
            X_train = X
            y_train = y

            X_dev = data_dev.drop(columns=[target_col])
            y_dev = data_dev[target_col]

            # Eliminar drop_features del Dev si las hay
            if drop_cols:
                drop_cols_dev = [col for col in drop_cols if col in X_dev.columns]
                X_dev = X_dev.drop(columns=drop_cols_dev)

            # Si la Y se codificó, codificamos la Y de Dev también
            if not pd.api.types.is_numeric_dtype(y_dev):
                y_dev = le.transform(y_dev)

        else:
            # MODO TRADICIONAL: DIVISIÓN TRAIN/DEV INTERNA
            print(f"\n✂️ Modo Tradicional: Dividiendo datos (Train/Dev)...")

            dev_size = config.get('dev_size', 0.25)
            random_state = config.get('random_state', 42)

            X_train, X_dev, y_train, y_dev = train_test_split(
                X, y,
                test_size=dev_size,
                stratify=y,
                random_state=random_state
            )
    
    print(f"   - Train: {X_train.shape[0]} muestras")
    print(f"   - Dev: {X_dev.shape[0]} muestras")
    print(f"   - Proporción: {(1-dev_size)*100:.0f}% / {dev_size*100:.0f}%")
    
    # Entrenar cada modelo activo
    modelos = config.get('modelos', [])
    
    if not modelos:
        print("⚠️ No hay modelos activos para entrenar. Revisa config.json")
        sys.exit(1)
    
    print(f"\n🤖 Modelos a entrenar: {len(modelos)}")
    
    for i, modelo_config in enumerate(modelos, 1):
        print(f"\n{'='*70}")
        print(f"MODELO {i}/{len(modelos)}")
        print(f"{'='*70}")
        
        try:
            entrenar_modelo(modelo_config, X_train, X_dev, y_train, y_dev, config)
        except Exception as e:
            modelo_nombre = get_model_name(modelo_config)
            print(f"\n❌ Error entrenando {modelo_nombre}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print_section_header("ENTRENAMIENTO COMPLETADO", char="═")
    print("✅ Todos los modelos han sido entrenados y guardados")
    print("📁 Revisa la carpeta 'modelos/' para los archivos .pkl")
    print("\n💡 Próximo paso: ejecutar test.py para evaluar en el conjunto de test")


if __name__ == '__main__':
    main()
