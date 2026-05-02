"""
Funciones auxiliares para la plantilla híbrida de Machine Learning.
"""

import json
import sys
import pandas as pd


def load_config(config_file: str, sections: list = None) -> dict:
    """
    Carga la configuración desde el archivo JSON.
    
    Args:
        config_file: Ruta al archivo de configuración JSON
        sections: Lista de secciones a cargar. Si es None, carga todo.
                 Opciones: ['general', 'preprocessing', 'train', 'test']
    
    Returns:
        dict: Diccionario con la configuración solicitada
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_completa = json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {config_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error al parsear JSON: {e}")
        sys.exit(1)
    
    if sections is None:
        return config_completa
    
    config = {}
    
    # Siempre incluir 'general' si se especifican secciones
    if 'general' in sections or 'general' in config_completa:
        general = config_completa.get('general', {})
        for key, value in general.items():
            if key == 'data' and isinstance(value, dict):
                for data_key, data_value in value.items():
                    config[data_key] = data_value
            else:
                config[key] = value
    
    # Cargar secciones específicas
    for section in sections:
        if section == 'general':
            continue  # Ya procesado
        
        section_data = config_completa.get(section, {})
        
        if section == 'train':
            # Filtrar solo modelos activos
            for key, value in section_data.items():
                if key == 'modelos' and isinstance(value, list):
                    modelos_activos = []
                    for modelo in value:
                        # Buscar si algún modelo está en true
                        es_activo = any(
                            v is True for k, v in modelo.items() 
                            if k not in ['parametros', 'modelo_output']
                        )
                        if es_activo:
                            modelos_activos.append(modelo)
                    config['modelos'] = modelos_activos
                else:
                    config[key] = value
        else:
            # Para otras secciones, copiar todo
            for key, value in section_data.items():
                config[key] = value
    
    return config


def load_data(filepath: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Carga un archivo CSV y limpia columnas innecesarias.
    
    Args:
        filepath: Ruta al archivo CSV
        encoding: Codificación del archivo (default: utf-8)
    
    Returns:
        pd.DataFrame: DataFrame con los datos cargados
    """
    try:
        data = pd.read_csv(filepath, encoding=encoding)
    except UnicodeDecodeError:
        print(f"Intentando con encoding 'latin1'...")
        data = pd.read_csv(filepath, encoding='latin1')
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {filepath}")
        sys.exit(1)
    
    # Eliminar columnas 'Unnamed' que a veces genera pandas
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    
    print(f"✓ Datos cargados: {data.shape[0]} filas, {data.shape[1]} columnas")
    
    return data


def procesar_parametros(params_dict: dict) -> dict:
    """
    Procesa parámetros del JSON convirtiendo rangos {min, max, step} en listas.
    Esta es la función brillante de Eder que automatiza la generación de rangos.
    
    Args:
        params_dict: Diccionario con parámetros que pueden contener rangos
    
    Returns:
        dict: Diccionario con rangos expandidos a listas
    
    Ejemplo:
        Input:  {"n_neighbors": {"min": 1, "max": 15, "step": 2}}
        Output: {"n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15]}
    """
    params_procesados = {}
    
    for clave, valor in params_dict.items():
        # Si es un diccionario con min/max/step, expandir a lista
        if isinstance(valor, dict) and all(k in valor for k in ['min', 'max', 'step']):
            params_procesados[clave] = list(range(
                valor['min'],
                valor['max'] + 1,  # +1 para incluir el máximo
                valor['step']
            ))
        else:
            # Si no, mantener el valor original (listas, valores únicos, etc.)
            params_procesados[clave] = valor
    
    return params_procesados


def get_model_name(modelo_dict: dict) -> str:
    """
    Extrae el nombre del modelo del diccionario de configuración.
    
    Args:
        modelo_dict: Diccionario con la configuración de un modelo
    
    Returns:
        str: Nombre del modelo (e.g., 'knn', 'random_forest')
    """
    for key in modelo_dict.keys():
        if key not in ['parametros', 'modelo_output'] and modelo_dict[key] is True:
            return key
    return None


def print_section_header(title: str, char: str = "=", length: int = 70):
    """
    Imprime un encabezado de sección formateado.
    
    Args:
        title: Título de la sección
        char: Carácter para el borde
        length: Longitud total de la línea
    """
    print(f"\n{char * length}")
    print(f"{title:^{length}}")
    print(f"{char * length}\n")
