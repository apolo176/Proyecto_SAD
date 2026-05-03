import pandas as pd

# Definimos las parejas de archivos: (Archivo de Resultados, Archivo de Tópicos con descripciones)
parejas_archivos = [
    ('./AppleMusic/General/resultados_tableau_AppleMusic_General.csv', './AppleMusic/General/topicos_AppleMusic_General.csv'),
    ('./AppleMusic/2015-2018/resultados_tableau_AppleMusic_2015_2018.csv', './AppleMusic/2015-2018/topicos_AppleMusic_2015_2018.csv'),
    ('./AppleMusic/2019-2022/resultados_tableau_AppleMusic_2019_2022.csv', './AppleMusic/2019-2022/topicos_AppleMusic_2019_2022.csv'),
    ('./AppleMusic/2023-2025/resultados_tableau_AppleMusic_2023_2025.csv', './AppleMusic/2023-2025/topicos_AppleMusic_2023_2025.csv'),
    ('./Spotify/General/resultados_tableau_Spotify_General.csv', './Spotify/General/topicos_Spotify_General.csv'),
    ('./Spotify/2008-2014/resultados_tableau_Spotify_2008_2014.csv', './Spotify/2008-2014/topicos_Spotify_2008_2014.csv'),
    ('./Spotify/2015-2018/resultados_tableau_Spotify_2015_2018.csv', './Spotify/2015-2018/topicos_Spotify_2015_2018.csv'),
    ('./Spotify/2019-2022/resultados_tableau_Spotify_2019_2022.csv', './Spotify/2019-2022/topicos_Spotify_2019_2022.csv'),
    ('./Spotify/2023-2025/resultados_tableau_Spotify_2023_2025.csv', './Spotify/2023-2025/topicos_Spotify_2023_2025.csv'),
]

lista_dataframes_finales = []

for archivo_res, archivo_top in parejas_archivos:
    try:
        # 1. Cargar ambos datasets
        df_res = pd.read_csv(archivo_res)
        df_top = pd.read_csv(archivo_top)
        
        # 2. Hacer el cruce (Merge) usando las dos claves que comparten
        df_cruzado = pd.merge(df_res, df_top[['Polaridad_Clustering', 'Topic_ID', 'Descripcion']], 
                              on=['Polaridad_Clustering', 'Topic_ID'], 
                              how='left')
        
        # 3. Extraer Empresa y Periodo del nombre del archivo para no perder el rastro
        df_cruzado['Empresa'] = 'Apple Music' if 'Apple' in archivo_res else 'Spotify'
        
        if 'General' in archivo_res:
            df_cruzado['Periodo'] = 'General'
        elif '2008_2014' in archivo_res:
            df_cruzado['Periodo'] = '2008-2014'
        elif '2015_2018' in archivo_res:
            df_cruzado['Periodo'] = '2015-2018'
        elif '2019_2022' in archivo_res:
            df_cruzado['Periodo'] = '2019-2022'
        elif '2023_2025' in archivo_res:
            df_cruzado['Periodo'] = '2023-2025'
            
        lista_dataframes_finales.append(df_cruzado)
        print(f"Cruce exitoso para: {archivo_res}")
        
    except Exception as e:
        print(f"Error procesando {archivo_res}: {e}")

# 4. Unir todo en un solo dataset 
df_maestro = pd.concat(lista_dataframes_finales, ignore_index=True)
df_maestro.to_csv('union_clustering_topicos.csv', index=False)
print("¡Archivo 'union_clustering_topicos.csv' generado con éxito!")
