#!/bin/bash

PROJECT_ROOT="../../"

# Definición de empresas y sus franjas correspondientes
declare -A FRANJAS
FRANJAS["AppleMusic"]="2015-2018 2019-2022 2023-2025"
FRANJAS["Spotify"]="2008-2014 2015-2018 2019-2022 2023-2025"

echo "===================================================="
echo "Iniciando clustering..."
echo "===================================================="

for EMPRESA in "${!FRANJAS[@]}"; do
    echo ">>> Empresa: $EMPRESA"
    
    for FRANJA in ${FRANJAS[$EMPRESA]}; do
        # Ruta del config relativa a esta carpeta
        CONFIG_FILE="./$EMPRESA/config_$FRANJA.json"
        
        if [ -f "$CONFIG_FILE" ]; then
            echo "--------------------------------------------"
            echo "Procesando la franja $FRANJA de $EMPRESA..."
            
            # Creamos la carpeta de salida (relativa a donde estamos)
            mkdir -p "./$EMPRESA/$FRANJA"
            
            # EJECUCIÓN:
            # 1. Entramos al root del proyecto
            # 2. Ejecutamos el módulo usando la ruta del config desde el root
            # 3. Volvemos a la carpeta auxiliarTableau
            
            CONFIG_PATH_FOR_PYTHON="data/auxiliarTableau/$EMPRESA/config_$FRANJA.json"
            
            cd "$PROJECT_ROOT"
            python3 -m src.analysis.clustering -c "$CONFIG_PATH_FOR_PYTHON"
            cd "data/auxiliarTableau"
            
            echo "OK: $FRANJA finalizado."
        else
            echo "[ERROR] No existe: $CONFIG_FILE"
        fi
    done
done

echo "===================================================="
echo "Todos los procesos finalizados."
echo "===================================================="
