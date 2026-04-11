# 🚀 Proyecto SAD: Análisis de Sentimientos y Toma de Decisiones

Este proyecto representa una situación real en una empresa de música. Como equipo de científicos de datos, nuestra misión es analizar el feedback de los usuarios para corregir el rumbo de la empresa y superar a la competencia. 

> "¡Sin análisis no hay mejora, los datos son nuestro poder!"

## 👥 El Equipo
Los integrantes del equipo somos

* **Eder Torres:**.
* **Liviu Deleanu:**.
* **Alexandriu Darie:**.
* **Aitor Cotano:**.

---

## 🎯 Objetivos del Proyecto
1.  **Análisis Comparativo:** Evaluar nuestra aplicación frente al competidor directo.
2.  **Detección de Fortalezas y Debilidades:** Identificar puntos flacos y fuertes mediante datos.
3.  **Sentiment Analysis:** Entrenar un clasificador (positivo, negativo, neutro) con datos de la app para luego analizar redes sociales como X, Instagram o TikTok.
4.  **Clustering:** Indagar en las razones subyacentes tras las opiniones de los usuarios.

## 🛠️ Requisitos Técnicos
### Modelado Tradicional
* Implementación de algoritmos como KNN, Decision Trees, Random Forest o Logistic Regression.
* Técnicas de balanceado: **Oversampling** y **Undersampling**.

### Modelado Generativo
* Uso de **Ollama** para clasificación mediante Prompt Engineering.
* Generación de paráfrasis para aumentar datos de clases minoritarias.

### Visualización (Tableau)
* Creación de una historia de datos que responda a la evolución por género/localización y correlación zona/satisfacción.

---

## 📂 Entregables (Fecha límite: 03/05/2026) 
1.  **Poster:** Incluye abstract gráfico, tablas de resultados (con y sin balanceo) y gráfico del codo para clustering.
2.  **Código del Clasificador:** Programa funcional, `requirements.txt` y manual de ejecución.
3.  **Historia de Tableau:** Archivo con la narrativa de datos y visualización de clusters.
4.  **CSV Generativo:** Informe de prompts empleados, modelos y resultados obtenidos.

---

## 🔧 Instalación y Uso
Para replicar nuestro entorno de trabajo:

```bash
# Instalar dependencias
pip install -r requirements.txt
```
# Ejecutar el clasificador
python clasificador.py --data train.csv
📚 Referencias Bibliográficas
[1] The Big Book of Dashboards, Jeffrey Shaffer et al..

[2] Storytelling con datos, Cole Nussbaumer Knaffic.

[3] Tutoriales de HuggingFace sobre Sentiment Analysis.

Curso: Sistemas de Ayuda a la Decisión (SAD)


Fecha: Marzo 2026
