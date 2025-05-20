# Sistema de Anotación de Video para Análisis de Actividades Físicas

## Descripción del Proyecto

Este proyecto implementa un sistema de análisis de actividades físicas mediante computer vision, capaz de identificar cinco actividades específicas (caminar hacia la cámara, caminar desde la cámara, girar, sentarse y levantarse) y realizar un análisis de patrones posturales en tiempo real.

El sistema utiliza MediaPipe para extraer landmarks corporales de video, procesa estos datos para extraer características relevantes, y aplica algoritmos de machine learning para clasificar las actividades y analizar los patrones de movimiento.

## Características Principales

- Detección de actividades físicas en tiempo real
- Cálculo de ángulos articulares (rodillas, cadera, tronco)
- Análisis de patrones posturales
- Interfaz gráfica para visualización de resultados
- Grabación de actividades para entrenamiento
- Procesamiento optimizado para ejecución en tiempo real

## Estructura del Proyecto

XD POR DEFINIR


## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/santiesleo/AI-MotionClassifier
cd AI-MotionClassifier
```

2. Crear y activar entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```
3. Instalar dependencias:
```bash
pip install -r requirements.txt
```