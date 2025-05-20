# src/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

def plot_feature_distributions(features_df, output_dir=None):
    """
    Visualiza distribuciones de características clave por actividad.
    """
    # Configurar el estilo
    plt.style.use('seaborn-whitegrid')
    sns.set_palette("colorblind")
    
    # Seleccionar características numéricas importantes
    numeric_features = [
        'LEFT_KNEE_ANGLE', 'RIGHT_KNEE_ANGLE', 
        'LEFT_WRIST_VELOCITY', 'RIGHT_WRIST_VELOCITY',
        'LATERAL_INCLINATION'
    ]
    
    # Filtrar características disponibles
    available_features = [f for f in numeric_features if f in features_df.columns]
    
    if not available_features:
        print("No hay características numéricas disponibles para visualizar.")
        return {}
    
    # Crear directorio para visualizaciones si no existe
    if output_dir:
        output_path = Path(output_dir) / "visualizations"
        output_path.mkdir(exist_ok=True)
    else:
        output_path = None
    
    # Diccionario para almacenar estadísticas por característica y actividad
    feature_stats = {}
    
    # Visualizar distribuciones por característica
    for feature in available_features:
        # Crear figura
        plt.figure(figsize=(12, 8))
        
        # Visualizar distribución por actividad
        sns.violinplot(x='activity', y=feature, data=features_df, inner='quartile')
        
        plt.title(f'Distribución de {feature} por Actividad')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Guardar figura si se especificó directorio
        if output_path:
            plt.savefig(output_path / f"{feature}_distribution.png")
            plt.close()
        else:
            plt.show()
        
        # Calcular estadísticas por actividad
        stats = features_df.groupby('activity')[feature].agg(['mean', 'std', 'min', 'max']).to_dict()
        feature_stats[feature] = stats
    
    # Crear matriz de correlación
    plt.figure(figsize=(10, 8))
    numeric_cols = features_df.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = features_df[numeric_cols].corr()
    
    # Visualizar matriz de correlación
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Matriz de Correlación de Características')
    
    # Guardar figura si se especificó directorio
    if output_path:
        plt.savefig(output_path / "correlation_matrix.png")
        plt.close()
        
        # Guardar estadísticas
        with open(output_path / "feature_statistics.json", 'w') as f:
            # Convertir arrays numpy a valores Python nativos para serialización JSON
            stats_serializable = {}
            for feature, stats in feature_stats.items():
                stats_serializable[feature] = {
                    stat_name: {
                        activity: float(value) if isinstance(value, (np.float32, np.float64)) else value
                        for activity, value in activity_stats.items()
                    }
                    for stat_name, activity_stats in stats.items()
                }
            json.dump(stats_serializable, f, indent=2)
    else:
        plt.show()
    
    return feature_stats

def plot_model_results(model_comparison_path, confusion_matrices_dir=None, output_dir=None):
    """
    Visualiza los resultados de entrenamiento de los modelos.
    """
    # Crear directorio para visualizaciones si no existe
    if output_dir:
        output_path = Path(output_dir) / "visualizations"
        output_path.mkdir(exist_ok=True)
    else:
        output_path = None
    
    # Cargar comparación de modelos
    model_df = pd.read_csv(model_comparison_path)
    
    # Visualizar comparación de modelos
    plt.figure(figsize=(12, 8))
    
    # Convertir a formato long para seaborn
    model_df_long = pd.melt(model_df, id_vars=['modelo'],
                           value_vars=['accuracy', 'weighted_precision', 'weighted_recall', 'weighted_f1'],
                           var_name='Métrica', value_name='Valor')
    
    # Gráfico de barras
    sns.barplot(x='modelo', y='Valor', hue='Métrica', data=model_df_long)
    plt.title('Comparación de Modelos')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Guardar figura si se especificó directorio
    if output_path:
        plt.savefig(output_path / "model_comparison.png")
        plt.close()
    else:
        plt.show()
    
    # Cargar e interpretar matrices de confusión si se especificó directorio
    if confusion_matrices_dir:
        confusion_dir = Path(confusion_matrices_dir)
        
        # Buscar archivos de matrices de confusión
        conf_files = list(confusion_dir.glob("confusion_matrix_*.png"))
        
        # Crear figura con todas las matrices
        if conf_files:
            fig, axes = plt.subplots(1, len(conf_files), figsize=(6*len(conf_files), 5))
            
            # Si solo hay una matriz, axes no es iterable
            if len(conf_files) == 1:
                axes = [axes]
            
            for i, conf_file in enumerate(conf_files):
                # Cargar imagen
                img = plt.imread(conf_file)
                axes[i].imshow(img)
                axes[i].set_title(conf_file.stem.replace("confusion_matrix_", ""))
                axes[i].axis('off')
            
            plt.tight_layout()
            
            # Guardar figura combinada si se especificó directorio
            if output_path:
                plt.savefig(output_path / "all_confusion_matrices.png")
                plt.close()
            else:
                plt.show()
    
    return model_df

def create_deployment_diagram(output_dir=None):
    """
    Crea un diagrama visual del plan de despliegue.
    """
    # Crear directorio para visualizaciones si no existe
    if output_dir:
        output_path = Path(output_dir) / "visualizations"
        output_path.mkdir(exist_ok=True)
    else:
        output_path = None
    
    # Configuración inicial
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    
    # Crear diagrama de arquitectura del sistema
    components = ['Captura\nde Video', 'Extracción\nde Landmarks', 'Preprocesamiento', 
                 'Extracción de\nCaracterísticas', 'Clasificación', 'Visualización']
    
    positions = np.arange(len(components))
    
    plt.barh(positions, [0.7] * len(components), height=0.5, left=0, color='lightblue', edgecolor='black')
    
    # Añadir etiquetas
    for i, (comp, pos) in enumerate(zip(components, positions)):
        plt.text(0.35, pos, comp, ha='center', va='center', fontweight='bold')
        if i < len(components) - 1:
            plt.arrow(0.7, pos, 0.15, 0, head_width=0.05, head_length=0.05, fc='black', ec='black')
            plt.text(0.775, pos, f"{(i+1)*30}ms", fontsize=8)
    
    plt.xlim(0, 1)
    plt.title('Arquitectura del Sistema en Tiempo Real')
    plt.axis('off')
    
    # Diagrama de Gantt para el plan de implementación
    plt.subplot(2, 1, 2)
    
    tasks = ['Desarrollo de prototipo', 'Optimización para tiempo real', 
             'Pruebas con usuarios', 'Refinamiento', 'Entrega final']
    
    durations = [2, 1.5, 1, 1.5, 0.5]  # semanas
    starts = [0, 2, 3.5, 4.5, 6]
    
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99', '#C0C0C0']
    
    for i, (task, duration, start, color) in enumerate(zip(tasks, durations, starts, colors)):
        plt.barh(i, duration, left=start, height=0.5, color=color, edgecolor='black')
        plt.text(start + duration/2, i, task, ha='center', va='center')
    
    plt.yticks([])
    plt.xlabel('Semanas')
    plt.title('Plan de Implementación')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Guardar figura si se especificó directorio
    if output_path:
        plt.savefig(output_path / "deployment_plan.png")
        plt.close()
    else:
        plt.show()
    
    # Generar diagrama de requisitos técnicos
    plt.figure(figsize=(10, 8))
    
    # Categorías y requisitos
    categories = ['Hardware', 'Software', 'Rendimiento']
    requirements = [
        ['CPU i5+', '8GB RAM', 'Cámara 720p', 'GPU (opcional)'],
        ['Python 3.8+', 'OpenCV 4.5+', 'MediaPipe', 'scikit-learn'],
        ['<100ms/frame', 'Exactitud >85%', 'Estabilidad']
    ]
    
    # Crear diagrama
    for i, (category, reqs) in enumerate(zip(categories, requirements)):
        plt.subplot(3, 1, i+1)
        
        # Crear barras horizontales
        bars = plt.barh(np.arange(len(reqs)), [0.8] * len(reqs), height=0.6, 
                       color=['#99CCFF', '#FFCC99', '#99FF99'][i], alpha=0.7)
        
        # Añadir etiquetas
        for j, req in enumerate(reqs):
            plt.text(0.4, j, req, ha='center', va='center', fontweight='bold')
        
        plt.title(f'Requisitos de {category}')
        plt.yticks([])
        plt.xticks([])
        plt.xlim(0, 1)
    
    plt.tight_layout()
    
    # Guardar figura si se especificó directorio
    if output_path:
        plt.savefig(output_path / "technical_requirements.png")
        plt.close()
    else:
        plt.show()
    
    return {
        'architecture': components,
        'implementation_plan': tasks,
        'technical_requirements': {cat: reqs for cat, reqs in zip(categories, requirements)}
    }

def impact_analysis_diagram(output_dir=None):
    """
    Crea un diagrama visual del análisis de impactos.
    """
    # Crear directorio para visualizaciones si no existe
    if output_dir:
        output_path = Path(output_dir) / "visualizations"
        output_path.mkdir(exist_ok=True)
    else:
        output_path = None
    
    # Categorías de impacto
    impact_categories = ['Técnico', 'Clínico', 'Educativo', 'Social', 'Privacidad']
    impact_types = ['Positivo', 'Negativo', 'Mitigación']
    
    # Datos de impacto
    impact_data = {
        'Técnico': {
            'Positivo': 'Sistema no invasivo para medición de movimientos\nClasificación automática de actividades\nCuantificación objetiva de ángulos articulares',
            'Negativo': 'Dependencia de buena iluminación\nResultados afectados por oclusiones\nRequiere hardware mínimo para tiempo real',
            'Mitigación': 'Algoritmos robustos a iluminación variable\nMúltiples cámaras para reducir oclusiones\nOptimización de código para hardware menos potente'
        },
        'Clínico': {
            'Positivo': 'Seguimiento objetivo de la evolución del paciente\nDetección temprana de anomalías de movimiento\nApoyo a diagnóstico en enfermedades neurodegenerativas',
            'Negativo': 'No reemplaza evaluación médica profesional\nPosible sobrediagnóstico por falsos positivos\nNo está validado clínicamente',
            'Mitigación': 'Presentar como herramienta de apoyo, no diagnóstica\nValidación con expertos clínicos\nClaras advertencias sobre limitaciones'
        },
        'Educativo': {
            'Positivo': 'Herramienta para enseñanza de biomecánica\nFeedback visual a estudiantes de medicina\nDemostración interactiva de patrones de movimiento',
            'Negativo': 'Curva de aprendizaje para nuevos usuarios\nPosibilidad de malinterpretación de resultados\nNecesidad de conocimientos técnicos básicos',
            'Mitigación': 'Desarrollo de tutoriales claros\nInterfaz intuitiva con explicaciones\nCapacitación específica para educadores'
        },
        'Social': {
            'Positivo': 'Acceso a tecnología de análisis de movimiento\nEmpoderamiento de pacientes en su tratamiento\nPosibilidad de uso remoto/telemedicina',
            'Negativo': 'Brecha digital para quien no tiene acceso\nPosible estigmatización al identificar patrones anormales\nPreocupaciones sobre vigilancia',
            'Mitigación': 'Versiones simplificadas para dispositivos básicos\nEnfoque en mejora y no en etiquetado negativo\nTransparencia total sobre uso de datos'
        },
        'Privacidad': {
            'Positivo': 'Procesamiento local sin almacenamiento de video\nNo requiere identificación personal\nControl del usuario sobre sus datos',
            'Negativo': 'Captura de video podría preocupar a usuarios\nPosible identificación indirecta por patrones de movimiento\nRiesgo de uso inadecuado',
            'Mitigación': 'Procesamiento en tiempo real sin guardar videos\nAnonymización de datos para investigación\nPolíticas claras de uso aceptable y consentimiento'
        }
    }
    
    # Crear visualización
    plt.figure(figsize=(15, 12))
    
    for i, category in enumerate(impact_categories):
        plt.subplot(len(impact_categories), 1, i+1)
        
        for j, impact_type in enumerate(impact_types):
            plt.barh(j, 0.8, left=j, height=0.6, 
                    color=['green', 'red', 'blue'][j], alpha=0.3)
            plt.text(j+0.4, j, impact_data[category][impact_type], 
                    ha='center', va='center', fontsize=9, wrap=True)
        
        plt.yticks(range(len(impact_types)), impact_types)
        plt.title(f'Impacto {category}')
        plt.xticks([])
        
    plt.tight_layout()
    
    # Guardar figura si se especificó directorio
    if output_path:
        plt.savefig(output_path / "impact_analysis.png")
        plt.close()
    else:
        plt.show()
    
    return impact_data

if __name__ == "__main__":
    # Ejemplo de uso
    data_dir = "dataset_processed"
    features_path = os.path.join(data_dir, "features.csv")
    
    # Verificar si existe el archivo de características
    if os.path.exists(features_path):
        features_df = pd.read_csv(features_path)
        plot_feature_distributions(features_df, data_dir)
    
    # Verificar si existe el archivo de comparación de modelos
    model_comparison_path = os.path.join(data_dir, "model_comparison.csv")
    if os.path.exists(model_comparison_path):
        plot_model_results(model_comparison_path, data_dir, data_dir)
    
    # Crear diagramas de despliegue e impacto
    create_deployment_diagram(data_dir)
    impact_analysis_diagram(data_dir)