# src/model_training.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

class ActivityClassifier:
    """
    Clase para entrenar y evaluar modelos de clasificación de actividades físicas
    basados en características extraídas de landmarks corporales.
    """
    
    def __init__(self, models_to_try=None, output_dir="dataset_processed"):
        """
        Inicializa el clasificador con los modelos especificados.
        """
        self.models = {}
        self.best_models = {}
        self.results = {}
        self.output_dir = Path(output_dir)
        
        # Definir modelos a probar si no se especifican
        if models_to_try is None:
            self.models_to_try = ['svm', 'random_forest', 'xgboost']
        else:
            self.models_to_try = models_to_try
            
    def prepare_data(self, features_df, test_size=0.2, random_state=42):
        """
        Prepara los datos para entrenamiento y prueba.
        """
        # Seleccionar características y etiquetas
        if 'activity' not in features_df.columns:
            raise ValueError("La columna 'activity' no está presente en los datos")
            
        # Eliminar columnas que no son características
        exclude_cols = ['frame_index', 'activity', 'subject_id']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        
        # Verificar que hay características disponibles
        if len(feature_cols) == 0:
            raise ValueError("No hay características disponibles para entrenamiento")
            
        # Separar características y etiquetas
        X = features_df[feature_cols].fillna(0)  # Reemplazar NaN con 0
        y = features_df['activity']
        
        # Dividir en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Normalizar características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Guardar datos
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_cols
        self.scaler = scaler
        
        print(f"Datos preparados: {X_train.shape[0]} muestras de entrenamiento, {X_test.shape[0]} muestras de prueba")
        print(f"Distribución de clases en entrenamiento:\n{y_train.value_counts()}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_and_tune_models(self):
        """
        Entrena y ajusta hiperparámetros para los modelos seleccionados.
        """
        if not hasattr(self, 'X_train'):
            raise ValueError("Primero debe preparar los datos con prepare_data()")
            
        # Configurar parámetros para cada modelo
        param_grids = {
            'svm': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        }
        
        # Crear y ajustar cada modelo
        for model_name in self.models_to_try:
            print(f"\nEntrenando y ajustando modelo: {model_name}")
            
            if model_name == 'svm':
                base_model = SVC(probability=True)
                param_grid = param_grids['svm']
            elif model_name == 'random_forest':
                base_model = RandomForestClassifier(random_state=42)
                param_grid = param_grids['random_forest']
            elif model_name == 'xgboost':
                # Crear el modelo base sin parámetros conflictivos
                base_model = xgb.XGBClassifier(
                    random_state=42,
                    use_label_encoder=False,  # Evitar advertencias
                    eval_metric='mlogloss'    # Métrica para multiclase
                )
                param_grid = param_grids['xgboost']
            else:
                print(f"Modelo {model_name} no reconocido. Omitiendo.")
                continue
                
            # Usar validación cruzada para encontrar mejores hiperparámetros
            # Reducir el número de combinaciones para modelos grandes
            if model_name in ['random_forest', 'xgboost']:
                # Tomar una muestra más pequeña de la cuadrícula de parámetros
                param_grid = {k: v[:2] for k, v in param_grid.items()}
                
            grid_search = GridSearchCV(
                base_model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            # Guardar el mejor modelo y sus resultados
            self.best_models[model_name] = grid_search.best_estimator_
            self.results[model_name] = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
            print(f"Mejores parámetros para {model_name}: {grid_search.best_params_}")
            print(f"Mejor F1 ponderado (CV): {grid_search.best_score_:.4f}")
    
    def evaluate_models(self):
        """
        Evalúa los modelos entrenados en el conjunto de prueba.
        """
        if not hasattr(self, 'best_models') or len(self.best_models) == 0:
            raise ValueError("Primero debe entrenar los modelos con train_and_tune_models()")
            
        evaluation_results = {}
        
        for model_name, model in self.best_models.items():
            print(f"\nEvaluando modelo: {model_name}")
            
            # Predecir en conjunto de prueba
            y_pred = model.predict(self.X_test)
            
            # Calcular métricas
            report = classification_report(self.y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            
            # Guardar resultados
            evaluation_results[model_name] = {
                'classification_report': report,
                'confusion_matrix': conf_matrix,
                'accuracy': report['accuracy'],
                'weighted_precision': report['weighted avg']['precision'],
                'weighted_recall': report['weighted avg']['recall'],
                'weighted_f1': report['weighted avg']['f1-score']
            }
            
            # Mostrar resultados
            print(f"Exactitud: {report['accuracy']:.4f}")
            print(f"Precisión ponderada: {report['weighted avg']['precision']:.4f}")
            print(f"Recall ponderado: {report['weighted avg']['recall']:.4f}")
            print(f"F1 ponderado: {report['weighted avg']['f1-score']:.4f}")
            
            # Visualizar matriz de confusión
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=np.unique(self.y_test),
                       yticklabels=np.unique(self.y_test))
            plt.xlabel('Predicción')
            plt.ylabel('Valor real')
            plt.title(f'Matriz de confusión - {model_name}')
            plt.tight_layout()
            plt.savefig(self.output_dir / f"confusion_matrix_{model_name}.png")
            plt.close()
        
        self.evaluation_results = evaluation_results
        
        # Guardar resultados
        results_file = self.output_dir / "model_evaluation.json"
        with open(results_file, 'w') as f:
            # Convertir arrays numpy a listas para serialización JSON
            serializable_results = {}
            for model_name, results in evaluation_results.items():
                serializable_results[model_name] = {
                    'accuracy': float(results['accuracy']),
                    'weighted_precision': float(results['weighted_precision']),
                    'weighted_recall': float(results['weighted_recall']),
                    'weighted_f1': float(results['weighted_f1']),
                    'classification_report': {
                        k: (float(v) if isinstance(v, (np.float32, np.float64)) else 
                            v.tolist() if hasattr(v, 'tolist') else v)
                        for k, v in results['classification_report'].items()
                        if k not in ['confusion_matrix']
                    }
                }
            json.dump(serializable_results, f, indent=2)
        
        print(f"Resultados de evaluación guardados en {results_file}")
        
        return evaluation_results
    
    def plot_feature_importance(self):
        """
        Visualiza la importancia de características para modelos que lo soportan.
        """
        models_with_importance = ['random_forest', 'xgboost']
        
        for model_name in models_with_importance:
            if model_name in self.best_models:
                model = self.best_models[model_name]
                
                # Extraer importancia de características
                if model_name == 'random_forest':
                    importances = model.feature_importances_
                elif model_name == 'xgboost':
                    importances = model.feature_importances_
                else:
                    continue
                
                # Crear DataFrame para visualización
                feature_importance = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importances
                })
                
                # Ordenar por importancia
                feature_importance = feature_importance.sort_values('importance', ascending=False)
                
                # Visualizar y guardar
                plt.figure(figsize=(12, 8))
                sns.barplot(x='importance', y='feature', data=feature_importance)
                plt.title(f'Importancia de características - {model_name}')
                plt.tight_layout()
                plt.savefig(self.output_dir / f"feature_importance_{model_name}.png")
                plt.close()
                
                # Guardar datos de importancia
                feature_importance.to_csv(self.output_dir / f"feature_importance_{model_name}.csv", index=False)
    
    def compare_models(self):
        """
        Compara el rendimiento de los modelos entrenados.
        """
        if not hasattr(self, 'evaluation_results'):
            raise ValueError("Primero debe evaluar los modelos con evaluate_models()")
            
        # Extraer métricas para comparación
        metrics = ['accuracy', 'weighted_precision', 'weighted_recall', 'weighted_f1']
        model_names = list(self.evaluation_results.keys())
        
        comparison_data = []
        for model_name in model_names:
            row = {'modelo': model_name}
            for metric in metrics:
                row[metric] = self.evaluation_results[model_name][metric]
            comparison_data.append(row)
            
        comparison_df = pd.DataFrame(comparison_data)
        
        # Guardar comparación
        comparison_df.to_csv(self.output_dir / "model_comparison.csv", index=False)
        
        # Visualizar comparación
        plt.figure(figsize=(14, 8))
        comparison_melted = pd.melt(comparison_df, id_vars=['modelo'], 
                                   value_vars=metrics, 
                                   var_name='métrica', value_name='valor')
        
        sns.barplot(x='modelo', y='valor', hue='métrica', data=comparison_melted)
        plt.title('Comparación de métricas entre modelos')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "model_comparison.png")
        plt.close()
        
        # Encontrar el mejor modelo
        best_model_idx = comparison_df['weighted_f1'].idxmax()
        best_model_name = comparison_df.loc[best_model_idx, 'modelo']
        print(f"El mejor modelo basado en F1 ponderado es: {best_model_name}")
        
        return comparison_df
    
    def save_model(self, model_name=None):
        """
        Guarda el modelo y scaler para su uso posterior.
        """
        if not hasattr(self, 'best_models') or len(self.best_models) == 0:
            raise ValueError("No hay modelos entrenados para guardar.")
        
        # Si no se especifica un modelo, usar el mejor
        if model_name is None:
            if hasattr(self, 'evaluation_results'):
                # Crear DataFrame para comparación
                metrics_df = pd.DataFrame([
                    {'model': name, 'f1': results['weighted_f1']}
                    for name, results in self.evaluation_results.items()
                ])
                model_name = metrics_df.loc[metrics_df['f1'].idxmax(), 'model']
            else:
                # Si no hay evaluación, usar el primer modelo
                model_name = list(self.best_models.keys())[0]
        
        # Verificar que el modelo existe
        if model_name not in self.best_models:
            raise ValueError(f"Modelo '{model_name}' no encontrado")
        
        # Crear directorio para el modelo
        model_dir = self.output_dir / "model"
        model_dir.mkdir(exist_ok=True)
        
        # Guardar modelo
        model_path = model_dir / f"{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_models[model_name], f)
        
        # Guardar scaler
        scaler_path = model_dir / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Guardar metadatos
        metadata = {
            'model_name': model_name,
            'feature_names': self.feature_names,
            'classes': list(np.unique(self.y_train)),
            'model_params': str(self.best_models[model_name].get_params())
        }
        
        with open(model_dir / "model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Modelo guardado: {model_path}")
        print(f"Scaler guardado: {scaler_path}")
        
        return {
            'model_path': str(model_path),
            'scaler_path': str(scaler_path),
            'metadata_path': str(model_dir / "model_metadata.json")
        }

def train_models(features_path, output_dir="dataset_processed"):
    """
    Función principal para entrenar y evaluar modelos.
    """
    # Cargar características
    features_df = pd.read_csv(features_path)
    print(f"Datos cargados: {len(features_df)} muestras con {features_df.shape[1]} columnas")
    
    # Crear y entrenar clasificador
    classifier = ActivityClassifier(output_dir=output_dir)
    
    # Preparar datos
    classifier.prepare_data(features_df)
    
    # Entrenar y ajustar modelos
    classifier.train_and_tune_models()
    
    # Evaluar modelos
    classifier.evaluate_models()
    
    # Visualizar importancia de características
    classifier.plot_feature_importance()
    
    # Comparar modelos
    classifier.compare_models()
    
    # Guardar el mejor modelo
    model_info = classifier.save_model()
    
    return classifier, model_info

if __name__ == "__main__":
    # Definir rutas
    data_dir = "dataset_processed"
    features_path = os.path.join(data_dir, "features.csv")
    
    # Entrenar modelos
    classifier, model_info = train_models(features_path, data_dir)
    print("Entrenamiento completado.")