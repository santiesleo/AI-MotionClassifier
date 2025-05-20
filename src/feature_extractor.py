# src/feature_extractor.py
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import os

class FeatureExtractor:
    def __init__(self, processed_dir="dataset_processed"):
        self.processed_dir = Path(processed_dir)
        self.features_path = self.processed_dir / "features.csv"
    
    def calculate_angle(self, p1, p2, p3):
        """Calculate the angle between three points."""
        if not all(key in p1 for key in ['x', 'y']) or not all(key in p2 for key in ['x', 'y']) or not all(key in p3 for key in ['x', 'y']):
            return None
            
        v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
        v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm > 0 and v2_norm > 0:
            angle = np.degrees(np.arccos(np.clip(
                np.dot(v1, v2) / (v1_norm * v2_norm), 
                -1.0, 1.0
            )))
            return angle
        return None

    def calculate_velocity(self, p1, p2, delta_t=1/30):
        """Calculate velocity between two points given the time difference."""
        if not all(key in p1 for key in ['x', 'y']) or not all(key in p2 for key in ['x', 'y']):
            return None
            
        distance = np.sqrt((p1['x'] - p2['x']) ** 2 + (p1['y'] - p2['y']) ** 2)
        return distance / delta_t

    def calculate_lateral_inclination(self, left_shoulder, right_shoulder, left_hip, right_hip):
        """Calculate lateral inclination based on shoulder and hip positions."""
        if None in [left_shoulder, right_shoulder, left_hip, right_hip]:
            return None
        shoulder_mid = (left_shoulder + right_shoulder) / 2
        hip_mid = (left_hip + right_hip) / 2
        return shoulder_mid - hip_mid

    def extract_features_from_file(self, json_file: Path) -> List[Dict]:
        """Extract features from a single processed JSON file."""
        features_list = []
        
        try:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
                
            if 'frames' not in json_data:
                print(f"Key 'frames' not found in {json_file.name}.")
                return []
                
            frames = json_data['frames']
            activity = json_data['metadata'].get('activity', 'unknown')
            subject_id = json_data['metadata'].get('subject_id', 'unknown')
            
            for i, frame in enumerate(frames):
                landmarks = frame.get('landmarks', {})
                
                # Verificar si hay landmarks suficientes
                if not all(key in landmarks for key in ['LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE']):
                    continue
                
                # Calcular ángulos de rodilla
                left_knee_angle = self.calculate_angle(
                    landmarks.get('LEFT_HIP', {}), 
                    landmarks.get('LEFT_KNEE', {}), 
                    landmarks.get('LEFT_ANKLE', {})
                )
                
                right_knee_angle = self.calculate_angle(
                    landmarks.get('RIGHT_HIP', {}), 
                    landmarks.get('RIGHT_KNEE', {}), 
                    landmarks.get('RIGHT_ANKLE', {})
                )
                
                # Extraer posiciones de las articulaciones
                left_shoulder_x = landmarks.get('LEFT_SHOULDER', {}).get('x')
                right_shoulder_x = landmarks.get('RIGHT_SHOULDER', {}).get('x')
                left_hip_x = landmarks.get('LEFT_HIP', {}).get('x')
                right_hip_x = landmarks.get('RIGHT_HIP', {}).get('x')
                
                # Calcular inclinación lateral
                lateral_inclination = self.calculate_lateral_inclination(
                    left_shoulder_x, right_shoulder_x, left_hip_x, right_hip_x)
                
                # Calcular velocidades si no es el primer frame
                left_wrist_velocity = None
                right_wrist_velocity = None
                
                if i > 0 and 'landmarks' in frames[i-1]:
                    prev_landmarks = frames[i-1]['landmarks']
                    delta_t = 1/30  # Asumiendo 30 fps
                    
                    if 'LEFT_WRIST' in landmarks and 'LEFT_WRIST' in prev_landmarks:
                        left_wrist_velocity = self.calculate_velocity(
                            prev_landmarks['LEFT_WRIST'],
                            landmarks['LEFT_WRIST'],
                            delta_t
                        )
                    
                    if 'RIGHT_WRIST' in landmarks and 'RIGHT_WRIST' in prev_landmarks:
                        right_wrist_velocity = self.calculate_velocity(
                            prev_landmarks['RIGHT_WRIST'],
                            landmarks['RIGHT_WRIST'],
                            delta_t
                        )
                
                # Crear diccionario de características
                features = {
                    'LEFT_KNEE_ANGLE': left_knee_angle,
                    'RIGHT_KNEE_ANGLE': right_knee_angle,
                    'LEFT_WRIST_X': landmarks.get('LEFT_WRIST', {}).get('x'),
                    'RIGHT_WRIST_X': landmarks.get('RIGHT_WRIST', {}).get('x'),
                    'LEFT_SHOULDER_X': left_shoulder_x,
                    'RIGHT_SHOULDER_X': right_shoulder_x,
                    'NOSE_X': landmarks.get('NOSE', {}).get('x'),
                    'LATERAL_INCLINATION': lateral_inclination,
                    'LEFT_WRIST_VELOCITY': left_wrist_velocity,
                    'RIGHT_WRIST_VELOCITY': right_wrist_velocity,
                    'activity': activity,
                    'frame_index': i,
                    'subject_id': subject_id
                }
                
                features_list.append(features)
                
        except Exception as e:
            print(f"Error processing {json_file.name}: {str(e)}")
        
        return features_list

    def extract_all_features(self) -> pd.DataFrame:
        """Extract features from all processed JSON files."""
        all_features = []
        invalid_files = []
        
        for json_file in self.processed_dir.glob('*.json'):
            # Saltar archivos de registro
            if json_file.name == "processing_log.json":
                continue
                
            print(f"Extrayendo características de {json_file.name}")
            file_features = self.extract_features_from_file(json_file)
            
            if not file_features:
                invalid_files.append(json_file.name)
            else:
                all_features.extend(file_features)
                
        # Crear DataFrame
        if not all_features:
            print("No se pudieron extraer características de ningún archivo.")
            return pd.DataFrame()
            
        df = pd.DataFrame(all_features)
        
        # Eliminar filas con valores NaN
        df_clean = df.dropna()
        print(f"Filas originales: {len(df)}, Filas después de eliminar NaN: {len(df_clean)}")
        
        # Guardar características
        df_clean.to_csv(self.features_path, index=False)
        print(f"Características guardadas en {self.features_path}")
        
        # Reportar archivos inválidos
        if invalid_files:
            print(f"\nSe encontraron {len(invalid_files)} archivos JSON inválidos:")
            for invalid_file in invalid_files:
                print(f"- {invalid_file}")
        
        return df_clean
    
    def load_features(self) -> pd.DataFrame:
        """Load extracted features from CSV file if it exists."""
        if self.features_path.exists():
            return pd.read_csv(self.features_path)
        else:
            print(f"Archivo de características no encontrado: {self.features_path}")
            return pd.DataFrame()

if __name__ == "__main__":
    extractor = FeatureExtractor()
    features_df = extractor.extract_all_features()
    print(f"Extracción completa: {len(features_df)} muestras con {len(features_df.columns)} características")