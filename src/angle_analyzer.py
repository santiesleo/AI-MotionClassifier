# src/angle_analyzer.py
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

class AngleAnalyzer:
    """Análisis de ángulos articulares para clasificación de actividades físicas."""
    
    def __init__(self):
        self.angle_history = {}  # Historial de ángulos por articulación
        self.activity_patterns = {}  # Patrones detectados por actividad
    
    def calculate_knee_angle(self, hip, knee, ankle):
        """Calcula el ángulo de la rodilla."""
        if not all(coord in hip for coord in ['x', 'y']) or \
           not all(coord in knee for coord in ['x', 'y']) or \
           not all(coord in ankle for coord in ['x', 'y']):
            return None
            
        v1 = np.array([hip['x'] - knee['x'], hip['y'] - knee['y']])
        v2 = np.array([ankle['x'] - knee['x'], ankle['y'] - knee['y']])
        
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm > 0 and v2_norm > 0:
            cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            return angle
        return None
    
    def calculate_trunk_tilt(self, left_shoulder, right_shoulder):
        """Calcula la inclinación del tronco."""
        if not all(coord in left_shoulder for coord in ['x', 'y']) or \
           not all(coord in right_shoulder for coord in ['x', 'y']):
            return None
            
        shoulder_vector = np.array([right_shoulder['x'] - left_shoulder['x'], 
                                   right_shoulder['y'] - left_shoulder['y']])
        horizontal = np.array([1, 0])
        
        shoulder_norm = np.linalg.norm(shoulder_vector)
        if shoulder_norm > 0:
            cos_angle = np.clip(np.dot(shoulder_vector, horizontal) / shoulder_norm, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            return angle
        return None
    
    def calculate_lateral_inclination(self, left_shoulder, right_shoulder, left_hip, right_hip):
        """Calcula inclinación lateral del tronco."""
        if None in [left_shoulder, right_shoulder, left_hip, right_hip]:
            return None
            
        # Calcular punto medio de hombros y caderas
        shoulder_mid_x = (left_shoulder['x'] + right_shoulder['x']) / 2
        hip_mid_x = (left_hip['x'] + right_hip['x']) / 2
        
        # Diferencia indica inclinación lateral
        return shoulder_mid_x - hip_mid_x
    
    def update_angle_history(self, joint, angle):
        """Actualiza el historial de ángulos para una articulación."""
        if joint not in self.angle_history:
            self.angle_history[joint] = []
            
        # Limitar la longitud del historial para evitar consumo excesivo de memoria
        max_history = 300  # 10 segundos a 30fps
        if len(self.angle_history[joint]) >= max_history:
            self.angle_history[joint].pop(0)
            
        self.angle_history[joint].append(angle)
    
    def analyze_joint_angles(self, landmarks):
        """Analiza los ángulos de las articulaciones clave para un frame."""
        results = {}
        
        # Analizar rodillas
        for side in ['LEFT', 'RIGHT']:
            if all(key in landmarks for key in [f'{side}_HIP', f'{side}_KNEE', f'{side}_ANKLE']):
                knee_angle = self.calculate_knee_angle(
                    landmarks[f'{side}_HIP'],
                    landmarks[f'{side}_KNEE'],
                    landmarks[f'{side}_ANKLE']
                )
                if knee_angle is not None:
                    joint = f'{side}_KNEE_ANGLE'
                    results[joint] = knee_angle
                    self.update_angle_history(joint, knee_angle)
        
        # Analizar inclinación del tronco
        if all(key in landmarks for key in ['LEFT_SHOULDER', 'RIGHT_SHOULDER']):
            trunk_tilt = self.calculate_trunk_tilt(
                landmarks['LEFT_SHOULDER'],
                landmarks['RIGHT_SHOULDER']
            )
            if trunk_tilt is not None:
                results['TRUNK_TILT'] = trunk_tilt
                self.update_angle_history('TRUNK_TILT', trunk_tilt)
        
        # Analizar inclinación lateral
        if all(key in landmarks for key in ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP']):
            lateral_incl = self.calculate_lateral_inclination(
                landmarks['LEFT_SHOULDER'],
                landmarks['RIGHT_SHOULDER'],
                landmarks['LEFT_HIP'],
                landmarks['RIGHT_HIP']
            )
            if lateral_incl is not None:
                results['LATERAL_INCLINATION'] = lateral_incl
                self.update_angle_history('LATERAL_INCLINATION', lateral_incl)
        
        return results

    def identify_activity_pattern(self, angles_history, window_size=30):
        """Identifica patrones de actividad basados en secuencias de ángulos."""
        if len(angles_history.get('LEFT_KNEE_ANGLE', [])) < window_size:
            return "Desconocida", 0.0
            
        # Extraer características de la ventana reciente
        features = {}
        
        # Ángulos de rodilla (promedio y rango)
        for knee in ['LEFT_KNEE_ANGLE', 'RIGHT_KNEE_ANGLE']:
            if knee in angles_history and len(angles_history[knee]) >= window_size:
                recent = angles_history[knee][-window_size:]
                features[f"{knee}_avg"] = np.mean(recent)
                features[f"{knee}_range"] = np.max(recent) - np.min(recent)
                features[f"{knee}_std"] = np.std(recent)
        
        # Inclinación del tronco
        if 'TRUNK_TILT' in angles_history and len(angles_history['TRUNK_TILT']) >= window_size:
            recent = angles_history['TRUNK_TILT'][-window_size:]
            features["TRUNK_TILT_avg"] = np.mean(recent)
            features["TRUNK_TILT_range"] = np.max(recent) - np.min(recent)
        
        # Criterios para identificar actividades
        # Estos umbrales son ejemplos y deben ajustarse con datos reales
        
        # Sentarse: Disminución significativa en ángulo de rodilla
        if ('LEFT_KNEE_ANGLE_range' in features and features['LEFT_KNEE_ANGLE_range'] > 30 and
            'LEFT_KNEE_ANGLE_avg' in features and features['LEFT_KNEE_ANGLE_avg'] < 100):
            return "Sentarse", 0.8
            
        # Levantarse: Aumento significativo en ángulo de rodilla
        if ('LEFT_KNEE_ANGLE_range' in features and features['LEFT_KNEE_ANGLE_range'] > 30 and
            'LEFT_KNEE_ANGLE_avg' in features and features['LEFT_KNEE_ANGLE_avg'] > 140):
            return "Levantarse", 0.8
            
        # Girar: Cambios en inclinación del tronco
        if ('TRUNK_TILT_range' in features and features['TRUNK_TILT_range'] > 30):
            return "Girar", 0.7
            
        # Caminar hacia/desde: Basado en posición relativa y movimiento
        # Esta lógica es simplificada y necesitaría datos reales para mejorar
        
        return "Desconocida", 0.0
    
    def visualize_angle_history(self, save_path=None):
        """Visualiza el historial de ángulos para análisis."""
        if not self.angle_history:
            print("No hay datos de ángulos para visualizar.")
            return
            
        plt.figure(figsize=(12, 8))
        for joint, angles in self.angle_history.items():
            if len(angles) > 1:  # Asegurar que hay datos suficientes
                plt.plot(angles, label=joint)
        
        plt.title('Historial de Ángulos Articulares')
        plt.xlabel('Frames')
        plt.ylabel('Ángulo (grados)')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()