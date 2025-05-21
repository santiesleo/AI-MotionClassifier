# src/video_processor.py
import cv2
import mediapipe as mp
import numpy as np
import json
import os
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

class SmartVideoProcessor:
    def __init__(self, input_dir="dataset_raw", output_dir="dataset_processed"):
        # Inicializar MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Crear archivo de registro si no existe
        self.log_file = self.output_dir / "processing_log.json"
        self.processed_videos = self.load_processing_log()
        
        # Articulaciones clave
        self.key_joints = [
            'LEFT_HIP', 'RIGHT_HIP',
            'LEFT_KNEE', 'RIGHT_KNEE',
            'LEFT_ANKLE', 'RIGHT_ANKLE',
            'LEFT_WRIST', 'RIGHT_WRIST',
            'LEFT_SHOULDER', 'RIGHT_SHOULDER',
            'NOSE'
        ]

    def load_processing_log(self):
        """Carga el registro de videos procesados."""
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return {}

    def update_processing_log(self, video_path, status="processed"):
        """Actualiza el registro de procesamiento."""
        self.processed_videos[video_path.name] = {
            "processed_date": datetime.now().isoformat(),
            "status": status
        }
        with open(self.log_file, 'w') as f:
            json.dump(self.processed_videos, f, indent=2)

    def needs_processing(self, video_path):
        """Verifica si un video necesita ser procesado."""
        # Verificar si el video ya fue procesado
        if video_path.name in self.processed_videos:
            # Verificar si existe el archivo de salida
            output_path = self.output_dir / f"{video_path.stem}_processed.json"
            if output_path.exists():
                return False
        return True

    def extract_landmarks(self, frame):
        """Extrae landmarks de un frame usando MediaPipe."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks_dict = {}
            for joint in self.key_joints:
                idx = getattr(self.mp_pose.PoseLandmark, joint)
                landmark = results.pose_landmarks.landmark[idx]
                landmarks_dict[joint] = {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                }
            return landmarks_dict
        return None

    def calculate_angles(self, landmarks_dict):
        """Calcula ángulos relevantes entre articulaciones."""
        angles = {}
        
        # Ángulo de rodillas
        for side in ['LEFT', 'RIGHT']:
            hip = landmarks_dict[f'{side}_HIP']
            knee = landmarks_dict[f'{side}_KNEE']
            ankle = landmarks_dict[f'{side}_ANKLE']
            
            v1 = np.array([hip['x'] - knee['x'], hip['y'] - knee['y']])
            v2 = np.array([ankle['x'] - knee['x'], ankle['y'] - knee['y']])
            
            angle = np.degrees(np.arccos(np.clip(
                np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), 
                -1.0, 1.0
            )))
            angles[f'{side}_KNEE_ANGLE'] = angle
        
        # Inclinación lateral del tronco
        left_shoulder = np.array([landmarks_dict['LEFT_SHOULDER']['x'],
                                landmarks_dict['LEFT_SHOULDER']['y']])
        right_shoulder = np.array([landmarks_dict['RIGHT_SHOULDER']['x'],
                                 landmarks_dict['RIGHT_SHOULDER']['y']])
        
        shoulder_vector = right_shoulder - left_shoulder
        horizontal = np.array([1, 0])
        
        trunk_angle = np.degrees(np.arccos(np.clip(
            np.dot(shoulder_vector, horizontal) / np.linalg.norm(shoulder_vector),
            -1.0, 1.0
        )))
        angles['TRUNK_TILT'] = trunk_angle
        
        return angles

    def process_video(self, video_path):
        """Procesa un video y extrae landmarks y ángulos."""
        filename = video_path.stem
        subject_id, activity, timestamp = filename.split('_', 2)
        
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        video_data = {
            'metadata': {
                'subject_id': subject_id,
                'activity': activity,
                'timestamp': timestamp,
                'total_frames': frame_count
            },
            'frames': []
        }
        
        with tqdm(total=frame_count, desc=f"Procesando {filename}") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                landmarks = self.extract_landmarks(frame)
                if landmarks:
                    angles = self.calculate_angles(landmarks)
                    frame_data = {
                        'landmarks': landmarks,
                        'angles': angles
                    }
                    video_data['frames'].append(frame_data)
                
                pbar.update(1)
        
        cap.release()
        return video_data

    def process_all_videos(self):
        """Procesa solo los videos nuevos en el directorio de entrada."""
        video_files = list(self.input_dir.glob('*.mp4'))
        new_videos = [v for v in video_files if self.needs_processing(v)]
        
        print(f"Encontrados {len(video_files)} videos totales")
        print(f"Videos nuevos para procesar: {len(new_videos)}")
        
        if not new_videos:
            print("No hay nuevos videos para procesar")
            return
        
        for video_path in new_videos:
            print(f"\nProcesando {video_path.name}")
            try:
                # Procesar video
                video_data = self.process_video(video_path)
                
                # Guardar resultados
                output_path = self.output_dir / f"{video_path.stem}_processed.json"
                with open(output_path, 'w') as f:
                    json.dump(video_data, f, indent=2)
                
                # Actualizar registro
                self.update_processing_log(video_path)
                print(f"Datos guardados en {output_path}")
                
            except Exception as e:
                print(f"Error procesando {video_path.name}: {str(e)}")
                self.update_processing_log(video_path, status="error")
    def draw_landmarks(self, frame, landmarks_dict):
        """Dibuja los landmarks detectados en el frame."""
        if not landmarks_dict:
            return frame
            
        # Crear una copia del frame para no modificar el original
        frame_with_landmarks = frame.copy()
        
        # Colores para dibujar (BGR)
        color_landmark = (0, 255, 0)  # Verde para landmarks
        color_connection = (255, 255, 0)  # Amarillo para conexiones
        
        # Dibujar landmarks
        for joint, data in landmarks_dict.items():
            if joint in self.key_joints:
                # Obtener coordenadas normalizadas
                x, y = data['x'], data['y']
                
                # Convertir a coordenadas de pixeles
                h, w = frame.shape[:2]
                px, py = int(x * w), int(y * h)
                
                # Dibujar círculo en la posición del landmark
                cv2.circle(frame_with_landmarks, (px, py), 5, color_landmark, -1)
                
                # Dibujar etiqueta
                cv2.putText(frame_with_landmarks, joint, (px, py - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Dibujar conexiones entre landmarks
        connections = [
            ('LEFT_HIP', 'LEFT_KNEE'), ('LEFT_KNEE', 'LEFT_ANKLE'),
            ('RIGHT_HIP', 'RIGHT_KNEE'), ('RIGHT_KNEE', 'RIGHT_ANKLE'),
            ('LEFT_SHOULDER', 'LEFT_HIP'), ('RIGHT_SHOULDER', 'RIGHT_HIP'),
            ('LEFT_SHOULDER', 'RIGHT_SHOULDER'), ('LEFT_HIP', 'RIGHT_HIP')
        ]
        
        for start_joint, end_joint in connections:
            if start_joint in landmarks_dict and end_joint in landmarks_dict:
                # Obtener coordenadas normalizadas
                x1, y1 = landmarks_dict[start_joint]['x'], landmarks_dict[start_joint]['y']
                x2, y2 = landmarks_dict[end_joint]['x'], landmarks_dict[end_joint]['y']
                
                # Convertir a coordenadas de pixeles
                h, w = frame.shape[:2]
                px1, py1 = int(x1 * w), int(y1 * h)
                px2, py2 = int(x2 * w), int(y2 * h)
                
                # Dibujar línea entre landmarks
                cv2.line(frame_with_landmarks, (px1, py1), (px2, py2), color_connection, 2)
        
        return frame_with_landmarks

    def draw_angles(self, frame, landmarks_dict, angles):
        """Dibuja ángulos articulares en el frame."""
        if not landmarks_dict or not angles:
            return frame
            
        # Crear una copia del frame para no modificar el original
        frame_with_angles = frame.copy()
        
        # Dibujar ángulos de rodillas
        for side in ['LEFT', 'RIGHT']:
            if f'{side}_KNEE_ANGLE' in angles and f'{side}_KNEE' in landmarks_dict:
                # Obtener valor del ángulo
                angle_value = angles[f'{side}_KNEE_ANGLE']
                
                # Obtener coordenadas de la rodilla
                x, y = landmarks_dict[f'{side}_KNEE']['x'], landmarks_dict[f'{side}_KNEE']['y']
                
                # Convertir a coordenadas de pixeles
                h, w = frame.shape[:2]
                px, py = int(x * w), int(y * h)
                
                # Determinar color basado en el ángulo (verde si está en rango normal)
                if 160 <= angle_value <= 180:  # Pierna recta
                    color = (0, 255, 0)  # Verde
                elif 70 <= angle_value < 160:  # Pierna flexionada normalmente
                    color = (255, 255, 0)  # Amarillo
                else:  # Flexión excesiva o valor inusual
                    color = (0, 0, 255)  # Rojo
                
                # Dibujar texto con el valor del ángulo
                text = f"{angle_value:.1f}°"
                cv2.putText(frame_with_angles, text, (px + 10, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Dibujar inclinación del tronco si está disponible
        if 'TRUNK_TILT' in angles:
            # Determinar posición para mostrar la inclinación
            if 'NOSE' in landmarks_dict:
                x, y = landmarks_dict['NOSE']['x'], landmarks_dict['NOSE']['y']
                h, w = frame.shape[:2]
                px, py = int(x * w), int(y * h) - 30
            else:
                px, py = 50, 50
            
            tilt_value = angles['TRUNK_TILT']
            
            # Determinar color basado en la inclinación
            if abs(90 - tilt_value) < 10:  # Casi vertical
                color = (0, 255, 0)  # Verde
            elif abs(90 - tilt_value) < 20:  # Ligera inclinación
                color = (255, 255, 0)  # Amarillo
            else:  # Inclinación significativa
                color = (0, 0, 255)  # Rojo
            
            # Dibujar texto con el valor de inclinación
            text = f"Tronco: {tilt_value:.1f}°"
            cv2.putText(frame_with_angles, text, (px, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame_with_angles

if __name__ == "__main__":
    processor = SmartVideoProcessor()
    processor.process_all_videos()