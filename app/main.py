# app/main_enhanced_detection.py
import cv2
import numpy as np
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QSplitter
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

# Suprimir advertencias
warnings.filterwarnings("ignore", category=UserWarning)

# Agregar directorio src al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar módulos propios
from src.angle_analyzer import AngleAnalyzer

# IMPORTACIONES CORREGIDAS
try:
    from gui import ModernVideoDisplay, AdvancedControlPanel, AdvancedInfoPanel
    print("Interfaz mejorada cargada correctamente")
except ImportError as e:
    print(f"Error importando interfaz mejorada: {e}")
    try:
        from gui import OptimizedVideoDisplay as ModernVideoDisplay, SimpleControlPanel as AdvancedControlPanel, MinimalInfoPanel as AdvancedInfoPanel
        print("Interfaz alternativa cargada")
    except ImportError as e2:
        print(f"Error con nombres alternativos: {e2}")
        print("Creando clases básicas...")
        
        from PyQt5.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget
        
        class ModernVideoDisplay(QWidget):
            def __init__(self):
                super().__init__()
                self.video_label = QLabel("Video aquí")
                layout = QVBoxLayout()
                layout.addWidget(self.video_label)
                self.setLayout(layout)
            
            def update_frame(self, frame):
                if frame is not None:
                    height, width, channel = frame.shape
                    bytes_per_line = 3 * width
                    q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                    pixmap = QPixmap.fromImage(q_image)
                    self.video_label.setPixmap(pixmap.scaled(640, 480, Qt.KeepAspectRatio))
            
            def update_activity_info(self, activity, confidence):
                pass
            
            def update_angles(self, angles):
                pass
            
            def update_fps(self, fps):
                pass
        
        class AdvancedControlPanel(QWidget):
            start_capture_signal = pyqtSignal(int)
            stop_capture_signal = pyqtSignal()
            
            def __init__(self):
                super().__init__()
                layout = QVBoxLayout()
                
                self.start_button = QPushButton("Iniciar Captura")
                self.stop_button = QPushButton("Detener Captura")
                
                self.start_button.clicked.connect(lambda: self.start_capture_signal.emit(0))
                self.stop_button.clicked.connect(self.stop_capture_signal.emit)
                
                layout.addWidget(self.start_button)
                layout.addWidget(self.stop_button)
                self.setLayout(layout)
        
        class AdvancedInfoPanel(QWidget):
            def __init__(self):
                super().__init__()
                self.info_label = QLabel("Panel de información")
                layout = QVBoxLayout()
                layout.addWidget(self.info_label)
                self.setLayout(layout)
            
            def update_activity(self, activity, confidence):
                self.info_label.setText(f"Actividad: {activity}\nConfianza: {confidence:.1f}%")
            
            def update_angles(self, angles):
                pass
            
            def update_fps(self, fps):
                pass
            
            def update_performance_metrics(self, processing_time, frames_count):
                pass

class FixedVideoProcessor:
    """Procesador de video con detección CORREGIDA."""
    
    def __init__(self):
        import mediapipe as mp
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Tu configuración optimizada
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,  # Tu configuración
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=0.5,  # Tu configuración
            min_tracking_confidence=0.5,   # Tu configuración
        )
        
        # Landmarks completos incluyendo cabeza
        self.key_joints = [
            'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER',
            'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
            'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
            'LEFT_SHOULDER', 'RIGHT_SHOULDER', 
            'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST',
            'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE',
            'LEFT_ANKLE', 'RIGHT_ANKLE'
        ]
        
        # Conexiones completas
        self.connections = [
            # Cabeza y cara
            ('NOSE', 'LEFT_EYE'), ('NOSE', 'RIGHT_EYE'),
            ('LEFT_EYE', 'LEFT_EAR'), ('RIGHT_EYE', 'RIGHT_EAR'),
            ('MOUTH_LEFT', 'MOUTH_RIGHT'),
            # Torso
            ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
            ('LEFT_SHOULDER', 'LEFT_HIP'),
            ('RIGHT_SHOULDER', 'RIGHT_HIP'),
            ('LEFT_HIP', 'RIGHT_HIP'),
            # Brazos
            ('LEFT_SHOULDER', 'LEFT_ELBOW'),
            ('LEFT_ELBOW', 'LEFT_WRIST'),
            ('RIGHT_SHOULDER', 'RIGHT_ELBOW'),
            ('RIGHT_ELBOW', 'RIGHT_WRIST'),
            # Piernas
            ('LEFT_HIP', 'LEFT_KNEE'),
            ('LEFT_KNEE', 'LEFT_ANKLE'),
            ('RIGHT_HIP', 'RIGHT_KNEE'),
            ('RIGHT_KNEE', 'RIGHT_ANKLE'),
            # Conexión cuello
            ('NOSE', 'LEFT_SHOULDER'),
            ('NOSE', 'RIGHT_SHOULDER'),
        ]
        
        # Optimización para mejor detección
        self.frame_skip_counter = 0
        self.frame_skip_rate = 1  # Procesar todos los frames para mejor detección
        self.last_landmarks = None
        self.landmarks_cache_frames = 0
        self.max_cache_frames = 1  # Reducir cache para mayor responsividad
        
        self.original_size = None
        self.processed_size = None
    
    def extract_landmarks_fixed(self, frame):
        """Extracción CORREGIDA de landmarks."""
        # Frame skipping reducido para mejor detección
        self.frame_skip_counter += 1
        if self.frame_skip_counter % self.frame_skip_rate != 0:
            if self.last_landmarks and self.landmarks_cache_frames < self.max_cache_frames:
                self.landmarks_cache_frames += 1
                return self.last_landmarks
            return None
        
        # Procesar frame
        original_height, original_width = frame.shape[:2]
        self.original_size = (original_width, original_height)
        
        # Mantener buena resolución para mejor detección
        if original_width > 640:
            scale_factor = 640 / original_width
            new_width = 640
            new_height = int(original_height * scale_factor)
            frame_resized = cv2.resize(frame, (new_width, new_height))
            self.processed_size = (new_width, new_height)
        else:
            frame_resized = frame
            scale_factor = 1.0
            self.processed_size = self.original_size
        
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks_dict = {}
            
            for joint in self.key_joints:
                try:
                    idx = getattr(self.mp_pose.PoseLandmark, joint)
                    landmark = results.pose_landmarks.landmark[idx]
                    
                    landmarks_dict[joint] = {
                        'x': landmark.x,
                        'y': landmark.y,  
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    }
                except AttributeError:
                    continue
            
            self.last_landmarks = landmarks_dict
            self.landmarks_cache_frames = 0
            return landmarks_dict
        
        return None
    
    def draw_skeleton_fixed(self, frame, landmarks_dict):
        """Dibujar esqueleto CORREGIDO."""
        if not landmarks_dict:
            return frame
        
        h, w = frame.shape[:2]
        
        # Dibujar conexiones con colores
        for start_joint, end_joint in self.connections:
            if (start_joint in landmarks_dict and end_joint in landmarks_dict and
                landmarks_dict[start_joint]['visibility'] > 0.5 and
                landmarks_dict[end_joint]['visibility'] > 0.5):
                
                start_x = int(landmarks_dict[start_joint]['x'] * w)
                start_y = int(landmarks_dict[start_joint]['y'] * h)
                end_x = int(landmarks_dict[end_joint]['x'] * w)
                end_y = int(landmarks_dict[end_joint]['y'] * h)
                
                # Colores por tipo
                if any(joint in ['EYE', 'EAR', 'NOSE', 'MOUTH'] for joint in [start_joint, end_joint]):
                    color = (255, 255, 0)  # Amarillo para cabeza
                elif any(joint in ['ARM', 'ELBOW', 'WRIST', 'SHOULDER'] for joint in [start_joint, end_joint]):
                    color = (255, 165, 0)  # Naranja para brazos
                elif any(joint in ['HIP', 'KNEE', 'ANKLE'] for joint in [start_joint, end_joint]):
                    color = (0, 255, 255)  # Cyan para piernas
                else:
                    color = (0, 255, 0)    # Verde para torso
                
                cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 2)
        
        # Dibujar puntos importantes
        joint_colors = {
            'NOSE': (255, 0, 255),
            'LEFT_EYE': (255, 255, 0), 'RIGHT_EYE': (255, 255, 0),
            'LEFT_EAR': (0, 255, 255), 'RIGHT_EAR': (0, 255, 255),
            'LEFT_SHOULDER': (0, 255, 0), 'RIGHT_SHOULDER': (0, 255, 0),
            'LEFT_HIP': (255, 0, 0), 'RIGHT_HIP': (255, 0, 0),
            'LEFT_KNEE': (0, 0, 255), 'RIGHT_KNEE': (0, 0, 255),
            'LEFT_ANKLE': (128, 0, 128), 'RIGHT_ANKLE': (128, 0, 128),
        }
        
        for joint, data in landmarks_dict.items():
            if data['visibility'] > 0.5:
                x = int(data['x'] * w)
                y = int(data['y'] * h)
                
                color = joint_colors.get(joint, (255, 255, 255))
                cv2.circle(frame, (x, y), 4, color, -1)
                cv2.circle(frame, (x, y), 4, (0, 0, 0), 1)
        
        return frame

class EnhancedActivityClassifier:
    """Clasificador MEJORADO para detectar todas las actividades."""
    
    def __init__(self):
        self.activity_names = {
            0: "Parado",
            1: "Caminar hacia cámara", 
            2: "Caminar desde cámara",
            3: "Girar",
            4: "Sentarse",
            5: "Levantarse"
        }
        
        # Historiales específicos para diferentes métricas
        self.movement_history = []
        self.position_history = []
        self.angle_history = []
        self.depth_history = []
        self.shoulder_width_history = []
        self.head_position_history = []
        
        self.max_history = 20
        
        # Umbrales calibrados - AJUSTABLES para tu setup
        self.thresholds = {
            'movement_walking': 0.012,      # Reducido para más sensibilidad
            'movement_turning': 0.006,      # Reducido para más sensibilidad
            'depth_approaching': 0.015,     # Cambio en profundidad
            'rotation_threshold': 0.008,    # Cambio en ancho de hombros
            'head_turn_threshold': 0.012,   # Movimiento de cabeza lateral
        }
        
        # Estados para mejor tracking
        self.current_state = "analyzing"
        self.state_confidence = 0.0
        self.state_frames = 0
        
    def analyze_activity_pattern(self, landmarks_dict, angles):
        """Análisis avanzado de patrones de actividad."""
        if not landmarks_dict:
            return "Parado", 0.0
        
        # Calcular múltiples métricas
        movement_score = self.calculate_movement_advanced(landmarks_dict)
        depth_change = self.calculate_depth_change(landmarks_dict)
        rotation_score = self.calculate_rotation(landmarks_dict)
        head_movement = self.calculate_head_movement(landmarks_dict)
        posture_score = self.analyze_posture_advanced(angles)
        
        # Análisis temporal avanzado
        temporal_pattern = self.analyze_temporal_patterns_advanced()
        
        # Clasificación multimodal mejorada
        activity, confidence = self.classify_multimodal_enhanced(
            movement_score, depth_change, rotation_score, head_movement,
            posture_score, temporal_pattern, angles
        )
        
        # Filtro de estabilidad
        activity, confidence = self.apply_stability_filter(activity, confidence)
        
        return activity, confidence
    
    def calculate_movement_advanced(self, landmarks_dict):
        """Cálculo de movimiento con pesos específicos."""
        if not hasattr(self, 'prev_landmarks') or not self.prev_landmarks:
            self.prev_landmarks = landmarks_dict
            return 0.0
        
        total_movement = 0.0
        joint_count = 0
        
        # Pesos específicos por articulación
        joint_weights = {
            'LEFT_HIP': 2.5,      # Más peso a caderas
            'RIGHT_HIP': 2.5,
            'LEFT_SHOULDER': 1.8,  
            'RIGHT_SHOULDER': 1.8,
            'LEFT_KNEE': 1.2,     
            'RIGHT_KNEE': 1.2,
            'NOSE': 1.5,          # Cabeza importante para giros
        }
        
        for joint, weight in joint_weights.items():
            if joint in landmarks_dict and joint in self.prev_landmarks:
                curr = landmarks_dict[joint]
                prev = self.prev_landmarks[joint]
                
                dx = curr['x'] - prev['x']
                dy = curr['y'] - prev['y']
                movement = np.sqrt(dx*dx + dy*dy) * weight
                
                total_movement += movement
                joint_count += weight
        
        self.prev_landmarks = landmarks_dict
        
        avg_movement = total_movement / max(joint_count, 1)
        self.movement_history.append(avg_movement)
        
        if len(self.movement_history) > self.max_history:
            self.movement_history.pop(0)
        
        return avg_movement
    
    def calculate_depth_change(self, landmarks_dict):
        """Detectar acercamiento/alejamiento por ancho de hombros."""
        if not landmarks_dict:
            return 0.0
        
        if 'LEFT_SHOULDER' in landmarks_dict and 'RIGHT_SHOULDER' in landmarks_dict:
            left_shoulder = landmarks_dict['LEFT_SHOULDER']
            right_shoulder = landmarks_dict['RIGHT_SHOULDER']
            
            # Distancia entre hombros (proxy de profundidad)
            shoulder_distance = abs(left_shoulder['x'] - right_shoulder['x'])
            
            self.shoulder_width_history.append(shoulder_distance)
            if len(self.shoulder_width_history) > self.max_history:
                self.shoulder_width_history.pop(0)
            
            # Calcular tendencia
            if len(self.shoulder_width_history) >= 10:
                recent = self.shoulder_width_history[-8:]
                older = self.shoulder_width_history[-16:-8] if len(self.shoulder_width_history) >= 16 else recent
                
                recent_avg = np.mean(recent)
                older_avg = np.mean(older)
                
                # Positivo = acercándose, Negativo = alejándose
                depth_change = recent_avg - older_avg
                return depth_change
        
        return 0.0
    
    def calculate_rotation(self, landmarks_dict):
        """Detectar rotación por orientación de hombros."""
        if not landmarks_dict:
            return 0.0
        
        if 'LEFT_SHOULDER' in landmarks_dict and 'RIGHT_SHOULDER' in landmarks_dict:
            left_shoulder = landmarks_dict['LEFT_SHOULDER']
            right_shoulder = landmarks_dict['RIGHT_SHOULDER']
            
            # Ángulo de inclinación de hombros
            dx = right_shoulder['x'] - left_shoulder['x']
            dy = right_shoulder['y'] - left_shoulder['y']
            shoulder_angle = np.arctan2(dy, dx)
            
            if not hasattr(self, 'prev_shoulder_angle'):
                self.prev_shoulder_angle = shoulder_angle
                return 0.0
            
            # Cambio en ángulo
            angle_change = abs(shoulder_angle - self.prev_shoulder_angle)
            self.prev_shoulder_angle = shoulder_angle
            
            return angle_change
        
        return 0.0
    
    def calculate_head_movement(self, landmarks_dict):
        """Detectar movimiento de cabeza para giros."""
        if 'NOSE' not in landmarks_dict:
            return 0.0
        
        nose_pos = landmarks_dict['NOSE']
        current_head_pos = np.array([nose_pos['x'], nose_pos['y']])
        
        self.head_position_history.append(current_head_pos)
        if len(self.head_position_history) > self.max_history:
            self.head_position_history.pop(0)
        
        # Movimiento lateral de cabeza
        if len(self.head_position_history) >= 6:
            recent_positions = self.head_position_history[-6:]
            head_movement_x = np.std([pos[0] for pos in recent_positions])
            return head_movement_x
        
        return 0.0
    
    def analyze_posture_advanced(self, angles):
        """Análisis de postura avanzado."""
        if not angles:
            return {"type": "standing", "confidence": 0.5}
        
        left_knee = angles.get('LEFT_KNEE_ANGLE', 180)
        right_knee = angles.get('RIGHT_KNEE_ANGLE', 180)
        avg_knee_angle = (left_knee + right_knee) / 2
        
        if avg_knee_angle < 100:
            return {"type": "sitting", "confidence": 0.9}
        elif avg_knee_angle < 130:
            return {"type": "squatting", "confidence": 0.8}
        elif avg_knee_angle < 160:
            return {"type": "crouching", "confidence": 0.7}
        else:
            return {"type": "standing", "confidence": 0.8}
    
    def analyze_temporal_patterns_advanced(self):
        """Análisis temporal con múltiples métricas."""
        if len(self.movement_history) < 8:
            return {"pattern": "insufficient_data", "confidence": 0.0}
        
        recent_movement = self.movement_history[-8:]
        movement_mean = np.mean(recent_movement)
        movement_std = np.std(recent_movement)
        
        # Tendencia de profundidad
        depth_trend = 0.0
        if len(self.shoulder_width_history) >= 8:
            recent_depth = self.shoulder_width_history[-8:]
            depth_trend = np.polyfit(range(len(recent_depth)), recent_depth, 1)[0]
        
        # Movimiento de cabeza
        head_movement = 0.0
        if len(self.head_position_history) >= 6:
            recent_head = self.head_position_history[-6:]
            head_movement = np.std([pos[0] for pos in recent_head])
        
        # Clasificar patrón
        if movement_mean > self.thresholds['movement_walking'] and movement_std > 0.003:
            if depth_trend > self.thresholds['depth_approaching']:
                return {"pattern": "walking_toward", "confidence": 0.85, "depth_trend": depth_trend}
            elif depth_trend < -self.thresholds['depth_approaching']:
                return {"pattern": "walking_away", "confidence": 0.85, "depth_trend": depth_trend}
            else:
                return {"pattern": "walking_lateral", "confidence": 0.75}
        elif movement_mean > self.thresholds['movement_turning'] and head_movement > self.thresholds['head_turn_threshold']:
            return {"pattern": "turning", "confidence": 0.80, "head_movement": head_movement}
        elif movement_mean > 0.005:
            return {"pattern": "moving", "confidence": 0.6}
        else:
            return {"pattern": "static", "confidence": 0.8}
    
    def classify_multimodal_enhanced(self, movement_score, depth_change, rotation_score, 
                                   head_movement, posture_score, temporal_pattern, angles):
        """Clasificación multimodal mejorada."""
        
        # Análisis de transiciones de postura
        left_knee = angles.get('LEFT_KNEE_ANGLE', 180)
        right_knee = angles.get('RIGHT_KNEE_ANGLE', 180)
        avg_knee = (left_knee + right_knee) / 2
        
        self.angle_history.append(avg_knee)
        if len(self.angle_history) > self.max_history:
            self.angle_history.pop(0)
        
        # Detectar transiciones de postura
        if len(self.angle_history) >= 8:
            recent_angles = self.angle_history[-8:]
            angle_trend = recent_angles[-1] - recent_angles[0]
            angle_velocity = np.mean(np.diff(recent_angles))
            
            # Sentarse: ángulos disminuyen rápidamente
            if angle_trend < -20 and avg_knee < 130 and angle_velocity < -1.5:
                return "Sentarse", min(92, abs(angle_trend) * 3)
            
            # Levantarse: ángulos aumentan rápidamente
            if angle_trend > 20 and avg_knee > 130 and angle_velocity > 1.5:
                return "Levantarse", min(92, angle_trend * 3)
        
        # Clasificación basada en patrón temporal
        pattern = temporal_pattern.get("pattern", "static")
        pattern_confidence = temporal_pattern.get("confidence", 0.0)
        
        if pattern == "walking_toward":
            # CAMINAR HACIA CÁMARA: Mejorada
            base_confidence = pattern_confidence * 85
            depth_bonus = temporal_pattern.get("depth_trend", 0) * 800
            confidence = min(90, base_confidence + depth_bonus)
            if confidence > 65:  # Umbral más bajo
                return "Caminar hacia cámara", confidence
            
        elif pattern == "walking_away":
            # CAMINAR DESDE CÁMARA: Mejorada
            base_confidence = pattern_confidence * 85
            depth_bonus = abs(temporal_pattern.get("depth_trend", 0)) * 800
            confidence = min(90, base_confidence + depth_bonus)
            if confidence > 65:  # Umbral más bajo
                return "Caminar desde cámara", confidence
            
        elif pattern == "turning":
            # GIRAR: Mucho más sensible
            base_confidence = pattern_confidence * 75
            head_bonus = temporal_pattern.get("head_movement", 0) * 2500
            rotation_bonus = rotation_score * 1500
            confidence = min(88, base_confidence + head_bonus + rotation_bonus)
            if confidence > 55:  # Umbral más bajo para giros
                return "Girar", confidence
            
        elif pattern == "walking_lateral":
            # Movimiento lateral - decidir dirección
            if depth_change > 0.003:
                return "Caminar hacia cámara", min(78, movement_score * 2200)
            elif depth_change < -0.003:
                return "Caminar desde cámara", min(78, movement_score * 2200)
            else:
                # Si no hay cambio de profundidad, puede ser giro
                if head_movement > self.thresholds['head_turn_threshold'] * 0.7:
                    return "Girar", min(75, head_movement * 2000)
                else:
                    return "Caminar hacia cámara", min(70, movement_score * 2000)
        
        # Estados estáticos
        if posture_score["type"] == "sitting" and posture_score["confidence"] > 0.8:
            return "Sentarse", min(85, posture_score["confidence"] * 95)
        
        # Default: Parado
        static_confidence = max(75, 100 - movement_score * 1800)
        return "Parado", static_confidence
    
    def apply_stability_filter(self, activity, confidence):
        """Filtro de estabilidad para evitar cambios bruscos."""
        # Si es la misma actividad, aumentar confianza
        if activity == self.current_state:
            self.state_frames += 1
            # Bonus por estabilidad
            stability_bonus = min(8, self.state_frames * 1.5)
            confidence = min(100, confidence + stability_bonus)
        else:
            # Nueva actividad detectada
            if confidence > 60 or self.state_frames < 4:  # Umbral más bajo
                self.current_state = activity
                self.state_confidence = confidence
                self.state_frames = 1
            else:
                # Mantener estado anterior
                activity = self.current_state
                confidence = max(50, self.state_confidence - 3)
        
        return activity, confidence

class OptimizedVideoThread(QThread):
    """Thread optimizado con detección mejorada."""
    
    frame_processed = pyqtSignal(object, object, object, float, str, float)
    
    def __init__(self, cap, angle_analyzer):
        super().__init__()
        self.cap = cap
        self.angle_analyzer = angle_analyzer
        self.video_processor = FixedVideoProcessor()
        self.activity_classifier = EnhancedActivityClassifier()  # USAR EL MEJORADO
        self.running = False
        
        # Configuraciones para mejor detección
        self.target_fps = 20  # Reducido un poco para procesar mejor
        self.frame_time = 1.0 / self.target_fps
        
    def run(self):
        """Procesamiento con detección mejorada."""
        self.running = True
        
        while self.running and self.cap and self.cap.isOpened():
            loop_start = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                break
            
            processing_start = time.time()
            
            # Procesar landmarks
            landmarks_dict = self.video_processor.extract_landmarks_fixed(frame)
            
            angles = {}
            activity = "Parado"
            confidence = 0.0
            
            if landmarks_dict:
                # Dibujar esqueleto
                frame = self.video_processor.draw_skeleton_fixed(frame, landmarks_dict)
                
                # Calcular ángulos
                angles = self.calculate_comprehensive_angles(landmarks_dict)
                
                # CLASIFICAR CON EL ALGORITMO MEJORADO
                activity, confidence = self.activity_classifier.analyze_activity_pattern(
                    landmarks_dict, angles
                )
            
            processing_time = (time.time() - processing_start) * 1000
            
            # Emitir señal
            self.frame_processed.emit(
                frame, landmarks_dict, angles, processing_time, str(activity), float(confidence)
            )
            
            # Control de FPS
            elapsed = time.time() - loop_start
            if elapsed < self.frame_time:
                time.sleep(self.frame_time - elapsed)
    
    def calculate_comprehensive_angles(self, landmarks_dict):
        """Calcular ángulos comprehensivos."""
        angles = {}
        
        try:
            # Ángulos de rodillas
            for side in ['LEFT', 'RIGHT']:
                if all(key in landmarks_dict for key in [f'{side}_HIP', f'{side}_KNEE', f'{side}_ANKLE']):
                    hip = landmarks_dict[f'{side}_HIP']
                    knee = landmarks_dict[f'{side}_KNEE']
                    ankle = landmarks_dict[f'{side}_ANKLE']
                    
                    v1 = np.array([hip['x'] - knee['x'], hip['y'] - knee['y']])
                    v2 = np.array([ankle['x'] - knee['x'], ankle['y'] - knee['y']])
                    
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                    
                    angles[f'{side}_KNEE_ANGLE'] = angle
            
            # Ángulos de codos
            for side in ['LEFT', 'RIGHT']:
                if all(key in landmarks_dict for key in [f'{side}_SHOULDER', f'{side}_ELBOW', f'{side}_WRIST']):
                    shoulder = landmarks_dict[f'{side}_SHOULDER']
                    elbow = landmarks_dict[f'{side}_ELBOW']
                    wrist = landmarks_dict[f'{side}_WRIST']
                    
                    v1 = np.array([shoulder['x'] - elbow['x'], shoulder['y'] - elbow['y']])
                    v2 = np.array([wrist['x'] - elbow['x'], wrist['y'] - elbow['y']])
                    
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
                    
                    angles[f'{side}_ELBOW_ANGLE'] = angle
            
            # Inclinación del tronco
            if all(key in landmarks_dict for key in ['LEFT_SHOULDER', 'RIGHT_SHOULDER']):
                left_shoulder = landmarks_dict['LEFT_SHOULDER']
                right_shoulder = landmarks_dict['RIGHT_SHOULDER']
                
                shoulder_vector = np.array([
                    right_shoulder['x'] - left_shoulder['x'],
                    right_shoulder['y'] - left_shoulder['y']
                ])
                horizontal = np.array([1, 0])
                
                cos_angle = np.dot(shoulder_vector, horizontal) / (np.linalg.norm(shoulder_vector) + 1e-8)
                trunk_angle = np.degrees(np.arccos(np.clip(abs(cos_angle), -1.0, 1.0)))
                
                angles['TRUNK_TILT'] = trunk_angle
        except:
            pass
        
        return angles
    
    def stop(self):
        """Detener thread."""
        self.running = False
        self.wait()

class OptimizedVideoAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.title = 'Sistema con Detección MEJORADA de Actividades'
        self.left = 50
        self.top = 50
        self.width = 1200
        self.height = 800
        
        self.is_capturing = False
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        self.processing_thread = None
        
        self.angle_analyzer = AngleAnalyzer()
        
        self.init_ui()
        self.setup_timers()
    
    def init_ui(self):
        """Interfaz optimizada."""
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        main_splitter = QSplitter(Qt.Horizontal)
        
        self.video_display = ModernVideoDisplay()
        main_splitter.addWidget(self.video_display)
        
        right_panel = QSplitter(Qt.Vertical)
        
        self.control_panel = AdvancedControlPanel()
        self.info_panel = AdvancedInfoPanel()
        
        right_panel.addWidget(self.control_panel)
        right_panel.addWidget(self.info_panel)
        right_panel.setSizes([300, 400])
        
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([900, 300])
        
        self.control_panel.start_capture_signal.connect(self.start_capture)
        self.control_panel.stop_capture_signal.connect(self.stop_capture)
        
        self.setCentralWidget(main_splitter)
    
    def setup_timers(self):
        """Timers optimizados."""
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.update_fps)
        self.fps_timer.start(2000)
    
    def start_capture(self, camera_idx):
        """Iniciar captura con detección mejorada."""
        try:
            self.cap = cv2.VideoCapture(camera_idx)
            
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                self.is_capturing = True
                self.frame_count = 0
                
                self.processing_thread = OptimizedVideoThread(self.cap, self.angle_analyzer)
                self.processing_thread.frame_processed.connect(self.on_frame_processed)
                self.processing_thread.start()
                
                print(f"Captura MEJORADA iniciada en cámara {camera_idx}")
                print("🎯 MEJORAS IMPLEMENTADAS:")
                print("✅ Detección de 'Caminar hacia cámara' mejorada")
                print("✅ Detección de 'Girar' mucho más sensible")
                print("✅ Algoritmo multimodal con análisis temporal")
                print("✅ Umbrales calibrados para mejor precisión")
                print("✅ Filtro de estabilidad para evitar cambios bruscos")
            else:
                print(f"No se pudo abrir la cámara {camera_idx}")
                
        except Exception as e:
            print(f"Error al iniciar captura: {e}")
    
    def stop_capture(self):
        """Detener captura."""
        self.is_capturing = False
        
        if self.processing_thread:
            self.processing_thread.stop()
            self.processing_thread = None
        
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
        
        print("Captura detenida")
    
    def on_frame_processed(self, frame, landmarks_dict, angles, processing_time, activity, confidence):
        """Manejo optimizado de frames."""
        if not self.is_capturing:
            return
        
        self.frame_count += 1
        
        try:
            self.video_display.update_frame(frame)
            
            if hasattr(self.video_display, 'update_activity_info'):
                self.video_display.update_activity_info(activity, confidence)
            if hasattr(self.video_display, 'update_angles'):
                self.video_display.update_angles(angles)
            
            self.info_panel.update_activity(activity, confidence)
            if hasattr(self.info_panel, 'update_angles'):
                self.info_panel.update_angles(angles)
            if hasattr(self.info_panel, 'update_performance_metrics'):
                self.info_panel.update_performance_metrics(processing_time, self.frame_count)
                
        except Exception as e:
            print(f"Error actualizando interfaz: {e}")
    
    def update_fps(self):
        """Actualizar FPS."""
        current_time = time.time()
        elapsed = current_time - self.last_time
        
        if elapsed > 0 and self.frame_count > 0:
            self.fps = self.frame_count / elapsed
            
            if hasattr(self.video_display, 'update_fps'):
                self.video_display.update_fps(self.fps)
            if hasattr(self.info_panel, 'update_fps'):
                self.info_panel.update_fps(self.fps)
            
            self.frame_count = 0
            self.last_time = current_time
    
    def closeEvent(self, event):
        """Cierre limpio."""
        if self.is_capturing:
            self.stop_capture()
        event.accept()

def main():
    """Función principal con detección mejorada."""
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    
    os.makedirs("data", exist_ok=True)
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    try:
        window = OptimizedVideoAnalyzerApp()
        window.show()
        
        print("=== Sistema con Detección MEJORADA de Actividades ===")
        print("🎯 NUEVAS CAPACIDADES:")
        print("✅ CAMINAR HACIA CÁMARA: Detecta por aumento en ancho de hombros")
        print("✅ GIRAR: Detecta por movimiento lateral de cabeza + rotación")
        print("✅ CAMINAR DESDE CÁMARA: Detecta por disminución en ancho de hombros")
        print("✅ Análisis temporal: 20 frames de historial")
        print("✅ Múltiples métricas: Movimiento + Profundidad + Rotación + Cabeza")
        print("✅ Umbrales calibrados más sensibles")
        print("✅ Filtro de estabilidad para evitar flickering")
        print("\n🔧 PARA AJUSTAR SENSIBILIDAD:")
        print("- Edita los valores en self.thresholds de EnhancedActivityClassifier")
        print("- Valores más bajos = más sensible")
        print("- Valores más altos = menos sensible")
        
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()