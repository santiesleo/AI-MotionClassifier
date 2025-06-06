# app/main_fixed_optimized.py
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
from src.video_processor import SmartVideoProcessor
from src.angle_analyzer import AngleAnalyzer
from src.feature_extractor import FeatureExtractor

# Importar interfaz
try:
    from gui import ModernVideoDisplay, AdvancedControlPanel, AdvancedInfoPanel
    print("Interfaz mejorada cargada correctamente")
except ImportError as e:
    print(f"Error importando interfaz mejorada: {e}")
    from gui import VideoDisplay as ModernVideoDisplay, ControlPanel as AdvancedControlPanel, InfoPanel as AdvancedInfoPanel

class OptimizedVideoProcessor:
    """Procesador de video optimizado con esqueleto completo."""
    
    def __init__(self):
        # Configurar MediaPipe con parámetros optimizados
        import mediapipe as mp
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Configuración optimizada para rendimiento
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # Modelo medio para mejor precisión
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=0.6,  # Reducido para mejor detección
            min_tracking_confidence=0.5
        )
        
        # Cache para optimización
        self.frame_skip_counter = 0
        self.frame_skip_rate = 1  # Procesar todos los frames para mejor detección
        self.last_landmarks = None
        self.landmarks_cache_frames = 0
        self.max_cache_frames = 3
        
        # Todas las articulaciones para mejor análisis
        self.key_joints = [
            'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER',
            'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
            'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
            'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
            'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY',
            'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB',
            'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE',
            'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL',
            'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
        ]
        
        # Conexiones del esqueleto
        self.connections = [
            # Torso
            ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
            ('LEFT_SHOULDER', 'LEFT_HIP'),
            ('RIGHT_SHOULDER', 'RIGHT_HIP'),
            ('LEFT_HIP', 'RIGHT_HIP'),
            
            # Brazo izquierdo
            ('LEFT_SHOULDER', 'LEFT_ELBOW'),
            ('LEFT_ELBOW', 'LEFT_WRIST'),
            
            # Brazo derecho
            ('RIGHT_SHOULDER', 'RIGHT_ELBOW'),
            ('RIGHT_ELBOW', 'RIGHT_WRIST'),
            
            # Pierna izquierda
            ('LEFT_HIP', 'LEFT_KNEE'),
            ('LEFT_KNEE', 'LEFT_ANKLE'),
            
            # Pierna derecha
            ('RIGHT_HIP', 'RIGHT_KNEE'),
            ('RIGHT_KNEE', 'RIGHT_ANKLE'),
            
            # Cabeza
            ('NOSE', 'LEFT_EAR'),
            ('NOSE', 'RIGHT_EAR'),
        ]
    
    def extract_landmarks_optimized(self, frame):
        """Extrae landmarks con mejor precisión."""
        # Procesar todos los frames para mejor detección
        self.frame_skip_counter += 1
        
        # Usar frame original sin reducir resolución para mejor precisión
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            landmarks_dict = {}
            
            # Extraer todas las articulaciones disponibles
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
    
    def draw_skeleton_clean(self, frame, landmarks_dict):
        """Dibuja esqueleto limpio sin información superpuesta."""
        if not landmarks_dict:
            return frame
        
        h, w = frame.shape[:2]
        
        # Dibujar conexiones del esqueleto
        for start_joint, end_joint in self.connections:
            if (start_joint in landmarks_dict and end_joint in landmarks_dict and
                landmarks_dict[start_joint]['visibility'] > 0.5 and
                landmarks_dict[end_joint]['visibility'] > 0.5):
                
                start_x = int(landmarks_dict[start_joint]['x'] * w)
                start_y = int(landmarks_dict[start_joint]['y'] * h)
                end_x = int(landmarks_dict[end_joint]['x'] * w)
                end_y = int(landmarks_dict[end_joint]['y'] * h)
                
                # Líneas del esqueleto con colores diferenciados
                if 'ARM' in start_joint or 'ELBOW' in start_joint or 'WRIST' in start_joint:
                    color = (255, 165, 0)  # Naranja para brazos
                elif 'LEG' in start_joint or 'KNEE' in start_joint or 'ANKLE' in start_joint:
                    color = (0, 255, 255)  # Amarillo para piernas
                elif 'HIP' in start_joint or 'SHOULDER' in start_joint:
                    color = (0, 255, 0)    # Verde para torso
                else:
                    color = (255, 0, 255)  # Magenta para cabeza
                
                cv2.line(frame, (start_x, start_y), (end_x, end_y), color, 3)
        
        # Dibujar puntos clave
        for joint, data in landmarks_dict.items():
            if data['visibility'] > 0.5:
                x, y = int(data['x'] * w), int(data['y'] * h)
                
                # Colores según tipo de articulación
                if 'KNEE' in joint:
                    color = (0, 0, 255)    # Rojo para rodillas
                elif 'HIP' in joint:
                    color = (255, 0, 0)    # Azul para caderas
                elif 'SHOULDER' in joint:
                    color = (0, 255, 0)    # Verde para hombros
                elif 'ELBOW' in joint or 'WRIST' in joint:
                    color = (255, 165, 0)  # Naranja para brazos
                elif 'ANKLE' in joint:
                    color = (0, 255, 255)  # Amarillo para tobillos
                else:
                    color = (255, 255, 255) # Blanco para otros
                
                cv2.circle(frame, (x, y), 6, color, -1)
                cv2.circle(frame, (x, y), 6, (0, 0, 0), 2)  # Borde negro
        
        return frame

class ImprovedActivityClassifier:
    """Clasificador mejorado de actividades con mejor lógica."""
    
    def __init__(self):
        self.activity_names = {
            0: "Parado",
            1: "Caminar hacia cámara", 
            2: "Caminar desde cámara",
            3: "Girar",
            4: "Sentarse",
            5: "Levantarse"
        }
        
        # Historial para análisis temporal
        self.movement_history = []
        self.position_history = []
        self.angle_history = []
        self.max_history = 30  # 1 segundo a 30fps
        
    def analyze_activity_pattern(self, landmarks_dict, angles):
        """Analiza patrones de actividad basado en múltiples factores."""
        if not landmarks_dict:
            return "Parado", 0.0
        
        # Calcular métricas de movimiento
        movement_score = self.calculate_movement(landmarks_dict)
        posture_score = self.analyze_posture(angles)
        temporal_pattern = self.analyze_temporal_patterns()
        
        # Determinar actividad basada en análisis multimodal
        activity, confidence = self.classify_multimodal(
            movement_score, posture_score, temporal_pattern, angles
        )
        
        return activity, confidence
    
    def calculate_movement(self, landmarks_dict):
        """Calcula el nivel de movimiento general."""
        if not hasattr(self, 'prev_landmarks') or not self.prev_landmarks:
            self.prev_landmarks = landmarks_dict
            return 0.0
        
        total_movement = 0.0
        joint_count = 0
        
        # Calcular movimiento de articulaciones clave
        key_joints = ['LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP', 
                     'LEFT_KNEE', 'RIGHT_KNEE', 'NOSE']
        
        for joint in key_joints:
            if joint in landmarks_dict and joint in self.prev_landmarks:
                curr = landmarks_dict[joint]
                prev = self.prev_landmarks[joint]
                
                # Distancia euclidiana entre posiciones
                dx = curr['x'] - prev['x']
                dy = curr['y'] - prev['y']
                movement = np.sqrt(dx*dx + dy*dy)
                
                total_movement += movement
                joint_count += 1
        
        self.prev_landmarks = landmarks_dict
        
        # Agregar al historial
        avg_movement = total_movement / max(joint_count, 1)
        self.movement_history.append(avg_movement)
        
        if len(self.movement_history) > self.max_history:
            self.movement_history.pop(0)
        
        return avg_movement
    
    def analyze_posture(self, angles):
        """Analiza la postura basada en ángulos articulares."""
        if not angles:
            return "standing"
        
        left_knee = angles.get('LEFT_KNEE_ANGLE', 180)
        right_knee = angles.get('RIGHT_KNEE_ANGLE', 180)
        
        avg_knee_angle = (left_knee + right_knee) / 2
        
        # Clasificar postura
        if avg_knee_angle < 110:
            return "sitting"
        elif avg_knee_angle < 140:
            return "squatting"
        else:
            return "standing"
    
    def analyze_temporal_patterns(self):
        """Analiza patrones temporales de movimiento."""
        if len(self.movement_history) < 10:
            return "insufficient_data"
        
        recent_movement = self.movement_history[-10:]
        movement_variance = np.var(recent_movement)
        movement_avg = np.mean(recent_movement)
        
        # Detectar patrones
        if movement_avg > 0.02 and movement_variance > 0.0001:
            return "walking"
        elif movement_avg > 0.015 and movement_variance < 0.0001:
            return "turning"
        elif movement_avg > 0.01:
            return "moving"
        else:
            return "static"
    
    def classify_multimodal(self, movement_score, posture_score, temporal_pattern, angles):
        """Clasificación multimodal mejorada."""
        # Análisis de transiciones de postura
        left_knee = angles.get('LEFT_KNEE_ANGLE', 180)
        right_knee = angles.get('RIGHT_KNEE_ANGLE', 180)
        avg_knee = (left_knee + right_knee) / 2
        
        self.angle_history.append(avg_knee)
        if len(self.angle_history) > self.max_history:
            self.angle_history.pop(0)
        
        # Detectar cambios en ángulos para sentarse/levantarse
        if len(self.angle_history) >= 10:
            recent_angles = self.angle_history[-10:]
            angle_trend = recent_angles[-1] - recent_angles[0]
            
            # Sentarse: ángulos disminuyen significativamente
            if angle_trend < -30 and avg_knee < 120:
                return "Sentarse", min(95, abs(angle_trend) * 2)
            
            # Levantarse: ángulos aumentan significativamente
            if angle_trend > 30 and avg_knee > 140:
                return "Levantarse", min(95, angle_trend * 2)
        
        # Análisis de movimiento
        if temporal_pattern == "walking":
            if movement_score > 0.025:
                return "Caminar hacia cámara", min(90, movement_score * 3000)
            else:
                return "Caminar desde cámara", min(85, movement_score * 4000)
        
        elif temporal_pattern == "turning":
            return "Girar", min(80, movement_score * 5000)
        
        elif posture_score == "sitting":
            return "Sentarse", min(75, (180 - avg_knee) * 2)
        
        else:
            return "Parado", max(60, 100 - movement_score * 1000)

class OptimizedVideoThread(QThread):
    """Thread optimizado para procesamiento de video."""
    
    frame_processed = pyqtSignal(object, object, object, float, str, float)
    
    def __init__(self, cap, angle_analyzer):
        super().__init__()
        self.cap = cap
        self.angle_analyzer = angle_analyzer
        self.video_processor = OptimizedVideoProcessor()
        self.activity_classifier = ImprovedActivityClassifier()
        self.running = False
        
        # Configuraciones de optimización
        self.target_fps = 25
        self.frame_time = 1.0 / self.target_fps
        
    def run(self):
        """Ejecuta procesamiento optimizado."""
        self.running = True
        last_frame_time = time.time()
        
        while self.running and self.cap and self.cap.isOpened():
            loop_start = time.time()
            
            # Control de FPS
            elapsed = loop_start - last_frame_time
            if elapsed < self.frame_time:
                sleep_time = self.frame_time - elapsed
                self.msleep(int(sleep_time * 1000))
            
            processing_start = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Extraer landmarks
            landmarks_dict = self.video_processor.extract_landmarks_optimized(frame)
            
            angles = {}
            activity = "Parado"
            confidence = 0.0
            
            if landmarks_dict:
                # Dibujar esqueleto limpio (sin información superpuesta)
                frame = self.video_processor.draw_skeleton_clean(frame, landmarks_dict)
                
                # Calcular ángulos esenciales
                angles = self.calculate_comprehensive_angles(landmarks_dict)
                
                # Clasificar actividad con algoritmo mejorado
                activity, confidence = self.activity_classifier.analyze_activity_pattern(
                    landmarks_dict, angles
                )
            
            # Calcular tiempo de procesamiento
            processing_time = (time.time() - processing_start) * 1000
            
            # Emitir señal con activity como string
            self.frame_processed.emit(
                frame, landmarks_dict, angles, processing_time, str(activity), float(confidence)
            )
            
            last_frame_time = time.time()
    
    def calculate_comprehensive_angles(self, landmarks_dict):
        """Calcula ángulos comprehensivos para mejor análisis."""
        angles = {}
        
        try:
            # Ángulos de rodillas
            for side in ['LEFT', 'RIGHT']:
                if all(key in landmarks_dict for key in [f'{side}_HIP', f'{side}_KNEE', f'{side}_ANKLE']):
                    hip = landmarks_dict[f'{side}_HIP']
                    knee = landmarks_dict[f'{side}_KNEE']
                    ankle = landmarks_dict[f'{side}_ANKLE']
                    
                    # Vector de cadera a rodilla
                    v1 = np.array([hip['x'] - knee['x'], hip['y'] - knee['y']])
                    # Vector de rodilla a tobillo
                    v2 = np.array([ankle['x'] - knee['x'], ankle['y'] - knee['y']])
                    
                    # Calcular ángulo
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
                
                # Vector de hombros
                shoulder_vector = np.array([
                    right_shoulder['x'] - left_shoulder['x'],
                    right_shoulder['y'] - left_shoulder['y']
                ])
                horizontal = np.array([1, 0])
                
                cos_angle = np.dot(shoulder_vector, horizontal) / (np.linalg.norm(shoulder_vector) + 1e-8)
                trunk_angle = np.degrees(np.arccos(np.clip(abs(cos_angle), -1.0, 1.0)))
                
                angles['TRUNK_TILT'] = trunk_angle
                
        except Exception as e:
            print(f"Error calculando ángulos: {e}")
        
        return angles
    
    def stop(self):
        """Detiene el thread."""
        self.running = False
        self.wait()

class OptimizedVideoAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Configuración inicial
        self.title = 'Sistema Optimizado de Análisis de Actividades'
        self.left = 50
        self.top = 50
        self.width = 1400
        self.height = 900
        
        # Variables de estado
        self.is_capturing = False
        self.current_activity = "Parado"
        self.current_angles = {}
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        self.session_start_time = None
        self.processing_thread = None
        
        # Estadísticas
        self.activity_counter = {}
        self.confidence_history = []
        self.max_history = 100
        
        # Inicializar componentes
        self.init_processors()
        self.init_ui()
        self.setup_timers()
    
    def init_processors(self):
        """Inicializa procesadores de forma optimizada."""
        self.angle_analyzer = AngleAnalyzer()
        
        # Cargar modelo de forma asíncrona
        self.model = None
        self.scaler = None
        self.label_encoder = None
        QTimer.singleShot(1000, self.load_model_async)
    
    def load_model_async(self):
        """Carga el modelo de forma asíncrona."""
        model_dir = Path("data/dataset_processed/model")
        
        if not model_dir.exists():
            print("Directorio de modelo no encontrado. Usando clasificador basado en reglas.")
            return
        
        try:
            import pickle
            
            model_files = list(model_dir.glob("*.pkl"))
            model_file = None
            
            for file in model_files:
                if "scaler" not in file.name and "encoder" not in file.name:
                    model_file = file
                    break
            
            if model_file:
                with open(model_file, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"Modelo cargado: {model_file}")
            
            scaler_file = model_dir / "scaler.pkl"
            if scaler_file.exists():
                with open(scaler_file, 'rb') as f:
                    self.scaler = pickle.load(f)
                print("Scaler cargado")
            
            encoder_file = model_dir / "label_encoder.pkl"
            if encoder_file.exists():
                with open(encoder_file, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                print("Label encoder cargado")
                
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            print("Usando clasificador basado en reglas.")
    
    def init_ui(self):
        """Configura interfaz optimizada."""
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        main_splitter = QSplitter(Qt.Horizontal)
        
        self.video_display = ModernVideoDisplay()
        main_splitter.addWidget(self.video_display)
        
        right_panel = QSplitter(Qt.Vertical)
        
        self.info_panel = AdvancedInfoPanel()
        self.control_panel = AdvancedControlPanel()
        
        right_panel.addWidget(self.info_panel)
        right_panel.addWidget(self.control_panel)
        right_panel.setSizes([500, 300])
        
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([1000, 400])
        
        self.control_panel.start_capture_signal.connect(self.start_capture)
        self.control_panel.stop_capture_signal.connect(self.stop_capture)
        
        self.setCentralWidget(main_splitter)
        
        self.setStyleSheet("""
            QMainWindow { background-color: #f0f0f0; }
            QSplitter::handle { background-color: #ccc; width: 2px; }
        """)
    
    def setup_timers(self):
        """Configura temporizadores optimizados."""
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.update_fps)
        self.fps_timer.start(2000)
    
    def start_capture(self, camera_idx):
        """Inicia captura optimizada."""
        try:
            self.cap = cv2.VideoCapture(camera_idx)
            
            if self.cap.isOpened():
                # Configuración balanceada: resolución media para buen rendimiento y precisión
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                self.is_capturing = True
                self.session_start_time = datetime.now()
                self.frame_count = 0
                
                # Reiniciar estadísticas
                self.activity_counter = {}
                self.confidence_history = []
                
                # Thread optimizado
                self.processing_thread = OptimizedVideoThread(self.cap, self.angle_analyzer)
                self.processing_thread.frame_processed.connect(self.on_frame_processed)
                self.processing_thread.start()
                
                print(f"Captura optimizada iniciada en cámara {camera_idx}")
                print("Resolución: 800x600 para balance óptimo")
            else:
                print(f"No se pudo abrir la cámara {camera_idx}")
                
        except Exception as e:
            print(f"Error al iniciar captura: {e}")
    
    def stop_capture(self):
        """Detiene captura."""
        self.is_capturing = False
        
        if self.processing_thread:
            self.processing_thread.stop()
            self.processing_thread = None
        
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
        
        print("Captura detenida")
    
    def on_frame_processed(self, frame, landmarks_dict, angles, processing_time, activity, confidence):
        """Maneja frames con procesamiento optimizado."""
        if not self.is_capturing:
            return
        
        self.frame_count += 1
        self.current_angles = angles
        self.current_activity = str(activity)  # Ya viene como string del thread
        
        # Actualizar historial
        self.confidence_history.append(float(confidence))
        if len(self.confidence_history) > self.max_history:
            self.confidence_history.pop(0)
        
        # Actualizar contador de actividades
        if activity not in ["Parado", "Analizando..."]:
            if activity in self.activity_counter:
                self.activity_counter[activity] += 1
            else:
                self.activity_counter[activity] = 1
        
        # Actualizar interfaz
        try:
            self.video_display.update_frame(frame)
            
            if hasattr(self.video_display, 'update_activity_info'):
                self.video_display.update_activity_info(str(activity), float(confidence))
            if hasattr(self.video_display, 'update_angles'):
                self.video_display.update_angles(angles)
            
            self.info_panel.update_activity(str(activity), float(confidence))
            self.info_panel.update_angles(angles)
            
            if hasattr(self.info_panel, 'update_performance_metrics'):
                self.info_panel.update_performance_metrics(float(processing_time), int(self.frame_count))
                
        except Exception as e:
            print(f"Error actualizando interfaz: {e}")
    
    def update_fps(self):
        """Actualiza FPS de forma eficiente."""
        current_time = time.time()
        elapsed = current_time - self.last_time
        
        if elapsed > 0 and self.frame_count > 0:
            self.fps = self.frame_count / elapsed
            
            if hasattr(self.video_display, 'update_fps'):
                self.video_display.update_fps(self.fps)
            self.info_panel.update_fps(self.fps)
            
            self.frame_count = 0
            self.last_time = current_time
    
    def closeEvent(self, event):
        """Cierre limpio."""
        if self.is_capturing:
            self.stop_capture()
        event.accept()

def main():
    """Función principal optimizada."""
    os.environ['OMP_NUM_THREADS'] = '4'
    
    for directory in ["data", "data/dataset_raw", "data/dataset_processed"]:
        os.makedirs(directory, exist_ok=True)
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    try:
        window = OptimizedVideoAnalyzerApp()
        window.show()
        
        print("=== Sistema Optimizado de Análisis de Actividades ===")
        print("✅ Esqueleto completo activado")
        print("✅ Información de cámara removida")
        print("✅ Clasificador de actividades mejorado")
        print("✅ Detección de nombres de actividades corregida")
        print("Resolución: 800x600 para balance óptimo")
        print("FPS objetivo: 25")
        
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()