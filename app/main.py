# app/main.py
import sys
import os
import cv2
import numpy as np
import mediapipe as mp
import pickle
import json
import time
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QWidget, QFrame
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

# Agregar directorio src al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar módulos propios
from src.video_processor import SmartVideoProcessor

class VideoAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Configuración inicial
        self.title = 'Analizador de Actividades Físicas'
        self.left = 100
        self.top = 100
        self.width = 1200
        self.height = 800
        
        # Inicializar componentes
        self.init_ui()
        
        # Inicializar procesador de video
        self.init_video_processor()
        
        # Cargar modelo entrenado
        self.load_model()
        
        # Variables de estado
        self.is_capturing = False
        self.current_activity = "Desconocida"
        self.current_angles = {}
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        
        # Iniciar temporizador para actualizar FPS
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_fps)
        self.timer.start(1000)  # Actualizar cada segundo
    
    def init_ui(self):
        # Configurar ventana principal
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        # Crear layout principal
        main_layout = QHBoxLayout()
        
        # Panel izquierdo (video)
        self.video_panel = QLabel()
        self.video_panel.setMinimumSize(640, 480)
        self.video_panel.setAlignment(Qt.AlignCenter)
        self.video_panel.setStyleSheet("background-color: black;")
        
        # Panel derecho (información y controles)
        right_panel = QVBoxLayout()
        
        # Sección de información
        info_section = QVBoxLayout()
        self.activity_label = QLabel("Actividad: Desconocida")
        self.activity_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.fps_label = QLabel("FPS: 0")
        self.confidence_label = QLabel("Confianza: 0%")
        
        info_section.addWidget(QLabel("<h2>Información</h2>"))
        info_section.addWidget(self.activity_label)
        info_section.addWidget(self.confidence_label)
        info_section.addWidget(self.fps_label)
        
        # Sección de ángulos articulares
        angles_section = QVBoxLayout()
        angles_section.addWidget(QLabel("<h2>Ángulos Articulares</h2>"))
        
        # Crear labels para ángulos
        self.angle_labels = {}
        for joint in ['Rodilla Izquierda', 'Rodilla Derecha', 'Inclinación Tronco']:
            label = QLabel(f"{joint}: 0°")
            angles_section.addWidget(label)
            self.angle_labels[joint] = label
        
   # Sección de controles
        controls_section = QVBoxLayout()
        controls_section.addWidget(QLabel("<h2>Controles</h2>"))
        
        # Botón para iniciar/detener captura
        self.capture_button = QPushButton("Iniciar Captura")
        self.capture_button.clicked.connect(self.toggle_capture)
        controls_section.addWidget(self.capture_button)
        
        # Selector de cámara
        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("Cámara:"))
        self.camera_selector = QComboBox()
        self.camera_selector.addItems([f"Cámara {i}" for i in range(3)])  # Hasta 3 cámaras
        camera_layout.addWidget(self.camera_selector)
        controls_section.addLayout(camera_layout)
        
        # Selector de actividad para grabación
        activity_layout = QHBoxLayout()
        activity_layout.addWidget(QLabel("Actividad:"))
        self.activity_selector = QComboBox()
        self.activity_selector.addItems(["Caminar hacia", "Caminar desde", "Girar", "Sentarse", "Levantarse"])
        activity_layout.addWidget(self.activity_selector)
        controls_section.addLayout(activity_layout)
        
        # Botón para grabar actividad
        self.record_button = QPushButton("Grabar Actividad")
        self.record_button.clicked.connect(self.record_activity)
        self.record_button.setEnabled(False)  # Deshabilitado hasta iniciar captura
        controls_section.addWidget(self.record_button)
        
        # Añadir secciones al panel derecho
        right_panel.addLayout(info_section)
        right_panel.addLayout(angles_section)
        right_panel.addLayout(controls_section)
        right_panel.addStretch()
        
        # Añadir paneles al layout principal
        main_layout.addWidget(self.video_panel, 2)
        
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        main_layout.addWidget(right_widget, 1)
        
        # Configurar widget central
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
    
    def init_video_processor(self):
        """Inicializa el procesador de video y MediaPipe."""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Inicializar captura de video
        self.cap = None
    
    def load_model(self):
        """Carga el modelo entrenado y el scaler."""
        try:
            # Directorio del modelo
            model_dir = Path("data/dataset_processed/model")
            
            # Cargar modelo
            with open(model_dir / "random_forest.pkl", "rb") as f:
                self.model = pickle.load(f)
            
            # Cargar scaler
            with open(model_dir / "scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)
            
            # Cargar metadatos
            with open(model_dir / "model_metadata.json", "r") as f:
                self.model_metadata = json.load(f)
            
            print("Modelo cargado correctamente")
            self.feature_names = self.model_metadata.get('feature_names', [])
            
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            self.model = None
            self.scaler = None
            self.model_metadata = {}
            self.feature_names = []
    
    def toggle_capture(self):
        """Inicia o detiene la captura de video."""
        if self.is_capturing:
            # Detener captura
            self.is_capturing = False
            self.capture_button.setText("Iniciar Captura")
            self.record_button.setEnabled(False)
            
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        else:
            # Iniciar captura
            camera_idx = self.camera_selector.currentIndex()
            self.cap = cv2.VideoCapture(camera_idx)
            
            if self.cap.isOpened():
                self.is_capturing = True
                self.capture_button.setText("Detener Captura")
                self.record_button.setEnabled(True)
                
                # Iniciar procesamiento de frames
                self.process_frames()
            else:
                print(f"No se pudo abrir la cámara {camera_idx}")
                self.cap = None
    
    def process_frames(self):
        """Procesa frames de video en un bucle."""
        if not self.is_capturing or self.cap is None:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            print("Error al leer frame")
            self.toggle_capture()  # Detener captura
            return
        
        # Incrementar contador de frames
        self.frame_count += 1
        
        # Convertir a RGB para MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar frame con MediaPipe
        results = self.pose.process(frame_rgb)
        
        # Si se detectaron landmarks
        if results.pose_landmarks:
            # Dibujar landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
            
            # Extraer características
            features = self.extract_features(results.pose_landmarks)
            
            # Clasificar actividad si hay modelo
            if self.model is not None and features is not None:
                self.classify_activity(features)
            
            # Calcular ángulos articulares
            self.calculate_angles(results.pose_landmarks)
            
            # Actualizar información en UI
            self.update_angle_labels()
        
        # Mostrar frame procesado
        self.update_video_frame(frame)
        
        # Programar siguiente frame
        QTimer.singleShot(1, self.process_frames)
    
    def extract_features(self, pose_landmarks):
        """Extrae características relevantes de los landmarks."""
        try:
            # Diccionario para almacenar características
            features = {}
            
            # Landmarks clave
            landmarks_dict = {}
            for landmark_name in ['LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 
                                 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER',
                                 'LEFT_WRIST', 'RIGHT_WRIST', 'NOSE']:
                idx = getattr(self.mp_pose.PoseLandmark, landmark_name)
                landmark = pose_landmarks.landmark[idx]
                landmarks_dict[landmark_name] = {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                }
            
            # Calcular ángulos de rodilla
            left_knee_angle = self.calculate_angle(
                landmarks_dict['LEFT_HIP'],
                landmarks_dict['LEFT_KNEE'],
                landmarks_dict['LEFT_ANKLE']
            )
            right_knee_angle = self.calculate_angle(
                landmarks_dict['RIGHT_HIP'],
                landmarks_dict['RIGHT_KNEE'],
                landmarks_dict['RIGHT_ANKLE']
            )
            
            # Calcular inclinación lateral
            left_shoulder_x = landmarks_dict['LEFT_SHOULDER']['x']
            right_shoulder_x = landmarks_dict['RIGHT_SHOULDER']['x']
            left_hip_x = landmarks_dict['LEFT_HIP']['x']
            right_hip_x = landmarks_dict['RIGHT_HIP']['x']
            lateral_inclination = self.calculate_lateral_inclination(
                left_shoulder_x, right_shoulder_x, left_hip_x, right_hip_x)
            
            # Crear vector de características
            features['LEFT_KNEE_ANGLE'] = left_knee_angle
            features['RIGHT_KNEE_ANGLE'] = right_knee_angle
            features['LEFT_WRIST_X'] = landmarks_dict['LEFT_WRIST']['x']
            features['RIGHT_WRIST_X'] = landmarks_dict['RIGHT_WRIST']['x']
            features['LEFT_SHOULDER_X'] = left_shoulder_x
            features['RIGHT_SHOULDER_X'] = right_shoulder_x
            features['NOSE_X'] = landmarks_dict['NOSE']['x']
            features['LATERAL_INCLINATION'] = lateral_inclination
            
            # Completar con valores 0 para otras características que espera el modelo
            for name in self.feature_names:
                if name not in features:
                    features[name] = 0.0
            
            # Crear vector de características en el orden que espera el modelo
            feature_vector = np.array([features.get(name, 0.0) for name in self.feature_names])
            
            return feature_vector
            
        except Exception as e:
            print(f"Error al extraer características: {e}")
            return None
    
    def calculate_angle(self, p1, p2, p3):
        """Calcula el ángulo entre tres puntos."""
        try:
            v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
            v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
            
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm > 0 and v2_norm > 0:
                cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_angle))
                return angle
            return 0.0
        except:
            return 0.0
    
    def calculate_lateral_inclination(self, left_shoulder, right_shoulder, left_hip, right_hip):
        """Calcula inclinación lateral basada en posiciones de hombros y caderas."""
        try:
            shoulder_mid = (left_shoulder + right_shoulder) / 2
            hip_mid = (left_hip + right_hip) / 2
            return shoulder_mid - hip_mid
        except:
            return 0.0
    
    def calculate_angles(self, pose_landmarks):
        """Calcula ángulos articulares para mostrar en la interfaz."""
        try:
            # Ángulo de rodilla izquierda
            left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            left_knee = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE]
            left_ankle = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            
            left_knee_angle = self.calculate_angle(
                {'x': left_hip.x, 'y': left_hip.y},
                {'x': left_knee.x, 'y': left_knee.y},
                {'x': left_ankle.x, 'y': left_ankle.y}
            )
            
            # Ángulo de rodilla derecha
            right_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            right_knee = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE]
            right_ankle = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            
            right_knee_angle = self.calculate_angle(
                {'x': right_hip.x, 'y': right_hip.y},
                {'x': right_knee.x, 'y': right_knee.y},
                {'x': right_ankle.x, 'y': right_ankle.y}
            )
            
            # Inclinación del tronco
            left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            
            # Vector hombro
            shoulder_vector = np.array([right_shoulder.x - left_shoulder.x, right_shoulder.y - left_shoulder.y])
            
            # Vector horizontal
            horizontal = np.array([1, 0])
            
            # Calcular ángulo
            shoulder_norm = np.linalg.norm(shoulder_vector)
            if shoulder_norm > 0:
                cos_angle = np.clip(np.dot(shoulder_vector, horizontal) / shoulder_norm, -1.0, 1.0)
                trunk_angle = np.degrees(np.arccos(cos_angle))
            else:
                trunk_angle = 0.0
            
            # Actualizar ángulos
            self.current_angles = {
                'Rodilla Izquierda': left_knee_angle,
                'Rodilla Derecha': right_knee_angle,
                'Inclinación Tronco': trunk_angle
            }
            
        except Exception as e:
            print(f"Error al calcular ángulos: {e}")
    
    def update_angle_labels(self):
        """Actualiza las etiquetas de ángulos en la interfaz."""
        for joint, label in self.angle_labels.items():
            angle = self.current_angles.get(joint, 0.0)
            label.setText(f"{joint}: {angle:.1f}°")
    
    def classify_activity(self, features):
        """Clasifica la actividad actual a partir de las características extraídas."""
        try:
            # Normalizar características usando el scaler
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Predecir actividad
            activity_idx = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Obtener confianza (probabilidad de la clase predicha)
            confidence = probabilities.max() * 100
            
            # Mapear índice a nombre de actividad
            activities = ["caminar_hacia", "caminar_regreso", "girar", "sentarse", "levantarse"]
            activity_names = {
                "caminar_hacia": "Caminar hacia cámara",
                "caminar_regreso": "Caminar desde cámara",
                "girar": "Girar",
                "sentarse": "Sentarse",
                "levantarse": "Levantarse"
            }
            
            # Actualizar actividad y confianza en la interfaz
            if isinstance(activity_idx, str) and activity_idx in activity_names:
                activity_name = activity_names[activity_idx]
            elif isinstance(activity_idx, (int, np.integer)) and 0 <= activity_idx < len(activities):
                activity_name = activity_names[activities[activity_idx]]
            else:
                activity_name = "Desconocida"
            
            self.current_activity = activity_name
            self.activity_label.setText(f"Actividad: {activity_name}")
            self.confidence_label.setText(f"Confianza: {confidence:.1f}%")
            
        except Exception as e:
            print(f"Error al clasificar actividad: {e}")
    
    def update_video_frame(self, frame):
        """Actualiza el frame de video en la interfaz."""
        # Convertir a formato RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Añadir texto de actividad
        cv2.putText(rgb_frame, f"Actividad: {self.current_activity}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convertir a QImage
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Escalar manteniendo proporción
        pixmap = QPixmap.fromImage(qt_image)
        pixmap = pixmap.scaled(self.video_panel.width(), self.video_panel.height(),
                              Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Actualizar etiqueta
        self.video_panel.setPixmap(pixmap)
    
    def update_fps(self):
        """Actualiza el contador de FPS."""
        current_time = time.time()
        elapsed = current_time - self.last_time
        
        if elapsed > 0:
            self.fps = self.frame_count / elapsed
            self.fps_label.setText(f"FPS: {self.fps:.1f}")
            
            # Reiniciar contador
            self.frame_count = 0
            self.last_time = current_time
    
    def record_activity(self):
        """Graba un video de la actividad seleccionada."""
        if not self.is_capturing or self.cap is None:
            return
        
        # Obtener actividad seleccionada
        activity = self.activity_selector.currentText().lower().replace(" ", "_")
        
        # Crear directorio para guardar videos si no existe
        output_dir = Path("data/dataset_raw")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Generar nombre de archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sujeto_01_{activity}_{timestamp}.mp4"
        output_path = output_dir / filename
        
        # Configurar grabador de video
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # TODO: Implementar grabación en segundo plano
        # Por ahora mostramos un mensaje
        print(f"Grabando actividad '{activity}' en {output_path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoAnalyzerApp()
    window.show()
    sys.exit(app.exec_())