# app/main_app.py
import cv2
import numpy as np
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

# Agregar directorio src al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar módulos propios
from src.video_processor import SmartVideoProcessor
from src.angle_analyzer import AngleAnalyzer
from src.feature_extractor import FeatureExtractor
from app.gui_components import VideoDisplay, ControlPanel, InfoPanel

class VideoAnalyzerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Configuración inicial
        self.title = 'Sistema de Anotación de Video'
        self.left = 100
        self.top = 100
        self.width = 1200
        self.height = 800
        
        # Inicializar componentes
        self.init_ui()
        
        # Inicializar procesador de video
        self.video_processor = SmartVideoProcessor()
        
        # Inicializar analizador de ángulos
        self.angle_analyzer = AngleAnalyzer()
        
        # Inicializar extractor de características
        self.feature_extractor = FeatureExtractor()
        
        # Cargar modelo entrenado si existe
        self.model = None
        self.scaler = None
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
        
        # Agregar componentes de interfaz
        self.video_display = VideoDisplay()
        self.control_panel = ControlPanel()
        self.info_panel = InfoPanel()
        
        # Conectar señales
        self.control_panel.start_capture_signal.connect(self.start_capture)
        self.control_panel.stop_capture_signal.connect(self.stop_capture)
        self.control_panel.record_signal.connect(self.record_activity)
        
        # Agregar al layout principal
        main_layout.addWidget(self.video_display, 2)
        
        # Panel derecho
        right_panel = QVBoxLayout()
        right_panel.addWidget(self.info_panel)
        right_panel.addWidget(self.control_panel)
        
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        main_layout.addWidget(right_widget, 1)
        
        # Configurar widget central
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
    
    def load_model(self):
        """Carga el modelo de clasificación si existe."""
        model_dir = Path("data/dataset_processed/model")
        
        if not model_dir.exists():
            print("Directorio de modelo no encontrado.")
            return
            
        try:
            # Intentar cargar modelo y scaler
            # Código existente para cargar modelo...
            print("Modelo cargado correctamente")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
    
    def start_capture(self, camera_idx):
        """Inicia la captura de video."""
        self.cap = cv2.VideoCapture(camera_idx)
        
        if self.cap.isOpened():
            self.is_capturing = True
            self.process_frames()
        else:
            print(f"No se pudo abrir la cámara {camera_idx}")
    
    def stop_capture(self):
        """Detiene la captura de video."""
        self.is_capturing = False
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
    
    def process_frames(self):
        """Procesa frames de video y realiza análisis."""
        if not self.is_capturing:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.stop_capture()
            return
            
        # Incrementar contador de frames
        self.frame_count += 1
        
        # Procesar frame con MediaPipe
        landmarks_dict = self.video_processor.extract_landmarks(frame)
        
        if landmarks_dict:
            # Dibujar landmarks
            frame = self.video_processor.draw_landmarks(frame, landmarks_dict)
            
            # Analizar ángulos articulares
            angle_results = self.angle_analyzer.analyze_joint_angles(landmarks_dict)
            
            # Actualizar ángulos actuales
            for joint, angle in angle_results.items():
                if isinstance(angle, (int, float)):  # Solo incluir valores numéricos
                    self.current_angles[joint] = angle
            
            # Dibujar ángulos en el frame
            frame = self.video_processor.draw_angles(frame, landmarks_dict, angle_results)
            
            # Clasificar actividad si hay modelo
            if self.model is not None:
                features = self.feature_extractor.extract_features(landmarks_dict)
                if features is not None:
                    self.classify_activity(features)
            else:
                # Identificar actividad basado en patrones de ángulos
                activity, confidence = self.angle_analyzer.identify_activity_pattern(
                    self.angle_analyzer.angle_history
                )
                self.current_activity = activity
                self.info_panel.update_activity(activity, confidence * 100)
            
            # Actualizar información en UI
            self.info_panel.update_angles(self.current_angles)
        
        # Mostrar frame procesado
        self.video_display.update_frame(frame)
        
        # Programar siguiente frame
        QTimer.singleShot(1, self.process_frames)
    
    def classify_activity(self, features):
        """Clasifica la actividad basada en las características extraídas."""
        if self.model is None or self.scaler is None:
            return
            
        try:
            # Normalizar características
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Predecir actividad
            activity_idx = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            # Obtener confianza
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
            
            # Actualizar actividad y confianza
            if isinstance(activity_idx, str) and activity_idx in activity_names:
                activity_name = activity_names[activity_idx]
            elif isinstance(activity_idx, (int, np.integer)) and 0 <= activity_idx < len(activities):
                activity_name = activity_names[activities[activity_idx]]
            else:
                activity_name = "Desconocida"
            
            self.current_activity = activity_name
            self.info_panel.update_activity(activity_name, confidence)
            
        except Exception as e:
            print(f"Error al clasificar actividad: {e}")
    
    def update_fps(self):
        """Actualiza el contador de FPS."""
        current_time = time.time()
        elapsed = current_time - self.last_time
        
        if elapsed > 0:
            self.fps = self.frame_count / elapsed
            self.info_panel.update_fps(self.fps)
            
            # Reiniciar contador
            self.frame_count = 0
            self.last_time = current_time
    
    def record_activity(self, activity):
        """Graba un video de la actividad seleccionada."""
        if not self.is_capturing or not hasattr(self, 'cap') or self.cap is None:
            return
        
        # Obtener actividad seleccionada
        activity = activity.lower().replace(" ", "_")
        
        # Crear directorio para guardar videos si no existe
        output_dir = Path("data/dataset_raw")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Generar nombre de archivo con timestamp
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

def main():
    app = QApplication(sys.argv)
    window = VideoAnalyzerApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()