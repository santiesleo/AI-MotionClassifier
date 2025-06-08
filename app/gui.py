# app/gui_optimized.py
import cv2
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QComboBox, QRadioButton, QFrame, 
                            QGridLayout, QGroupBox, QSplitter, QProgressBar,
                            QTabWidget)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
import time

class OptimizedVideoDisplay(QWidget):
    """Widget de video optimizado - solo lo esencial."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_activity = "Desconocida"
        self.confidence = 0.0
        self.fps = 0.0
        self.init_ui()
        
        # Timer para actualizar métricas menos frecuentemente
        self.metrics_timer = QTimer()
        self.metrics_timer.timeout.connect(self.update_metrics_display)
        self.metrics_timer.start(500)  # Actualizar cada 500ms en lugar de cada frame
    
    def init_ui(self):
        """Interfaz minimalista para mejor rendimiento."""
        self.layout = QVBoxLayout()
        
        # Header simple con solo información crítica
        self.create_simple_header()
        
        # Video display principal - SIN overlays complejos
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)  # Resolución más pequeña
        self.video_label.setStyleSheet("""
            background-color: #1a1a1a; 
            border: 2px solid #333; 
            border-radius: 5px;
            color: white;
        """)
        
        font = QFont("Arial", 12)
        self.video_label.setFont(font)
        self.video_label.setText("Iniciando captura...")
        
        self.layout.addWidget(self.video_label)
        
        # Footer simple con métricas básicas
        self.create_simple_footer()
        
        self.setLayout(self.layout)
    
    def create_simple_header(self):
        """Header simplificado con solo información esencial."""
        header_frame = QFrame()
        header_frame.setFixedHeight(60)
        header_frame.setStyleSheet("""
            QFrame {
                background-color: #2c3e50;
                border-radius: 5px;
                margin: 2px;
            }
        """)
        
        header_layout = QHBoxLayout()
        
        # Solo actividad y confianza
        self.activity_display = QLabel("Esperando...")
        self.activity_display.setStyleSheet("color: #ecf0f1; font-size: 16px; font-weight: bold;")
        
        self.confidence_display = QLabel("0%")
        self.confidence_display.setStyleSheet("color: #e74c3c; font-size: 16px; font-weight: bold;")
        
        header_layout.addWidget(QLabel("Actividad:", styleSheet="color: #bdc3c7; font-size: 12px;"))
        header_layout.addWidget(self.activity_display)
        header_layout.addStretch()
        header_layout.addWidget(QLabel("Confianza:", styleSheet="color: #bdc3c7; font-size: 12px;"))
        header_layout.addWidget(self.confidence_display)
        
        header_frame.setLayout(header_layout)
        self.layout.addWidget(header_frame)
    
    def create_simple_footer(self):
        """Footer simple con solo FPS."""
        footer_frame = QFrame()
        footer_frame.setFixedHeight(40)
        footer_frame.setStyleSheet("""
            QFrame {
                background-color: #34495e;
                border-radius: 5px;
                margin: 2px;
            }
        """)
        
        footer_layout = QHBoxLayout()
        
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("color: #ecf0f1; font-size: 14px; font-weight: bold;")
        
        self.status_label = QLabel("Estado: Inactivo")
        self.status_label.setStyleSheet("color: #e74c3c; font-size: 14px;")
        
        footer_layout.addWidget(self.fps_label)
        footer_layout.addStretch()
        footer_layout.addWidget(self.status_label)
        
        footer_frame.setLayout(footer_layout)
        self.layout.addWidget(footer_frame)
    
    def update_frame(self, frame):
        """Actualización de frame ULTRA optimizada - sin overlays."""
        if frame is None:
            return
        
        # NO procesar overlays complejos - solo mostrar el frame con esqueleto básico
        # Convertir frame directamente sin procesamiento adicional
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        
        image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        
        # Escalar una sola vez
        label_size = self.video_label.size()
        scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.video_label.setPixmap(scaled_pixmap)
    
    def update_activity_info(self, activity, confidence):
        """Actualización simplificada de información."""
        self.current_activity = activity
        self.confidence = confidence
        # No actualizar inmediatamente - usar timer
    
    def update_fps(self, fps):
        """Actualizar FPS."""
        self.fps = fps
    
    def update_metrics_display(self):
        """Actualizar métricas menos frecuentemente."""
        self.activity_display.setText(self.current_activity)
        self.confidence_display.setText(f"{self.confidence:.1f}%")
        self.fps_label.setText(f"FPS: {self.fps:.1f}")
        
        # Cambiar color según confianza
        if self.confidence >= 80:
            color = "#27ae60"
            status = "Excelente"
        elif self.confidence >= 60:
            color = "#f39c12"
            status = "Bueno"
        else:
            color = "#e74c3c"
            status = "Bajo"
        
        self.confidence_display.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: bold;")
        self.status_label.setText(f"Estado: {status}")
        self.status_label.setStyleSheet(f"color: {color}; font-size: 14px;")

class SimpleControlPanel(QWidget):
    """Panel de control ultra simplificado."""
    
    start_capture_signal = pyqtSignal(int)
    stop_capture_signal = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Interfaz mínima de controles."""
        layout = QVBoxLayout()
        
        # Selección de cámara
        camera_group = QGroupBox("Cámara")
        camera_layout = QVBoxLayout()
        
        self.camera_combo = QComboBox()
        self.camera_combo.addItems([f"Cámara {i}" for i in range(3)])
        camera_layout.addWidget(self.camera_combo)
        
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        
        # Botones de control
        self.start_button = QPushButton("🎥 Iniciar")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 15px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
        """)
        self.start_button.clicked.connect(self.start_capture)
        layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("⏹️ Detener")
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 15px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.stop_button.clicked.connect(self.stop_capture)
        layout.addWidget(self.stop_button)
        
        # Estado simple
        self.status_display = QLabel("Sistema listo")
        self.status_display.setStyleSheet("""
            QLabel {
                background-color: #ecf0f1;
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                text-align: center;
            }
        """)
        layout.addWidget(self.status_display)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def start_capture(self):
        """Iniciar captura."""
        camera_idx = self.camera_combo.currentIndex()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_display.setText("Capturando...")
        self.status_display.setStyleSheet("""
            QLabel {
                background-color: #d5f4e6;
                border: 1px solid #27ae60;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                text-align: center;
                color: #27ae60;
            }
        """)
        self.start_capture_signal.emit(camera_idx)
    
    def stop_capture(self):
        """Detener captura."""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_display.setText("Detenido")
        self.status_display.setStyleSheet("""
            QLabel {
                background-color: #fadbd8;
                border: 1px solid #e74c3c;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                text-align: center;
                color: #e74c3c;
            }
        """)
        self.stop_capture_signal.emit()

class MinimalInfoPanel(QWidget):
    """Panel de información ultra minimal."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        
        # Timer para actualizar gráficos menos frecuentemente
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_displays)
        self.update_timer.start(1000)  # Actualizar cada segundo
        
        # Datos para actualización
        self.current_activity = "Desconocida"
        self.current_confidence = 0.0
        self.current_angles = {}
        self.current_fps = 0.0
    
    def init_ui(self):
        """Interfaz super simple."""
        layout = QVBoxLayout()
        
        # Métricas actuales básicas
        metrics_group = QGroupBox("Métricas Actuales")
        metrics_layout = QGridLayout()
        
        # Labels para mostrar datos
        metrics_layout.addWidget(QLabel("Actividad:"), 0, 0)
        self.activity_label = QLabel("Desconocida")
        self.activity_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        metrics_layout.addWidget(self.activity_label, 0, 1)
        
        metrics_layout.addWidget(QLabel("Confianza:"), 1, 0)
        self.confidence_label = QLabel("0%")
        self.confidence_label.setStyleSheet("font-weight: bold;")
        metrics_layout.addWidget(self.confidence_label, 1, 1)
        
        metrics_layout.addWidget(QLabel("FPS:"), 2, 0)
        self.fps_display = QLabel("0")
        self.fps_display.setStyleSheet("font-weight: bold; color: #3498db;")
        metrics_layout.addWidget(self.fps_display, 2, 1)
        
        # Solo ángulos de rodillas (lo más importante)
        metrics_layout.addWidget(QLabel("Rodilla Izq:"), 3, 0)
        self.left_knee_label = QLabel("0°")
        metrics_layout.addWidget(self.left_knee_label, 3, 1)
        
        metrics_layout.addWidget(QLabel("Rodilla Der:"), 4, 0)
        self.right_knee_label = QLabel("0°")
        metrics_layout.addWidget(self.right_knee_label, 4, 1)
        
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def update_activity(self, activity, confidence):
        """Actualizar actividad (solo guardar, no mostrar inmediatamente)."""
        self.current_activity = activity
        self.current_confidence = confidence
    
    def update_angles(self, angles):
        """Actualizar ángulos (solo guardar)."""
        self.current_angles = angles
    
    def update_fps(self, fps):
        """Actualizar FPS (solo guardar)."""
        self.current_fps = fps
    
    def update_displays(self):
        """Actualizar displays menos frecuentemente."""
        self.activity_label.setText(self.current_activity)
        self.confidence_label.setText(f"{self.current_confidence:.1f}%")
        self.fps_display.setText(f"{self.current_fps:.1f}")
        
        # Actualizar ángulos si están disponibles
        if 'LEFT_KNEE_ANGLE' in self.current_angles:
            angle = self.current_angles['LEFT_KNEE_ANGLE']
            self.left_knee_label.setText(f"{angle:.1f}°")
            
        if 'RIGHT_KNEE_ANGLE' in self.current_angles:
            angle = self.current_angles['RIGHT_KNEE_ANGLE']
            self.right_knee_label.setText(f"{angle:.1f}°")
        
        # Colores según confianza
        if self.current_confidence >= 80:
            color = "#27ae60"
        elif self.current_confidence >= 60:
            color = "#f39c12"
        else:
            color = "#e74c3c"
        
        self.confidence_label.setStyleSheet(f"font-weight: bold; color: {color};")

# Función para integrar con main.py - COMPONENTES OPTIMIZADOS
def create_optimized_interface():
    """
    Función para crear la interfaz optimizada que reemplaza la anterior.
    """
    return {
        'video_display': OptimizedVideoDisplay,
        'control_panel': SimpleControlPanel,
        'info_panel': MinimalInfoPanel
    }