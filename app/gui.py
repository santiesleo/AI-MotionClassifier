# app/gui.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QSlider
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
import cv2
import numpy as np
import time

class VideoDisplay(QWidget):
    """Widget para mostrar video con landmarks y etiquetas."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Etiqueta para mostrar video
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black;")
        
        # Añadir al layout
        self.layout.addWidget(self.video_label)
    
    def update_frame(self, frame):
        """Actualiza el frame de video mostrado."""
        if frame is None:
            return
            
        # Convertir a formato RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convertir a QImage
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Escalar manteniendo proporción
        pixmap = QPixmap.fromImage(qt_image)
        pixmap = pixmap.scaled(self.video_label.width(), self.video_label.height(),
                              Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Actualizar etiqueta
        self.video_label.setPixmap(pixmap)

class ControlPanel(QWidget):
    """Panel con controles para la aplicación."""
    
    # Definir señales
    start_capture_signal = pyqtSignal(int)  # Índice de cámara
    stop_capture_signal = pyqtSignal()
    record_signal = pyqtSignal(str)  # Actividad a grabar
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Título del panel
        title_label = QLabel("Controles")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.layout.addWidget(title_label)
        
        # Selección de cámara
        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("Cámara:"))
        self.camera_selector = QComboBox()
        self.camera_selector.addItems([f"Cámara {i}" for i in range(3)])
        camera_layout.addWidget(self.camera_selector)
        self.layout.addLayout(camera_layout)
        
        # Botón para iniciar/detener captura
        self.capture_button = QPushButton("Iniciar Captura")
        self.capture_button.clicked.connect(self.toggle_capture)
        self.layout.addWidget(self.capture_button)
        
        # Separador
        self.layout.addWidget(self.create_separator())
        
        # Selección de actividad
        activity_layout = QHBoxLayout()
        activity_layout.addWidget(QLabel("Actividad:"))
        self.activity_selector = QComboBox()
        self.activity_selector.addItems(["Caminar hacia", "Caminar desde", "Girar", "Sentarse", "Levantarse"])
        activity_layout.addWidget(self.activity_selector)
        self.layout.addLayout(activity_layout)
        
        # Botón para grabar
        self.record_button = QPushButton("Grabar Actividad")
        self.record_button.clicked.connect(self.record_activity)
        self.record_button.setEnabled(False)
        self.layout.addWidget(self.record_button)
        
        # Añadir espacio flexible
        self.layout.addStretch()
        
        # Variables de estado
        self.is_capturing = False
    
    def create_separator(self):
        """Crea una línea separadora horizontal."""
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        return separator
    
    def toggle_capture(self):
        """Inicia o detiene la captura de video."""
        if self.is_capturing:
            # Detener captura
            self.stop_capture_signal.emit()
            self.capture_button.setText("Iniciar Captura")
            self.record_button.setEnabled(False)
            self.is_capturing = False
        else:
            # Iniciar captura
            camera_idx = self.camera_selector.currentIndex()
            self.start_capture_signal.emit(camera_idx)
            self.capture_button.setText("Detener Captura")
            self.record_button.setEnabled(True)
            self.is_capturing = True
    
    def record_activity(self):
        """Emite señal para grabar la actividad seleccionada."""
        activity = self.activity_selector.currentText()
        self.record_signal.emit(activity)

class InfoPanel(QWidget):
    """Panel que muestra información de la actividad y ángulos."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Título del panel
        title_label = QLabel("Información")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.layout.addWidget(title_label)
        
        # Actividad detectada
        self.activity_label = QLabel("Actividad: Desconocida")
        self.activity_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.layout.addWidget(self.activity_label)
        
        # Confianza
        self.confidence_label = QLabel("Confianza: 0%")
        self.layout.addWidget(self.confidence_label)
        
        # FPS
        self.fps_label = QLabel("FPS: 0")
        self.layout.addWidget(self.fps_label)
        
        # Separador
        self.layout.addWidget(self.create_separator())
        
        # Título para ángulos
        angles_title = QLabel("Ángulos Articulares")
        angles_title.setFont(QFont("Arial", 12, QFont.Bold))
        self.layout.addWidget(angles_title)
        
        # Ángulos articulares
        self.angle_labels = {}
        for joint in ['Rodilla Izquierda', 'Rodilla Derecha', 'Inclinación Tronco']:
            label = QLabel(f"{joint}: 0°")
            self.layout.addWidget(label)
            self.angle_labels[joint] = label
        
        # Añadir espacio flexible
        self.layout.addStretch()
    
    def create_separator(self):
        """Crea una línea separadora horizontal."""
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        return separator
    
    def update_activity(self, activity, confidence):
        """Actualiza la información de actividad."""
        self.activity_label.setText(f"Actividad: {activity}")
        self.confidence_label.setText(f"Confianza: {confidence:.1f}%")
    
    def update_angles(self, angles):
        """Actualiza los ángulos articulares."""
        for joint, angle in angles.items():
            if joint in self.angle_labels:
                self.angle_labels[joint].setText(f"{joint}: {angle:.1f}°")
    
    def update_fps(self, fps):
        """Actualiza el contador de FPS."""
        self.fps_label.setText(f"FPS: {fps:.1f}")