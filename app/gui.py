# app/gui.py
import cv2
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QComboBox, QSlider, QRadioButton, QButtonGroup, QFrame, 
                            QGridLayout, QGroupBox, QSplitter)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette

class VideoDisplay(QWidget):
    """Widget para mostrar video procesado en tiempo real."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Inicializa la interfaz de usuario."""
        self.layout = QVBoxLayout()
        
        # Etiqueta para mostrar video
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        
        # Establecer un fondo oscuro para la etiqueta de video
        self.video_label.setStyleSheet("background-color: #222; border-radius: 10px;")
        
        # Mensaje inicial
        font = QFont()
        font.setPointSize(14)
        self.video_label.setFont(font)
        self.video_label.setText("No hay video disponible.\nInicie la captura para comenzar.")
        
        self.layout.addWidget(self.video_label)
        
        # Etiqueta para información del frame
        self.info_label = QLabel("Resolución: - | FPS: -")
        self.info_label.setAlignment(Qt.AlignRight)
        self.layout.addWidget(self.info_label)
        
        self.setLayout(self.layout)
    
    def update_frame(self, frame):
        """Actualiza el frame mostrado."""
        if frame is None:
            return
            
        # Convertir frame de OpenCV a QImage
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        
        # Crear QImage desde los datos del frame
        image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Crear QPixmap y ajustar a tamaño de etiqueta
        pixmap = QPixmap.fromImage(image)
        
        # Obtener tamaño de la etiqueta
        label_size = self.video_label.size()
        scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Mostrar frame
        self.video_label.setPixmap(scaled_pixmap)
        
        # Actualizar información
        self.info_label.setText(f"Resolución: {w}x{h}")

class ControlPanel(QWidget):
    """Panel de control para la aplicación."""
    
    # Señales
    start_capture_signal = pyqtSignal(int)  # Para iniciar captura con índice de cámara
    stop_capture_signal = pyqtSignal()      # Para detener captura
    record_signal = pyqtSignal(str)         # Para grabar actividad con nombre
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Inicializa la interfaz de usuario."""
        # Layout principal
        main_layout = QVBoxLayout()
        
        # Grupo para selección de cámara
        camera_group = QGroupBox("Selección de Cámara")
        camera_layout = QHBoxLayout()
        
        self.camera_combo = QComboBox()
        self.camera_combo.addItems([f"Cámara {i}" for i in range(3)])
        camera_layout.addWidget(self.camera_combo)
        
        camera_group.setLayout(camera_layout)
        main_layout.addWidget(camera_group)
        
        # Grupo para controles de captura
        capture_group = QGroupBox("Control de Captura")
        capture_layout = QVBoxLayout()
        
        self.start_button = QPushButton("Iniciar Captura")
        self.start_button.clicked.connect(self.start_capture)
        capture_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Detener Captura")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_capture)
        capture_layout.addWidget(self.stop_button)
        
        capture_group.setLayout(capture_layout)
        main_layout.addWidget(capture_group)
        
        # Grupo para grabación de actividades
        record_group = QGroupBox("Grabación de Actividades")
        record_layout = QVBoxLayout()
        
        self.activity_combo = QComboBox()
        self.activity_combo.addItems([
            "Caminar hacia cámara",
            "Caminar desde cámara",
            "Girar",
            "Sentarse",
            "Levantarse"
        ])
        record_layout.addWidget(self.activity_combo)
        
        self.record_button = QPushButton("Grabar Actividad")
        self.record_button.setEnabled(False)
        self.record_button.clicked.connect(self.record_activity)
        record_layout.addWidget(self.record_button)
        
        record_group.setLayout(record_layout)
        main_layout.addWidget(record_group)
        
        # Agregar espacio flexible
        main_layout.addStretch()
        
        # Información del proyecto
        info_label = QLabel(
            "Sistema de Anotación de Video\n"
            "para Análisis de Actividades Físicas\n\n"
            "Proyecto de Inteligencia Artificial 1\n"
            "Universidad ICESI - 2025-1"
        )
        info_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(info_label)
        
        self.setLayout(main_layout)
    
    def start_capture(self):
        """Inicia la captura de video."""
        camera_idx = self.camera_combo.currentIndex()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.record_button.setEnabled(True)
        self.start_capture_signal.emit(camera_idx)
    
    def stop_capture(self):
        """Detiene la captura de video."""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self.stop_capture_signal.emit()
    
    def record_activity(self):
        """Graba la actividad seleccionada."""
        activity = self.activity_combo.currentText()
        self.record_signal.emit(activity)

class InfoPanel(QWidget):
    """Panel para mostrar información de análisis."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.angle_history = {}  # Historial de ángulos para gráfico
        self.init_ui()
    
    def init_ui(self):
        """Inicializa la interfaz de usuario."""
        # Layout principal
        main_layout = QVBoxLayout()
        
        # Grupo para información de actividad
        activity_group = QGroupBox("Actividad Detectada")
        activity_layout = QVBoxLayout()
        
        self.activity_label = QLabel("Desconocida")
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        self.activity_label.setFont(font)
        self.activity_label.setAlignment(Qt.AlignCenter)
        activity_layout.addWidget(self.activity_label)
        
        self.confidence_label = QLabel("Confianza: 0%")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        activity_layout.addWidget(self.confidence_label)
        
        activity_group.setLayout(activity_layout)
        main_layout.addWidget(activity_group)
        
        # Grupo para ángulos articulares
        angles_group = QGroupBox("Ángulos Articulares")
        angles_layout = QGridLayout()
        
        # Etiquetas para ángulos
        self.angle_labels = {}
        joint_names = ["Rodilla Izquierda", "Rodilla Derecha", "Inclinación Tronco"]
        joint_keys = ["LEFT_KNEE_ANGLE", "RIGHT_KNEE_ANGLE", "TRUNK_TILT"]
        
        for i, (name, key) in enumerate(zip(joint_names, joint_keys)):
            label_name = QLabel(name + ":")
            label_value = QLabel("0°")
            label_value.setMinimumWidth(60)
            angles_layout.addWidget(label_name, i, 0)
            angles_layout.addWidget(label_value, i, 1)
            self.angle_labels[key] = label_value
        
        angles_group.setLayout(angles_layout)
        main_layout.addWidget(angles_group)
        
        # Gráfico de ángulos
        graph_group = QGroupBox("Gráfico de Ángulos")
        graph_layout = QVBoxLayout()
        
        # Crear figura y canvas de matplotlib
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title('Ángulos articulares')
        self.ax.set_xlabel('Frames')
        self.ax.set_ylabel('Ángulo (grados)')
        self.ax.grid(True)
        
        graph_layout.addWidget(self.canvas)
        
        graph_group.setLayout(graph_layout)
        main_layout.addWidget(graph_group)
        
        # Información adicional
        self.fps_label = QLabel("FPS: 0")
        main_layout.addWidget(self.fps_label)
        
        self.setLayout(main_layout)
    
    def update_activity(self, activity, confidence):
        """Actualiza la actividad detectada y su confianza."""
        self.activity_label.setText(activity)
        self.confidence_label.setText(f"Confianza: {confidence:.1f}%")
        
        # Cambiar color basado en confianza
        if confidence >= 80:
            color = "green"
        elif confidence >= 50:
            color = "orange"
        else:
            color = "red"
            
        self.activity_label.setStyleSheet(f"color: {color}")
    
    def update_angles(self, angles):
        """Actualiza los ángulos articulares mostrados."""
        for joint, value in angles.items():
            if joint in self.angle_labels:
                self.angle_labels[joint].setText(f"{value:.1f}°")
        
        # Actualizar gráfico
        self.update_angle_plot(angles)
    
    def update_fps(self, fps):
        """Actualiza la información de FPS."""
        self.fps_label.setText(f"FPS: {fps:.1f}")
    
    def update_angle_plot(self, angles):
        """Actualiza el gráfico de ángulos en tiempo real."""
        # Actualizar datos de historial
        for joint, angle in angles.items():
            if joint not in self.angle_history:
                self.angle_history[joint] = []
                
            # Limitar longitud del historial
            max_history = 100
            if len(self.angle_history[joint]) >= max_history:
                self.angle_history[joint].pop(0)
                
            self.angle_history[joint].append(angle)
        
        # Limpiar gráfico
        self.ax.clear()
        
        # Agregar líneas para cada articulación
        colors = ['r', 'g', 'b', 'c', 'm']
        for i, (joint, history) in enumerate(self.angle_history.items()):
            if len(history) > 1:
                color = colors[i % len(colors)]
                self.ax.plot(history, color=color, label=joint)
        
        # Configurar gráfico
        self.ax.set_title('Ángulos articulares')
        self.ax.set_xlabel('Frames')
        self.ax.set_ylabel('Ángulo (grados)')
        self.ax.set_ylim(0, 180)
        self.ax.legend(loc='upper right')
        self.ax.grid(True)
        
        # Actualizar canvas
        self.canvas.draw()