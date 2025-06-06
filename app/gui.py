# app/improved_gui.py
import cv2
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, 
                            QComboBox, QSlider, QRadioButton, QButtonGroup, QFrame, 
                            QGridLayout, QGroupBox, QSplitter, QProgressBar, QTextEdit,
                            QScrollArea, QTabWidget)
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette, QPainter, QPen
import time

class ModernVideoDisplay(QWidget):
    """Widget moderno para mostrar video procesado con overlays informativos."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_activity = "Desconocida"
        self.confidence = 0.0
        self.angles = {}
        self.init_ui()
    
    def init_ui(self):
        """Inicializa la interfaz de usuario moderna."""
        self.layout = QVBoxLayout()
        
        # Header con información de actividad
        self.create_activity_header()
        
        # Video display principal
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("""
            background-color: #1a1a1a; 
            border: 2px solid #333; 
            border-radius: 10px;
            color: white;
        """)
        
        font = QFont("Arial", 14)
        self.video_label.setFont(font)
        self.video_label.setText("Iniciando sistema de captura...\nPor favor espere")
        
        self.layout.addWidget(self.video_label)
        
        # Footer con métricas
        self.create_metrics_footer()
        
        self.setLayout(self.layout)
    
    def create_activity_header(self):
        """Crea el header con información de actividad detectada."""
        header_frame = QFrame()
        header_frame.setFixedHeight(80)
        header_frame.setStyleSheet("""
            QFrame {
                background-color: #2c3e50;
                border-radius: 10px;
                margin: 5px;
            }
        """)
        
        header_layout = QHBoxLayout()
        
        # Actividad detectada
        activity_section = QVBoxLayout()
        activity_title = QLabel("ACTIVIDAD DETECTADA")
        activity_title.setStyleSheet("color: #bdc3c7; font-size: 12px; font-weight: bold;")
        self.activity_display = QLabel("Esperando...")
        self.activity_display.setStyleSheet("color: #ecf0f1; font-size: 18px; font-weight: bold;")
        
        activity_section.addWidget(activity_title)
        activity_section.addWidget(self.activity_display)
        
        # Confianza
        confidence_section = QVBoxLayout()
        confidence_title = QLabel("CONFIANZA")
        confidence_title.setStyleSheet("color: #bdc3c7; font-size: 12px; font-weight: bold;")
        self.confidence_display = QLabel("0%")
        self.confidence_display.setStyleSheet("color: #e74c3c; font-size: 18px; font-weight: bold;")
        
        confidence_section.addWidget(confidence_title)
        confidence_section.addWidget(self.confidence_display)
        
        # Barra de confianza
        confidence_bar_section = QVBoxLayout()
        confidence_bar_title = QLabel("NIVEL DE CERTEZA")
        confidence_bar_title.setStyleSheet("color: #bdc3c7; font-size: 12px; font-weight: bold;")
        
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #34495e;
                border-radius: 5px;
                text-align: center;
                color: white;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #e74c3c;
                border-radius: 3px;
            }
        """)
        
        confidence_bar_section.addWidget(confidence_bar_title)
        confidence_bar_section.addWidget(self.confidence_bar)
        
        header_layout.addLayout(activity_section)
        header_layout.addWidget(self.create_separator())
        header_layout.addLayout(confidence_section)
        header_layout.addWidget(self.create_separator())
        header_layout.addLayout(confidence_bar_section)
        
        header_frame.setLayout(header_layout)
        self.layout.addWidget(header_frame)
    
    def create_separator(self):
        """Crea un separador vertical."""
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("color: #34495e;")
        return separator
    
    def create_metrics_footer(self):
        """Crea el footer con métricas en tiempo real."""
        footer_frame = QFrame()
        footer_frame.setFixedHeight(60)
        footer_frame.setStyleSheet("""
            QFrame {
                background-color: #34495e;
                border-radius: 10px;
                margin: 5px;
            }
        """)
        
        footer_layout = QHBoxLayout()
        
        # FPS
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("color: #ecf0f1; font-size: 14px; font-weight: bold;")
        
        # Resolución
        self.resolution_label = QLabel("Resolución: -")
        self.resolution_label.setStyleSheet("color: #ecf0f1; font-size: 14px;")
        
        # Estado
        self.status_label = QLabel("Estado: Inactivo")
        self.status_label.setStyleSheet("color: #e74c3c; font-size: 14px; font-weight: bold;")
        
        footer_layout.addWidget(self.fps_label)
        footer_layout.addWidget(self.create_separator())
        footer_layout.addWidget(self.resolution_label)
        footer_layout.addWidget(self.create_separator())
        footer_layout.addWidget(self.status_label)
        footer_layout.addStretch()
        
        footer_frame.setLayout(footer_layout)
        self.layout.addWidget(footer_frame)
    
    def update_frame(self, frame):
        """Actualiza el frame mostrado con overlays informativos."""
        if frame is None:
            return
        
        # Crear copia para overlay
        frame_with_overlay = frame.copy()
        
        # Agregar overlay de información
        self.draw_activity_overlay(frame_with_overlay)
        self.draw_angles_overlay(frame_with_overlay)
        
        # Convertir frame de OpenCV a QImage
        frame_rgb = cv2.cvtColor(frame_with_overlay, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        
        image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        
        # Escalar manteniendo proporción
        label_size = self.video_label.size()
        scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.video_label.setPixmap(scaled_pixmap)
        self.resolution_label.setText(f"Resolución: {w}x{h}")
    
    def draw_activity_overlay(self, frame):
        """Dibuja overlay con información de actividad."""
        h, w = frame.shape[:2]
        
        # Fondo semi-transparente para el overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 100), (44, 62, 80), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Texto de actividad
        cv2.putText(frame, f"Actividad: {self.current_activity}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Barra de confianza visual
        conf_width = int(300 * (self.confidence / 100))
        color = self.get_confidence_color(self.confidence)
        cv2.rectangle(frame, (20, 55), (20 + conf_width, 75), color, -1)
        cv2.rectangle(frame, (20, 55), (320, 75), (255, 255, 255), 2)
        
        cv2.putText(frame, f"Confianza: {self.confidence:.1f}%", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def draw_angles_overlay(self, frame):
        """Dibuja overlay con información de ángulos."""
        h, w = frame.shape[:2]
        
        if not self.angles:
            return
        
        # Fondo para ángulos
        overlay = frame.copy()
        cv2.rectangle(overlay, (w-350, 10), (w-10, 200), (52, 73, 94), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Título
        cv2.putText(frame, "ANGULOS ARTICULARES", (w-340, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset = 60
        for joint, angle in self.angles.items():
            if isinstance(angle, (int, float)):
                # Color basado en normalidad del ángulo
                color = self.get_angle_color(joint, angle)
                
                # Texto del ángulo
                joint_name = joint.replace('_', ' ').title()
                cv2.putText(frame, f"{joint_name}:", (w-340, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"{angle:.1f}°", (w-150, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Indicador visual del ángulo
                indicator_start = w-120
                indicator_width = int(80 * min(angle / 180, 1.0))
                cv2.rectangle(frame, (indicator_start, y_offset-10), 
                            (indicator_start + indicator_width, y_offset-5), color, -1)
                
                y_offset += 25
    
    def get_confidence_color(self, confidence):
        """Retorna color basado en nivel de confianza."""
        if confidence >= 80:
            return (0, 255, 0)  # Verde
        elif confidence >= 60:
            return (0, 255, 255)  # Amarillo
        elif confidence >= 40:
            return (0, 165, 255)  # Naranja
        else:
            return (0, 0, 255)  # Rojo
    
    def get_angle_color(self, joint, angle):
        """Retorna color basado en normalidad del ángulo."""
        if 'KNEE' in joint:
            if 160 <= angle <= 180:
                return (0, 255, 0)  # Verde - normal
            elif 90 <= angle < 160:
                return (0, 255, 255)  # Amarillo - flexionado
            else:
                return (0, 0, 255)  # Rojo - anormal
        else:
            # Para otros ángulos, usar escala general
            if 80 <= angle <= 100:
                return (0, 255, 0)
            elif 60 <= angle <= 120:
                return (0, 255, 255)
            else:
                return (0, 0, 255)
    
    def update_activity_info(self, activity, confidence):
        """Actualiza la información de actividad."""
        self.current_activity = activity
        self.confidence = confidence
        
        # Actualizar displays
        self.activity_display.setText(activity)
        self.confidence_display.setText(f"{confidence:.1f}%")
        self.confidence_bar.setValue(int(confidence))
        
        # Cambiar colores según confianza
        if confidence >= 80:
            color = "#27ae60"  # Verde
            self.status_label.setText("Estado: Excelente")
            self.status_label.setStyleSheet("color: #27ae60; font-size: 14px; font-weight: bold;")
        elif confidence >= 60:
            color = "#f39c12"  # Naranja
            self.status_label.setText("Estado: Bueno")
            self.status_label.setStyleSheet("color: #f39c12; font-size: 14px; font-weight: bold;")
        else:
            color = "#e74c3c"  # Rojo
            self.status_label.setText("Estado: Bajo")
            self.status_label.setStyleSheet("color: #e74c3c; font-size: 14px; font-weight: bold;")
        
        self.confidence_display.setStyleSheet(f"color: {color}; font-size: 18px; font-weight: bold;")
        
        # Actualizar barra de progreso
        self.confidence_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid #34495e;
                border-radius: 5px;
                text-align: center;
                color: white;
                font-weight: bold;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 3px;
            }}
        """)
    
    def update_angles(self, angles):
        """Actualiza los ángulos articulares."""
        self.angles = angles
    
    def update_fps(self, fps):
        """Actualiza el contador de FPS."""
        self.fps_label.setText(f"FPS: {fps:.1f}")

class AdvancedControlPanel(QWidget):
    """Panel de control avanzado con más opciones."""
    
    # Señales (se eliminó record_signal)
    start_capture_signal = pyqtSignal(int)
    stop_capture_signal = pyqtSignal()
    settings_changed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
    
    def init_ui(self):
        """Inicializa la interfaz de usuario avanzada."""
        layout = QVBoxLayout()
        
        # Crear tabs para organizar controles
        tab_widget = QTabWidget()
        
        # Tab 1: Captura
        capture_tab = self.create_capture_tab()
        tab_widget.addTab(capture_tab, "Captura")
        
        # Tab 2: Configuración
        config_tab = self.create_config_tab()
        tab_widget.addTab(config_tab, "Configuración")
        
        # Tab 3: Información
        info_tab = self.create_info_tab()
        tab_widget.addTab(info_tab, "Información")
        
        layout.addWidget(tab_widget)
        self.setLayout(layout)
    
    def create_capture_tab(self):
        """Crea el tab de captura."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Selección de cámara
        camera_group = QGroupBox("Selección de Cámara")
        camera_layout = QVBoxLayout()
        
        self.camera_combo = QComboBox()
        self.camera_combo.addItems([f"Cámara {i}" for i in range(3)])
        camera_layout.addWidget(self.camera_combo)
        
        # Botón de detección automática
        detect_button = QPushButton("Detectar Cámaras")
        detect_button.clicked.connect(self.detect_cameras)
        camera_layout.addWidget(detect_button)
        
        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)
        
        # Controles de captura
        capture_group = QGroupBox("Control de Captura")
        capture_layout = QVBoxLayout()
        
        self.start_button = QPushButton("🎥 Iniciar Captura")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2ecc71;
            }
        """)
        self.start_button.clicked.connect(self.start_capture)
        capture_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("⏹️ Detener Captura")
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.stop_button.clicked.connect(self.stop_capture)
        capture_layout.addWidget(self.stop_button)
        
        capture_group.setLayout(capture_layout)
        layout.addWidget(capture_group)
        
        # Estado del sistema
        status_group = QGroupBox("Estado del Sistema")
        status_layout = QVBoxLayout()
        
        self.status_display = QLabel("Sistema listo")
        self.status_display.setStyleSheet("""
            QLabel {
                background-color: #ecf0f1;
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                color: #2c3e50;
            }
        """)
        status_layout.addWidget(self.status_display)
        
        # Información de la sesión
        self.session_info = QLabel("Sesión no iniciada")
        self.session_info.setStyleSheet("""
            QLabel {
                color: #7f8c8d;
                font-size: 12px;
                padding: 5px;
            }
        """)
        status_layout.addWidget(self.session_info)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_config_tab(self):
        """Crea el tab de configuración."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Configuración de detección
        detection_group = QGroupBox("Configuración de Detección")
        detection_layout = QGridLayout()
        
        # Sensibilidad
        detection_layout.addWidget(QLabel("Sensibilidad:"), 0, 0)
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(1, 10)
        self.sensitivity_slider.setValue(5)
        self.sensitivity_label = QLabel("5")
        detection_layout.addWidget(self.sensitivity_slider, 0, 1)
        detection_layout.addWidget(self.sensitivity_label, 0, 2)
        
        self.sensitivity_slider.valueChanged.connect(
            lambda v: self.sensitivity_label.setText(str(v))
        )
        
        # Umbral de confianza
        detection_layout.addWidget(QLabel("Umbral de confianza:"), 1, 0)
        self.confidence_threshold = QSlider(Qt.Horizontal)
        self.confidence_threshold.setRange(10, 90)
        self.confidence_threshold.setValue(50)
        self.threshold_label = QLabel("50%")
        detection_layout.addWidget(self.confidence_threshold, 1, 1)
        detection_layout.addWidget(self.threshold_label, 1, 2)
        
        self.confidence_threshold.valueChanged.connect(
            lambda v: self.threshold_label.setText(f"{v}%")
        )
        
        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)
        
        # Configuración de visualización
        visual_group = QGroupBox("Configuración Visual")
        visual_layout = QVBoxLayout()
        
        # Checkboxes para opciones
        self.show_landmarks = QRadioButton("Mostrar landmarks")
        self.show_landmarks.setChecked(True)
        self.show_angles = QRadioButton("Mostrar ángulos")
        self.show_angles.setChecked(True)
        self.show_overlay = QRadioButton("Mostrar overlay de información")
        self.show_overlay.setChecked(True)
        
        visual_layout.addWidget(self.show_landmarks)
        visual_layout.addWidget(self.show_angles)
        visual_layout.addWidget(self.show_overlay)
        
        visual_group.setLayout(visual_layout)
        layout.addWidget(visual_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_info_tab(self):
        """Crea el tab de información."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Información del proyecto
        info_text = QTextEdit()
        info_text.setReadOnly(True)
        info_text.setHtml("""
        <h3>Sistema de Análisis de Actividades Físicas</h3>
        <p><b>Versión:</b> 2.0</p>
        <p><b>Autores:</b></p>
        <ul>
            <li>Alejandro Amu García</li>
            <li>Santiago Escobar Leon</li>
            <li>David Donneys</li>
        </ul>
        
        <h4>Características principales:</h4>
        <ul>
            <li>🎥 Detección en tiempo real de actividades físicas</li>
            <li>📊 Análisis de ángulos articulares</li>
            <li>🧠 Clasificación automática con IA</li>
            <li>📈 Visualización avanzada de métricas</li>
            <li>💾 Grabación y exportación de datos</li>
        </ul>
        
        <h4>Actividades detectadas:</h4>
        <ul>
            <li>Caminar hacia la cámara</li>
            <li>Caminar alejándose de la cámara</li>
            <li>Girar</li>
            <li>Sentarse</li>
            <li>Levantarse</li>
        </ul>
        
        <p><b>Universidad ICESI - 2025</b></p>
        <p><i>Proyecto de Inteligencia Artificial</i></p>
        """)
        
        layout.addWidget(info_text)
        widget.setLayout(layout)
        return widget
    
    def detect_cameras(self):
        """Detecta cámaras disponibles."""
        self.camera_combo.clear()
        cameras_found = 0
        
        for i in range(5):  # Probar hasta 5 cámaras
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.camera_combo.addItem(f"Cámara {i}")
                    cameras_found += 1
                cap.release()
        
        if cameras_found == 0:
            self.camera_combo.addItem("No se encontraron cámaras")
    
    def start_capture(self):
        """Inicia la captura de video."""
        camera_idx = self.camera_combo.currentIndex()
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.status_display.setText("Captura iniciada")
        self.status_display.setStyleSheet("""
            QLabel {
                background-color: #d5f4e6;
                border: 2px solid #27ae60;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                color: #27ae60;
            }
        """)
        self.start_capture_signal.emit(camera_idx)
    
    def stop_capture(self):
        """Detiene la captura de video."""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_display.setText("Captura detenida")
        self.status_display.setStyleSheet("""
            QLabel {
                background-color: #fadbd8;
                border: 2px solid #e74c3c;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                font-weight: bold;
                color: #e74c3c;
            }
        """)
        self.stop_capture_signal.emit()
    
    def update_session_info(self, info_text):
        """Actualiza la información de sesión."""
        self.session_info.setText(info_text)

class AdvancedInfoPanel(QWidget):
    """Panel de información avanzado con gráficos en tiempo real."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.angle_history = {}
        self.activity_history = []
        self.confidence_history = []
        self.init_ui()
    
    def init_ui(self):
        """Inicializa la interfaz de usuario."""
        layout = QVBoxLayout()
        
        # Crear tabs para diferentes visualizaciones
        tab_widget = QTabWidget()
        
        # Tab 1: Métricas actuales
        metrics_tab = self.create_metrics_tab()
        tab_widget.addTab(metrics_tab, "Métricas")
        
        # Tab 2: Gráficos en tiempo real
        graphs_tab = self.create_graphs_tab()
        tab_widget.addTab(graphs_tab, "Gráficos")
        
        # Tab 3: Estadísticas
        stats_tab = self.create_stats_tab()
        tab_widget.addTab(stats_tab, "Estadísticas")
        
        layout.addWidget(tab_widget)
        self.setLayout(layout)
    
    def create_metrics_tab(self):
        """Crea el tab de métricas."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Actividad actual
        activity_group = QGroupBox("Actividad Actual")
        activity_layout = QGridLayout()
        
        self.activity_label = QLabel("Desconocida")
        self.activity_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        activity_layout.addWidget(QLabel("Actividad:"), 0, 0)
        activity_layout.addWidget(self.activity_label, 0, 1)
        
        self.confidence_label = QLabel("0%")
        self.confidence_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #e74c3c;")
        activity_layout.addWidget(QLabel("Confianza:"), 1, 0)
        activity_layout.addWidget(self.confidence_label, 1, 1)
        
        activity_group.setLayout(activity_layout)
        layout.addWidget(activity_group)
        
        # Ángulos articulares con visualización mejorada
        angles_group = QGroupBox("Ángulos Articulares")
        angles_layout = QGridLayout()
        
        self.angle_displays = {}
        self.angle_bars = {}
        
        joints = ["Rodilla Izq.", "Rodilla Der.", "Inclinación"]
        joint_keys = ["LEFT_KNEE_ANGLE", "RIGHT_KNEE_ANGLE", "TRUNK_TILT"]
        
        for i, (name, key) in enumerate(zip(joints, joint_keys)):
            # Etiqueta
            angles_layout.addWidget(QLabel(name + ":"), i, 0)
            
            # Valor
            value_label = QLabel("0°")
            value_label.setMinimumWidth(60)
            value_label.setStyleSheet("font-weight: bold;")
            angles_layout.addWidget(value_label, i, 1)
            self.angle_displays[key] = value_label
            
            # Barra de progreso para visualizar el ángulo
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 180)
            progress_bar.setValue(0)
            progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #bdc3c7;
                    border-radius: 3px;
                    text-align: center;
                    font-size: 10px;
                }
                QProgressBar::chunk {
                    background-color: #3498db;
                    border-radius: 2px;
                }
            """)
            angles_layout.addWidget(progress_bar, i, 2)
            self.angle_bars[key] = progress_bar
        
        angles_group.setLayout(angles_layout)
        layout.addWidget(angles_group)
        
        # Métricas de rendimiento
        performance_group = QGroupBox("Rendimiento del Sistema")
        performance_layout = QGridLayout()
        
        self.fps_label = QLabel("0 FPS")
        self.processing_time_label = QLabel("0 ms")
        self.frames_processed_label = QLabel("0")
        
        performance_layout.addWidget(QLabel("FPS:"), 0, 0)
        performance_layout.addWidget(self.fps_label, 0, 1)
        performance_layout.addWidget(QLabel("Tiempo proc.:"), 1, 0)
        performance_layout.addWidget(self.processing_time_label, 1, 1)
        performance_layout.addWidget(QLabel("Frames proc.:"), 2, 0)
        performance_layout.addWidget(self.frames_processed_label, 2, 1)
        
        performance_group.setLayout(performance_layout)
        layout.addWidget(performance_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_graphs_tab(self):
        """Crea el tab de gráficos en tiempo real."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Gráfico de ángulos en tiempo real
        self.angles_figure = Figure(figsize=(8, 4), dpi=100)
        self.angles_canvas = FigureCanvas(self.angles_figure)
        self.angles_ax = self.angles_figure.add_subplot(111)
        self.angles_ax.set_title('Ángulos Articulares en Tiempo Real')
        self.angles_ax.set_xlabel('Tiempo (frames)')
        self.angles_ax.set_ylabel('Ángulo (grados)')
        self.angles_ax.grid(True, alpha=0.3)
        self.angles_ax.set_ylim(0, 180)
        
        layout.addWidget(self.angles_canvas)
        
        # Gráfico de confianza
        self.confidence_figure = Figure(figsize=(8, 3), dpi=100)
        self.confidence_canvas = FigureCanvas(self.confidence_figure)
        self.confidence_ax = self.confidence_figure.add_subplot(111)
        self.confidence_ax.set_title('Confianza de Detección')
        self.confidence_ax.set_xlabel('Tiempo (frames)')
        self.confidence_ax.set_ylabel('Confianza (%)')
        self.confidence_ax.grid(True, alpha=0.3)
        self.confidence_ax.set_ylim(0, 100)
        
        layout.addWidget(self.confidence_canvas)
        
        widget.setLayout(layout)
        return widget
    
    def create_stats_tab(self):
        """Crea el tab de estadísticas."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Estadísticas de sesión
        session_group = QGroupBox("Estadísticas de Sesión")
        session_layout = QGridLayout()
        
        self.session_duration_label = QLabel("00:00:00")
        self.total_activities_label = QLabel("0")
        self.avg_confidence_label = QLabel("0%")
        self.most_frequent_activity_label = QLabel("N/A")
        
        session_layout.addWidget(QLabel("Duración:"), 0, 0)
        session_layout.addWidget(self.session_duration_label, 0, 1)
        session_layout.addWidget(QLabel("Actividades detectadas:"), 1, 0)
        session_layout.addWidget(self.total_activities_label, 1, 1)
        session_layout.addWidget(QLabel("Confianza promedio:"), 2, 0)
        session_layout.addWidget(self.avg_confidence_label, 2, 1)
        session_layout.addWidget(QLabel("Actividad más frecuente:"), 3, 0)
        session_layout.addWidget(self.most_frequent_activity_label, 3, 1)
        
        session_group.setLayout(session_layout)
        layout.addWidget(session_group)
        
        # Log de actividades
        log_group = QGroupBox("Registro de Actividades")
        log_layout = QVBoxLayout()
        
        self.activity_log = QTextEdit()
        self.activity_log.setMaximumHeight(200)
        self.activity_log.setReadOnly(True)
        self.activity_log.setStyleSheet("font-family: monospace; font-size: 10px;")
        
        log_layout.addWidget(self.activity_log)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        # Botones de control
        buttons_layout = QHBoxLayout()
        
        clear_log_button = QPushButton("Limpiar Log")
        clear_log_button.clicked.connect(self.clear_activity_log)
        
        export_button = QPushButton("Exportar Datos")
        export_button.clicked.connect(self.export_session_data)
        
        buttons_layout.addWidget(clear_log_button)
        buttons_layout.addWidget(export_button)
        
        layout.addLayout(buttons_layout)
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
    
    def update_activity(self, activity, confidence):
        """Actualiza la información de actividad."""
        self.activity_label.setText(activity)
        self.confidence_label.setText(f"{confidence:.1f}%")
        
        # Cambiar color basado en confianza
        if confidence >= 80:
            color = "#27ae60"
        elif confidence >= 60:
            color = "#f39c12"
        else:
            color = "#e74c3c"
        
        self.confidence_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {color};")
        
        # Actualizar historial
        self.activity_history.append(activity)
        self.confidence_history.append(confidence)
        
        # Limitar historial
        max_history = 1000
        if len(self.activity_history) > max_history:
            self.activity_history.pop(0)
            self.confidence_history.pop(0)
        
        # Actualizar gráfico de confianza
        self.update_confidence_graph()
        
        # Agregar al log
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {activity} (Confianza: {confidence:.1f}%)\n"
        self.activity_log.append(log_entry.strip())
        
        # Scroll automático al final
        scrollbar = self.activity_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def update_angles(self, angles):
        """Actualiza los ángulos articulares."""
        for joint, angle in angles.items():
            if joint in self.angle_displays and isinstance(angle, (int, float)):
                # Actualizar display
                self.angle_displays[joint].setText(f"{angle:.1f}°")
                
                # Actualizar barra de progreso
                if joint in self.angle_bars:
                    self.angle_bars[joint].setValue(int(angle))
                    
                    # Cambiar color basado en normalidad
                    if joint in ['LEFT_KNEE_ANGLE', 'RIGHT_KNEE_ANGLE']:
                        if 160 <= angle <= 180:
                            color = "#27ae60"  # Verde
                        elif 90 <= angle < 160:
                            color = "#f39c12"  # Amarillo
                        else:
                            color = "#e74c3c"  # Rojo
                    else:
                        color = "#3498db"  # Azul por defecto
                    
                    self.angle_bars[joint].setStyleSheet(f"""
                        QProgressBar {{
                            border: 1px solid #bdc3c7;
                            border-radius: 3px;
                            text-align: center;
                            font-size: 10px;
                        }}
                        QProgressBar::chunk {{
                            background-color: {color};
                            border-radius: 2px;
                        }}
                    """)
                
                # Actualizar historial para gráficos
                if joint not in self.angle_history:
                    self.angle_history[joint] = []
                
                self.angle_history[joint].append(angle)
                
                # Limitar historial
                max_history = 300
                if len(self.angle_history[joint]) > max_history:
                    self.angle_history[joint].pop(0)
        
        # Actualizar gráfico de ángulos
        self.update_angles_graph()
    
    def update_angles_graph(self):
        """Actualiza el gráfico de ángulos en tiempo real."""
        self.angles_ax.clear()
        
        colors = ['#e74c3c', '#27ae60', '#3498db', '#f39c12', '#9b59b6']
        
        for i, (joint, history) in enumerate(self.angle_history.items()):
            if len(history) > 1:
                color = colors[i % len(colors)]
                x_data = list(range(len(history)))
                self.angles_ax.plot(x_data, history, color=color, label=joint.replace('_', ' '), linewidth=2)
        
        self.angles_ax.set_title('Ángulos Articulares en Tiempo Real')
        self.angles_ax.set_xlabel('Tiempo (frames)')
        self.angles_ax.set_ylabel('Ángulo (grados)')
        self.angles_ax.set_ylim(0, 180)
        self.angles_ax.grid(True, alpha=0.3)
        self.angles_ax.legend(loc='upper right', fontsize=8)
        
        self.angles_canvas.draw()
    
    def update_confidence_graph(self):
        """Actualiza el gráfico de confianza."""
        if len(self.confidence_history) > 1:
            self.confidence_ax.clear()
            
            x_data = list(range(len(self.confidence_history)))
            self.confidence_ax.plot(x_data, self.confidence_history, color='#3498db', linewidth=2)
            self.confidence_ax.fill_between(x_data, self.confidence_history, alpha=0.3, color='#3498db')
            
            # Líneas de referencia
            self.confidence_ax.axhline(y=80, color='#27ae60', linestyle='--', alpha=0.7, label='Excelente (80%)')
            self.confidence_ax.axhline(y=60, color='#f39c12', linestyle='--', alpha=0.7, label='Bueno (60%)')
            self.confidence_ax.axhline(y=40, color='#e74c3c', linestyle='--', alpha=0.7, label='Bajo (40%)')
            
            self.confidence_ax.set_title('Confianza de Detección')
            self.confidence_ax.set_xlabel('Tiempo (frames)')
            self.confidence_ax.set_ylabel('Confianza (%)')
            self.confidence_ax.set_ylim(0, 100)
            self.confidence_ax.grid(True, alpha=0.3)
            self.confidence_ax.legend(loc='upper right', fontsize=8)
            
            self.confidence_canvas.draw()
    
    def update_fps(self, fps):
        """Actualiza la información de FPS."""
        self.fps_label.setText(f"{fps:.1f} FPS")
    
    def update_performance_metrics(self, processing_time, frames_count):
        """Actualiza métricas de rendimiento."""
        self.processing_time_label.setText(f"{processing_time:.1f} ms")
        self.frames_processed_label.setText(str(frames_count))
    
    def update_session_stats(self, duration, total_activities, avg_confidence, most_frequent):
        """Actualiza estadísticas de sesión."""
        self.session_duration_label.setText(duration)
        self.total_activities_label.setText(str(total_activities))
        self.avg_confidence_label.setText(f"{avg_confidence:.1f}%")
        self.most_frequent_activity_label.setText(most_frequent)
    
    def clear_activity_log(self):
        """Limpia el log de actividades."""
        self.activity_log.clear()
        self.activity_history.clear()
        self.confidence_history.clear()
        self.angle_history.clear()
    
    def export_session_data(self):
        """Exporta los datos de la sesión."""
        # Esta función se puede implementar para exportar datos a CSV o JSON
        print("Exportando datos de sesión...")
        # Implementar lógica de exportación aquí

# Función para integrar con el main.py existente
def create_improved_interface():
    """
    Función para crear la interfaz mejorada que puede ser integrada
    en el archivo main.py existente.
    """
    return {
        'video_display': ModernVideoDisplay,
        'control_panel': AdvancedControlPanel,
        'info_panel': AdvancedInfoPanel
    }