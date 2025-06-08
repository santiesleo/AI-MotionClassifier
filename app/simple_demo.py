# app/simple_demo.py
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time
import os
from pathlib import Path
import random

class SimplePoseDetector:
    """Detector de poses simplificado que usa solo OpenCV."""
    
    def __init__(self):
        # Cargar modelo de detección de cuerpo (HOG)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    def detect(self, frame):
        """Detecta personas en el frame."""
        # Convertir a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar personas
        boxes, weights = self.hog.detectMultiScale(
            gray, 
            winStride=(8, 8),
            padding=(4, 4),
            scale=1.05
        )
        
        # Dibujar rectángulos
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Puntos simulados para articulaciones
            head_x, head_y = x + w//2, y + h//8
            l_shoulder_x, l_shoulder_y = x + w//4, y + h//4
            r_shoulder_x, r_shoulder_y = x + 3*w//4, y + h//4
            l_hip_x, l_hip_y = x + w//4, y + h//2
            r_hip_x, r_hip_y = x + 3*w//4, y + h//2
            l_knee_x, l_knee_y = x + w//4, y + 3*h//4
            r_knee_x, r_knee_y = x + 3*w//4, y + 3*h//4
            
            # Dibujar articulaciones simuladas
            cv2.circle(frame, (head_x, head_y), 5, (0, 0, 255), -1)
            cv2.circle(frame, (l_shoulder_x, l_shoulder_y), 5, (0, 0, 255), -1)
            cv2.circle(frame, (r_shoulder_x, r_shoulder_y), 5, (0, 0, 255), -1)
            cv2.circle(frame, (l_hip_x, l_hip_y), 5, (0, 0, 255), -1)
            cv2.circle(frame, (r_hip_x, r_hip_y), 5, (0, 0, 255), -1)
            cv2.circle(frame, (l_knee_x, l_knee_y), 5, (0, 0, 255), -1)
            cv2.circle(frame, (r_knee_x, r_knee_y), 5, (0, 0, 255), -1)
            
            # Dibujar conexiones
            cv2.line(frame, (l_shoulder_x, l_shoulder_y), (r_shoulder_x, r_shoulder_y), (0, 255, 255), 2)
            cv2.line(frame, (l_shoulder_x, l_shoulder_y), (l_hip_x, l_hip_y), (0, 255, 255), 2)
            cv2.line(frame, (r_shoulder_x, r_shoulder_y), (r_hip_x, r_hip_y), (0, 255, 255), 2)
            cv2.line(frame, (l_hip_x, l_hip_y), (r_hip_x, r_hip_y), (0, 255, 255), 2)
            cv2.line(frame, (l_hip_x, l_hip_y), (l_knee_x, l_knee_y), (0, 255, 255), 2)
            cv2.line(frame, (r_hip_x, r_hip_y), (r_knee_x, r_knee_y), (0, 255, 255), 2)
            
            # Ángulos simulados
            l_knee_angle = random.uniform(10, 170)
            r_knee_angle = random.uniform(10, 170)
            
            # Mostrar ángulos
            cv2.putText(frame, f"L Knee: {l_knee_angle:.1f}", (l_knee_x, l_knee_y+20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"R Knee: {r_knee_angle:.1f}", (r_knee_x, r_knee_y+20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame, boxes

class SimpleActivityClassifier:
    """Clasificador simple de actividades."""
    
    def __init__(self):
        self.activities = ["Caminar", "Girar", "Sentarse", "Levantarse", "Desconocida"]
        self.last_activity = "Desconocida"
        self.activity_count = 0
    
    def classify(self, frame, boxes):
        """Clasifica actividades basándose en detecciones simples."""
        if len(boxes) == 0:
            return "Desconocida", 0
        
        # Simular clasificación basada en estadísticas
        self.activity_count += 1
        
        # Cambiar actividad cada 50 frames para simular detección
        if self.activity_count % 50 == 0:
            self.last_activity = random.choice(self.activities[:-1])
        
        # Simular confianza
        confidence = random.uniform(75, 95)
        
        return self.last_activity, confidence

class SimpleVideoAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Demostración de Análisis de Actividades Físicas")
        self.root.geometry("1000x700")
        
        # Inicializar detectores simplificados
        self.pose_detector = SimplePoseDetector()
        self.activity_classifier = SimpleActivityClassifier()
        
        # Variables
        self.cap = None
        self.is_capturing = False
        self.current_activity = "Desconocida"
        self.confidence = 0.0
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        
        # Configurar interfaz
        self.setup_ui()
        
        # Actualizar FPS cada segundo
        self.root.after(1000, self.update_fps)
    
    def setup_ui(self):
        # Frame principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame de video
        video_frame = ttk.LabelFrame(main_frame, text="Video en Tiempo Real")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Panel de información
        info_frame = ttk.LabelFrame(main_frame, text="Información", width=300)
        info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)
        
        # Información de actividad
        activity_frame = ttk.Frame(info_frame)
        activity_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(activity_frame, text="Actividad:", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        self.activity_label = ttk.Label(activity_frame, text="Desconocida", font=("Arial", 12))
        self.activity_label.pack(anchor=tk.W, padx=10)
        
        ttk.Label(activity_frame, text="Confianza:").pack(anchor=tk.W)
        self.confidence_label = ttk.Label(activity_frame, text="0%")
        self.confidence_label.pack(anchor=tk.W, padx=10)
        
        ttk.Label(activity_frame, text="FPS:").pack(anchor=tk.W)
        self.fps_label = ttk.Label(activity_frame, text="0")
        self.fps_label.pack(anchor=tk.W, padx=10)
        
        # Separador
        ttk.Separator(info_frame, orient='horizontal').pack(fill=tk.X, padx=5, pady=10)
        
        # Controles
        controls_frame = ttk.Frame(info_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(controls_frame, text="Cámara:").pack(anchor=tk.W)
        
        # Selector de cámara
        self.camera_var = tk.StringVar(value="0")
        camera_frame = ttk.Frame(controls_frame)
        camera_frame.pack(fill=tk.X, pady=5)
        
        for i in range(3):
            ttk.Radiobutton(camera_frame, text=f"Cámara {i}", variable=self.camera_var, value=str(i)).pack(side=tk.LEFT, padx=5)
        
        # Botón de captura
        self.capture_button = ttk.Button(controls_frame, text="Iniciar Captura", command=self.toggle_capture)
        self.capture_button.pack(fill=tk.X, pady=10)
        
        # Separador
        ttk.Separator(info_frame, orient='horizontal').pack(fill=tk.X, padx=5, pady=10)
        
        # Información del proyecto
        info_text = (
            "Sistema de Anotación de Video\n"
            "para Análisis de Actividades Físicas\n\n"
            "Esta es una demostración simplificada\n"
            "que muestra la detección básica de personas\n"
            "y la clasificación simulada de actividades.\n\n"
            "La versión completa utiliza MediaPipe\n"
            "para un seguimiento preciso de articulaciones\n"
            "y modelos entrenados para la clasificación."
        )
        
        info_label = ttk.Label(info_frame, text=info_text, justify=tk.LEFT, wraplength=280)
        info_label.pack(padx=5, pady=10)
    
    def toggle_capture(self):
        if self.is_capturing:
            # Detener captura
            self.is_capturing = False
            self.capture_button.config(text="Iniciar Captura")
            
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        else:
            # Iniciar captura
            try:
                camera_idx = int(self.camera_var.get())
                self.cap = cv2.VideoCapture(camera_idx)
                
                if self.cap.isOpened():
                    self.is_capturing = True
                    self.capture_button.config(text="Detener Captura")
                    self.process_frames()
                else:
                    tk.messagebox.showerror("Error", f"No se pudo abrir la cámara {camera_idx}")
                    self.cap = None
            except Exception as e:
                tk.messagebox.showerror("Error", f"Error al iniciar la cámara: {str(e)}")
    
    def process_frames(self):
        if not self.is_capturing or self.cap is None:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.toggle_capture()  # Detener captura
            return
        
        # Incrementar contador de frames
        self.frame_count += 1
        
        # Procesar frame
        processed_frame, boxes = self.pose_detector.detect(frame)
        
        # Clasificar actividad
        activity, confidence = self.activity_classifier.classify(frame, boxes)
        
        # Actualizar información
        self.current_activity = activity
        self.confidence = confidence
        self.activity_label.config(text=activity)
        self.confidence_label.config(text=f"{confidence:.1f}%")
        
        # Añadir texto al frame
        cv2.putText(processed_frame, f"Actividad: {activity}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(processed_frame, f"Confianza: {confidence:.1f}%", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Actualizar visualización
        self.update_video_frame(processed_frame)
        
        # Programar siguiente frame
        self.root.after(10, self.process_frames)
    
    def update_video_frame(self, frame):
        # Convertir a RGB para PIL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convertir a formato PIL
        pil_img = Image.fromarray(rgb_frame)
        
        # Redimensionar manteniendo proporción
        width, height = 640, 480
        pil_img.thumbnail((width, height), Image.ANTIALIAS if hasattr(Image, 'ANTIALIAS') else Image.LANCZOS)
        
        # Convertir a formato tkinter
        tk_img = ImageTk.PhotoImage(image=pil_img)
        
        # Actualizar etiqueta
        self.video_label.config(image=tk_img)
        self.video_label.image = tk_img  # Mantener referencia
    
    def update_fps(self):
        current_time = time.time()
        elapsed = current_time - self.last_time
        
        if elapsed > 0:
            self.fps = self.frame_count / elapsed
            self.fps_label.config(text=f"{self.fps:.1f}")
            
            # Reiniciar contador
            self.frame_count = 0
            self.last_time = current_time
        
        # Programar siguiente actualización
        self.root.after(1000, self.update_fps)

def main():
    # Crear directorios si no existen
    for directory in ["data", "data/dataset_raw", "data/dataset_processed"]:
        os.makedirs(directory, exist_ok=True)
    
    # Iniciar aplicación
    root = tk.Tk()
    app = SimpleVideoAnalyzerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()