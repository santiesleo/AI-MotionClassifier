# src/data_collection.py
import cv2
import os
import sys
import time
from datetime import datetime

class VideoDataCollector:
    def __init__(self, output_dir="dataset_raw"):
        self.base_dir = output_dir
        self.activities = {
            '1': 'caminar_hacia',
            '2': 'caminar_regreso',
            '3': 'girar',
            '4': 'sentarse',
            '5': 'levantarse'
        }
        self.current_activity = None
        self.current_subject = None
        self.recording = False
        self.setup_directories()
        
    def setup_directories(self):
        """Crear estructura de directorios para los videos"""
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            print(f"Directorio de salida creado/verificado: {self.base_dir}")
        except Exception as e:
            print(f"Error al crear directorio: {e}")
            sys.exit(1)
    
    def init_camera(self):
        """Inicializar la cámara con múltiples intentos"""
        print("Intentando inicializar la cámara...")
        
        # Lista de posibles índices de cámara para probar
        camera_indices = [0, 1] if sys.platform == "darwin" else [0]  # Más opciones para macOS
        
        for idx in camera_indices:
            print(f"Intentando abrir cámara {idx}...")
            cap = cv2.VideoCapture(idx)
            
            if not cap.isOpened():
                print(f"No se pudo abrir la cámara {idx}")
                continue
            
            # Verificar si podemos leer un frame
            ret, frame = cap.read()
            if ret:
                print(f"Cámara {idx} inicializada correctamente")
                return cap
            
            cap.release()
        
        print("Error: No se pudo inicializar ninguna cámara")
        return None

    def get_video_path(self):
        """Generar ruta única para el video"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.current_subject}_{self.current_activity}_{timestamp}.mp4"
        return os.path.join(self.base_dir, filename)
    
    def run(self):
        # Inicializar cámara
        cap = self.init_camera()
        if cap is None:
            print("Error crítico: No se pudo inicializar la cámara")
            return
        
        # Determinar el codec apropiado según el sistema operativo
        if sys.platform == "darwin":  # macOS
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
        else:  # Windows/Linux
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        out = None
        print("Sistema iniciado. Presione 'q' para salir.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error al leer frame de la cámara")
                    break
                
                # Mostrar información en pantalla
                info_text = [
                    f"Sujeto actual: {self.current_subject if self.current_subject else 'No seleccionado'}",
                    f"Actividad: {self.current_activity if self.current_activity else 'No seleccionada'}",
                    "GRABANDO" if self.recording else "En espera"
                ]
                
                for i, text in enumerate(info_text):
                    cv2.putText(frame, text, (10, 30 + i*30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, 
                               (0, 255, 0) if self.recording else (0, 0, 255), 2)
                
                # Mostrar instrucciones
                instructions = [
                    "Teclas:",
                    "S + número: Seleccionar sujeto",
                    "1-5: Seleccionar actividad",
                    "R: Iniciar/Detener grabación",
                    "Q: Salir"
                ]
                
                for i, text in enumerate(instructions):
                    cv2.putText(frame, text, (10, frame.shape[0] - 150 + i*30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                cv2.imshow('Captura de Videos', frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                # Manejar entrada de sujeto
                if chr(key).upper() == 'S':
                    subject_num = input("Ingrese número de sujeto: ")
                    self.current_subject = f"sujeto_{subject_num.zfill(2)}"
                    print(f"Sujeto seleccionado: {self.current_subject}")
                    
                # Manejar selección de actividad
                elif chr(key) in self.activities:
                    self.current_activity = self.activities[chr(key)]
                    print(f"Actividad seleccionada: {self.current_activity}")
                    
                # Iniciar/Detener grabación
                elif chr(key).upper() == 'R':
                    if not self.recording and self.current_subject and self.current_activity:
                        video_path = self.get_video_path()
                        out = cv2.VideoWriter(video_path, fourcc, 30.0, 
                                            (frame.shape[1], frame.shape[0]))
                        if out.isOpened():
                            self.recording = True
                            print(f"Iniciando grabación: {video_path}")
                        else:
                            print("Error al crear archivo de video")
                    elif self.recording:
                        out.release()
                        self.recording = False
                        print("Grabación detenida")
                    else:
                        print("Error: Seleccione sujeto y actividad antes de grabar")
                
                # Grabar frame si está en grabación
                if self.recording and out is not None:
                    out.write(frame)
                
                # Salir
                if key == ord('q'):
                    break
                
        except Exception as e:
            print(f"Error durante la ejecución: {e}")
        
        finally:
            print("Limpiando recursos...")
            if out is not None:
                out.release()
            cap.release()
            cv2.destroyAllWindows()
            print("Programa finalizado")


if __name__ == "__main__":
    collector = VideoDataCollector()
    collector.run()