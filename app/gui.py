# app/gui_components.py
# Agregar a la clase InfoPanel

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