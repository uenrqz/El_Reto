import os
import sys
import time
import socket
import threading

# Agregar rutas adicionales para buscar módulos
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.13/lib/python3.13/site-packages')

# Intentar importar cv2
try:
    import cv2
except ImportError:
    print("Error: No se pudo importar OpenCV (cv2)")
    print("Rutas de búsqueda de Python:", sys.path)
    sys.exit(1)

import numpy as np
from flask import Flask, Response, render_template, request, jsonify
from waitress import serve
from camera import Camera
from detector import PersonDetector

# Configuración de la aplicación
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Configuración global
CAMERA_INDEX = 0
TEST_MODE = False  # Cambiado a False para usar la cámara real
CONFIDENCE_THRESHOLD = 0.8
AUTO_CAPTURE_INTERVAL = 5  # segundos
SHOW_BOUNDING_BOXES = True

# Instancias globales
camera = None
detector = None
last_result = {
    'boxes': [],
    'labels': [],
    'scores': [],
    'has_person': False,
    'suggestion': 'Inicializando...'
}
detection_lock = threading.Lock()

def initialize_system():
    """Inicializa la cámara y el detector"""
    global camera, detector
    camera = Camera(camera_index=CAMERA_INDEX, test_mode=TEST_MODE)
    detector = PersonDetector(confidence_threshold=CONFIDENCE_THRESHOLD)
    
    # Mostrar información sobre la cámara activa
    if camera.test_mode:
        print("\n📸 Modo de prueba: ✓ Activo (usando imágenes estáticas)")
    else:
        # Intentar obtener información de la cámara
        try:
            if camera.camera:
                # Obtener propiedades de la cámara si está disponible
                print(f"\n📸 Modo de prueba: ✗ Inactivo ({camera.camera_info})")
            else:
                print("\n📸 Modo de prueba: ✗ Inactivo (No se pudo inicializar la cámara)")
        except:
            print("\n📸 Modo de prueba: ✗ Inactivo (Error al obtener información de la cámara)")
    
    # Iniciar la captura automática
    camera.set_auto_capture(True, AUTO_CAPTURE_INTERVAL)

def generate_frames():
    """Generador para el streaming de video"""
    global last_result
    
    while True:
        # Leer un frame de la cámara
        success, frame = camera.read()
        if not success:
            print("Error: No se pudo leer de la cámara")
            # Generar un frame en blanco
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 255
            cv2.putText(frame, "Cámara no disponible", (150, 240), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Verificar si es hora de realizar una detección automática
        if camera.should_capture():
            detect_objects_in_frame(frame)
        
        # Dibujar las cajas delimitadoras si están habilitadas
        if SHOW_BOUNDING_BOXES and last_result['boxes']:
            with detection_lock:
                frame = camera.add_bounding_box(
                    frame, 
                    last_result['boxes'], 
                    last_result['labels'], 
                    last_result['scores']
                )
        
        # Codificar el frame a JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Enviar el frame como parte de la respuesta multipart
        yield (b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def detect_objects_in_frame(frame):
    """Detecta objetos en un frame y actualiza los resultados"""
    global last_result
    
    # Realizar detección
    boxes, labels, scores, has_person, suggestion = detector.detect(frame)
    
    # Actualizar resultados
    with detection_lock:
        last_result = {
            'boxes': boxes,
            'labels': labels,
            'scores': scores,
            'has_person': has_person,
            'suggestion': suggestion
        }
    
    print(f"Detección: {'✓ Persona detectada' if has_person else '✗ Ninguna persona'}")
    if labels:
        objects_str = ", ".join([f"{label} ({score:.2f})" for label, score in zip(labels, scores)])
        print(f"Objetos detectados: {objects_str}")
    else:
        print("No se detectaron objetos")

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Stream de video"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/detect', methods=['POST'])
def api_detect():
    """API para realizar detección manual"""
    success, frame = camera.manual_capture()
    if success:
        detect_objects_in_frame(frame)
        return jsonify({
            'success': True, 
            'has_person': last_result['has_person'],
            'suggestion': last_result['suggestion'],
            'objects': [{'label': label, 'score': score} 
                      for label, score in zip(last_result['labels'], last_result['scores'])],
            'history': detector.get_history()
        })
    return jsonify({'success': False, 'error': 'No se pudo capturar la imagen'})

@app.route('/api/status', methods=['GET'])
def api_status():
    """API para obtener el estado actual"""
    with detection_lock:
        return jsonify({
            'has_person': last_result['has_person'],
            'suggestion': last_result['suggestion'],
            'objects': [{'label': label, 'score': score} 
                      for label, score in zip(last_result['labels'], last_result['scores'])],
            'history': detector.get_history(),
            'settings': {
                'test_mode': TEST_MODE,
                'confidence': CONFIDENCE_THRESHOLD,
                'auto_capture': camera.auto_capture,
                'interval': camera.auto_capture_interval,
                'show_boxes': SHOW_BOUNDING_BOXES
            }
        })

@app.route('/api/settings', methods=['POST'])
def api_settings():
    """API para actualizar la configuración"""
    global SHOW_BOUNDING_BOXES, CONFIDENCE_THRESHOLD
    
    data = request.json
    if 'auto_capture' in data:
        camera.set_auto_capture(data['auto_capture'], 
                               data.get('interval', camera.auto_capture_interval))
    
    if 'show_boxes' in data:
        SHOW_BOUNDING_BOXES = data['show_boxes']
    
    if 'confidence' in data:
        CONFIDENCE_THRESHOLD = float(data['confidence'])
        detector.confidence_threshold = CONFIDENCE_THRESHOLD
    
    return jsonify({'success': True, 'settings': {
        'test_mode': TEST_MODE,
        'confidence': CONFIDENCE_THRESHOLD,
        'auto_capture': camera.auto_capture,
        'interval': camera.auto_capture_interval,
        'show_boxes': SHOW_BOUNDING_BOXES
    }})

def find_free_port():
    """Encuentra un puerto libre para usar"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def main():
    """Función principal"""
    initialize_system()
    
    # Encontrar un puerto libre
    port = find_free_port()
    host = '127.0.0.1'
    print(f"\n🚀 Iniciando servidor en http://{host}:{port}")
    print("🔍 Detector de personas iniciado")
    print("📸 Modo de prueba:" + (" ✓ Activo" if TEST_MODE else " ✗ Inactivo"))
    print(f"⚙️  Umbral de confianza: {CONFIDENCE_THRESHOLD}")
    print("👁️  Auto-captura:" + (" ✓ Activa" if camera.auto_capture else " ✗ Inactiva"))
    print(f"⏱️  Intervalo de captura: {camera.auto_capture_interval} segundos")
    print("📦 Bounding boxes:" + (" ✓ Activas" if SHOW_BOUNDING_BOXES else " ✗ Inactivas"))
    
    # Servir con Waitress
    serve(app, host=host, port=port)

if __name__ == '__main__':
    main()