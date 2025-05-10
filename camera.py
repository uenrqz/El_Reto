import cv2
import time
import os
import numpy as np
from datetime import datetime

class Camera:
    def __init__(self, camera_index=0, test_mode=False):
        """
        Inicializa la cámara
        :param camera_index: Índice de la cámara (0 para la cámara integrada)
        :param test_mode: Si es True, usará imágenes de prueba en lugar de la cámara real
        """
        self.camera_index = camera_index
        self.test_mode = test_mode
        self.cap = None
        self.camera = None
        self.test_images = []
        self.test_image_index = 0
        self.frame = None
        self.last_frame_time = 0
        self.auto_capture = False
        self.auto_capture_interval = 5  # segundos
        self.camera_info = "No inicializada"
        
        # Cargar imágenes de prueba si está en modo de prueba
        if self.test_mode:
            self._load_test_images()
        else:
            self._init_camera()
    
    def _init_camera(self):
        """Inicializa la conexión con la cámara"""
        try:
            print("Intentando abrir la cámara...")
            self.cap = cv2.VideoCapture(self.camera_index)
            
            # Verificar si la cámara se abrió correctamente
            if not self.cap.isOpened():
                print("Error: No se pudo abrir la cámara")
                self.camera_info = "Error: No se pudo abrir la cámara"
                self._fallback_to_test_mode()
                return
            
            # La cámara está abierta, obtener propiedades
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.camera = {
                "index": self.camera_index,
                "width": width,
                "height": height,
                "fps": fps
            }
            
            self.camera_info = f"Cámara {self.camera_index}: {width}x{height} @ {fps}fps"
            print(f"Cámara inicializada: {self.camera_info}")
            
            # Verificar si podemos leer un frame (para confirmar que funciona)
            ret, test_frame = self.cap.read()
            if not ret:
                print("Advertencia: No se pudo leer un frame de la cámara, verificando permisos...")
                self._fallback_to_test_mode()
                return
                
            print("Cámara funcionando correctamente")
            
        except Exception as e:
            print(f"Error al inicializar la cámara: {e}")
            self.camera_info = f"Error: {str(e)}"
            self._fallback_to_test_mode()
    
    def _fallback_to_test_mode(self):
        """Cambia al modo de prueba cuando falla la cámara"""
        print("Cambiando a modo de prueba debido a problemas con la cámara")
        if self.cap:
            self.cap.release()
            self.cap = None
        self.test_mode = True
        self._load_test_images()
    
    def _load_test_images(self):
        """Carga imágenes de prueba para el modo demo"""
        test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "test_images")
        os.makedirs(test_dir, exist_ok=True)
        
        # Si no hay imágenes de prueba, crear una imagen de ejemplo
        if not os.listdir(test_dir):
            # Crear una imagen de muestra con texto
            img = np.ones((480, 640, 3), dtype=np.uint8) * 200
            cv2.putText(img, "MODO DE PRUEBA", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imwrite(os.path.join(test_dir, "test_image.jpg"), img)
        
        # Cargar imágenes de prueba
        for file in os.listdir(test_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(test_dir, file)
                self.test_images.append(cv2.imread(img_path))
        
        # Si no se encontraron imágenes, crear una imagen de prueba
        if not self.test_images:
            img = np.ones((480, 640, 3), dtype=np.uint8) * 200
            cv2.putText(img, "NO SE ENCONTRARON IMÁGENES", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (0, 0, 255), 2, cv2.LINE_AA)
            self.test_images.append(img)
    
    def __del__(self):
        """Destructor: libera los recursos de la cámara"""
        self.release()
    
    def release(self):
        """Libera los recursos de la cámara"""
        if self.cap and not self.test_mode:
            self.cap.release()
            self.cap = None
    
    def read(self):
        """
        Lee un frame de la cámara o una imagen de prueba
        :return: Tupla (éxito, imagen)
        """
        if self.test_mode:
            if not self.test_images:
                return False, None
            # Rotar entre las imágenes de prueba
            self.frame = self.test_images[self.test_image_index].copy()
            self.test_image_index = (self.test_image_index + 1) % len(self.test_images)
            return True, self.frame
        
        if self.cap is None:
            self._init_camera()
            if self.cap is None:
                return False, None
        
        ret, self.frame = self.cap.read()
        return ret, self.frame
    
    def get_jpeg(self):
        """
        Convierte el frame actual a JPEG para transmitir por web
        :return: Bytes de la imagen en formato JPEG
        """
        if self.frame is None:
            success, frame = self.read()
            if not success:
                # Retornar una imagen en blanco si falla la lectura
                blank = np.ones((480, 640, 3), dtype=np.uint8) * 255
                cv2.putText(blank, "Cámara no disponible", (150, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                _, buffer = cv2.imencode('.jpg', blank)
                return buffer.tobytes()
        else:
            # Usar el frame almacenado
            frame = self.frame
        
        # Generar timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 1, cv2.LINE_AA)
        
        # Si está en modo de prueba, agregar marca de agua
        if self.test_mode:
            cv2.putText(frame, "MODO DEMO", (frame.shape[1] - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        _, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()
    
    def set_auto_capture(self, enabled, interval=5):
        """
        Configura la captura automática
        :param enabled: True para activar la captura automática
        :param interval: Intervalo en segundos entre capturas
        """
        self.auto_capture = enabled
        self.auto_capture_interval = interval
    
    def should_capture(self):
        """
        Determina si es momento de hacer una captura automática
        :return: True si debe capturar, False en caso contrario
        """
        if not self.auto_capture:
            return False
        
        current_time = time.time()
        if current_time - self.last_frame_time >= self.auto_capture_interval:
            self.last_frame_time = current_time
            return True
        return False
    
    def manual_capture(self):
        """
        Realiza una captura manual
        :return: La imagen capturada
        """
        success, frame = self.read()
        if success:
            self.last_frame_time = time.time()
        return success, frame
    
    def add_bounding_box(self, frame, boxes, labels, scores):
        """
        Añade cajas delimitadoras a la imagen
        :param frame: Imagen a modificar
        :param boxes: Lista de cajas [x1, y1, x2, y2]
        :param labels: Lista de etiquetas
        :param scores: Lista de puntuaciones de confianza
        :return: Imagen con cajas delimitadoras
        """
        if frame is None:
            return None
        
        img = frame.copy()
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Color basado en la etiqueta (rojo para personas, verde para otros)
            color = (0, 255, 0) if label.lower() == 'person' else (0, 165, 255)
            
            # Dibujar rectángulo
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Añadir etiqueta y puntuación
            text = f"{label}: {score:.2f}"
            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 2)
        
        return img