import os
import time
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

class PersonDetector:
    def __init__(self, model_name="facebook/detr-resnet-50", confidence_threshold=0.8):
        """
        Inicializa el detector de personas y objetos
        :param model_name: Nombre o ruta del modelo a usar
        :param confidence_threshold: Umbral de confianza para detecciones (0.0-1.0)
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.processor = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.history = []  # Historial de detecciones
        self.max_history = 3
        self.retry_count = 0
        self.max_retries = 3
        self.last_error_time = 0
        self.retry_wait = 5  # segundos
        self._load_model()
    
    def _load_model(self):
        """Carga el modelo de detección desde Hugging Face"""
        try:
            print(f"Cargando modelo de detección desde {self.model_name}...")
            print(f"Usando dispositivo: {self.device}")
            
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForObjectDetection.from_pretrained(self.model_name)
            
            # Mover modelo a GPU si está disponible
            self.model.to(self.device)
            print("Modelo cargado correctamente")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            self.model = None
            self.processor = None
    
    def _can_retry(self):
        """Determina si se puede reintentar después de un error"""
        if self.retry_count >= self.max_retries:
            current_time = time.time()
            # Reiniciar conteo de reintentos si ha pasado suficiente tiempo
            if current_time - self.last_error_time > self.retry_wait:
                self.retry_count = 0
                return True
            return False
        return True

    def detect(self, image):
        """
        Detecta personas y objetos en una imagen
        :param image: Imagen de OpenCV (numpy array en formato BGR)
        :return: Tupla (boxes, labels, scores, has_person, suggestions)
        """
        # Valores por defecto en caso de error
        empty_result = ([], [], [], False, "No se pudieron realizar detecciones.")
        
        # Verificar si el modelo está cargado
        if self.model is None or self.processor is None:
            if self._can_retry():
                print("Reintentando cargar el modelo...")
                self._load_model()
                self.retry_count += 1
                self.last_error_time = time.time()
            if self.model is None or self.processor is None:
                return empty_result
        
        try:
            # Convertir imagen de BGR (OpenCV) a RGB (PIL)
            image_rgb = cv2_to_pil(image)
            
            # Preprocesar la imagen
            inputs = self.processor(images=image_rgb, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Realizar la inferencia
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Postprocesar los resultados
            target_sizes = torch.tensor([image_rgb.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs, threshold=self.confidence_threshold, target_sizes=target_sizes)[0]
            
            # Extraer cajas, puntuaciones y etiquetas
            boxes = results["boxes"].cpu().numpy()
            scores = results["scores"].cpu().numpy()
            labels = results["labels"].cpu().numpy()
            
            # Convertir etiquetas numéricas a texto
            label_names = [self.model.config.id2label[label.item()] for label in labels]
            
            # Verificar si hay personas
            has_person = any(label.lower() == "person" for label in label_names)
            
            # Crear una sugerencia humorística si no hay personas
            suggestion = self._generate_suggestion(has_person, label_names)
            
            # Actualizar historial
            detection_entry = {
                'timestamp': time.strftime("%H:%M:%S"),
                'has_person': has_person,
                'objects': [f"{label} ({score:.2f})" for label, score in zip(label_names, scores)]
            }
            self._update_history(detection_entry)
            
            return boxes.tolist(), label_names, scores.tolist(), has_person, suggestion
            
        except Exception as e:
            print(f"Error al detectar objetos: {e}")
            self.retry_count += 1
            self.last_error_time = time.time()
            return empty_result
    
    def _generate_suggestion(self, has_person, detected_objects):
        """
        Genera sugerencias humorísticas basadas en detecciones
        :param has_person: Si hay personas detectadas
        :param detected_objects: Lista de objetos detectados
        :return: Sugerencia humorística
        """
        if has_person:
            return "¡Humano detectado! ¡Sistema de monitoreo funcionando correctamente!"
        
        if not detected_objects:
            suggestions = [
                "No veo a nadie. ¿Se habrán ido todos a tomar café?",
                "La habitación está vacía. Momento perfecto para practicar tu baile.",
                "Parece que estoy solo. Tal vez debería aprender a meditar."
            ]
        else:
            object_types = list(set(detected_objects))  # Eliminar duplicados
            
            if 'cat' in object_types or 'dog' in object_types:
                suggestions = [
                    "¡Veo una mascota! ¿Quién es el animalito bueno?",
                    "Una mascota detectada. ¡Los humanos no están pero dejaron un supervisor peludo!"
                ]
            elif any(item in object_types for item in ['cup', 'bottle']):
                suggestions = [
                    "¿Hora del café? Detecto bebidas, pero no humanos.",
                    "Veo tazas... ¿los humanos volverán pronto por su café?"
                ]
            elif any(item in object_types for item in ['chair', 'couch', 'bed']):
                suggestions = [
                    "Muebles vacíos detectados. ¿Se habrán cansado de sentarse?",
                    "Hay muebles pero nadie los usa. ¿Están en huelga los humanos?"
                ]
            elif any(item in object_types for item in ['laptop', 'cell phone', 'keyboard']):
                suggestions = [
                    "Veo tecnología pero no humanos. ¿La IA ya nos reemplazó?",
                    "Dispositivos abandonados. ¿Tal vez fueron a cargar sus baterías biológicas?"
                ]
            else:
                suggestions = [
                    f"Veo {', '.join(object_types)}, pero ningún humano a la vista.",
                    "Interesante colección de objetos, pero sin rastro de sus dueños."
                ]
        
        # Seleccionar una sugerencia aleatoria
        import random
        return random.choice(suggestions)
    
    def _update_history(self, entry):
        """
        Actualiza el historial de detecciones
        :param entry: Entrada nueva para el historial
        """
        self.history.append(entry)
        # Limitar el tamaño del historial
        if len(self.history) > self.max_history:
            self.history.pop(0)  # Eliminar la entrada más antigua
    
    def get_history(self):
        """
        Obtiene el historial de detecciones
        :return: Lista de detecciones recientes
        """
        return self.history


def cv2_to_pil(cv2_img):
    """
    Convierte una imagen de OpenCV a formato PIL
    :param cv2_img: Imagen de OpenCV (numpy array en formato BGR)
    :return: Imagen en formato PIL
    """
    # Convertir de BGR a RGB
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    # Convertir a formato PIL
    return Image.fromarray(rgb_img)


# Para la importación de cv2 en la función cv2_to_pil
import cv2