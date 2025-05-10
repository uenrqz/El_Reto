# Detector de Personas con IA

Sistema de detección de personas y objetos en tiempo real utilizando OpenCV y modelos de Hugging Face. Desarrollado específicamente para MacBook M2 con Python 3.9+.

## Características

- **Captura de Imágenes**: 
  - Vista previa en tiempo real de la cámara integrada
  - Captura manual con botón o automática temporizada
  - Resolución ajustada a 640x480 píxeles

- **Análisis con IA**:
  - Detección de personas con confianza configurable
  - Identificación de múltiples objetos con modelo ViT
  - Gestión de errores y reintentos automáticos

- **Interfaz Web Local**:
  - Diseño responsive con Flask
  - Visualización en tiempo real de detecciones
  - Sugerencias humorísticas cuando no hay personas
  - Auto-gestión de puertos para evitar conflictos

- **Funciones Adicionales**:
  - Visualización opcional de bounding boxes
  - Historial de las 3 últimas detecciones
  - Modo demo con imágenes de prueba
  - Panel de configuración (sensibilidad, intervalos)

## Requisitos

- Python 3.9 o superior
- macOS (optimizado para MacBook M2)
- Cámara integrada funcional
- Conexión a Internet (para la primera carga del modelo)

## Instalación

1. **Clone o descargue este repositorio**:
   ```bash
   git clone <url_del_repositorio>
   cd detector_personas
   ```

2. **Cree un entorno virtual** (recomendado):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Instale las dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure su entorno** (opcional):
   - Para modo de prueba sin cámara, edite la variable `TEST_MODE` en `app.py` a `True`
   - Ajuste la sensibilidad modificando `CONFIDENCE_THRESHOLD` en `app.py`

## Uso

1. **Inicie la aplicación**:
   ```bash
   python app.py
   ```

2. **Acceda a la interfaz web**:
   - Abra su navegador web
   - Vaya a la dirección URL que aparece en la consola (por defecto `http://127.0.0.1:PUERTO`)

3. **Funciones principales**:
   - **Captura**: Pulse el botón "Capturar" para tomar una foto y analizarla
   - **Auto-captura**: Active/desactive la captura automática con el botón "Auto-captura"
   - **Configuración**: Ajuste el intervalo de captura, umbral de confianza, y visibilidad de cajas

## Estructura del Proyecto

```
detector_personas/
├── app.py              # Punto de entrada principal y servidor web
├── camera.py           # Módulo para manejo de la cámara
├── detector.py         # Módulo para detección de objetos con IA
├── requirements.txt    # Dependencias del proyecto
├── models/             # Carpeta donde se almacenan modelos y datos de prueba
│   └── test_images/    # Imágenes para el modo demo
├── static/             # Archivos estáticos para la web
│   └── style.css       # Estilos CSS
└── templates/          # Plantillas HTML
    └── index.html      # Página principal de la aplicación
```

## Solución de Problemas

- **Error al acceder a la cámara**: Asegúrese de que ninguna otra aplicación esté usando la cámara
- **Modelo no se descarga**: Verifique su conexión a Internet. El modelo se descarga automáticamente la primera vez
- **Rendimiento lento**: Ajuste el intervalo de captura a un valor más alto

## Notas Técnicas

- El modelo utilizado es `facebook/detr-resnet-50` de Hugging Face
- La primera ejecución será más lenta debido a la descarga del modelo
- La detección considera que hay personas cuando la confianza es superior al umbral configurado (0.8 por defecto)

## Licencia

Este proyecto está bajo la Licencia MIT.

---

Desarrollado para el curso EL RETO - Quinto semestre