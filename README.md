# Detector de Personas con IA - EL RETO

> **INFORMACIÓN PARA EVALUACIÓN**
>
> **Proyecto:** Detector de Personas con IA  
> **Fecha:** 10 de mayo de 2025  
> **Curso:** Quinto semestre - EL RETO  
> **Plataforma:** MacOS (optimizado para MacBook M2)  
> **Lenguaje:** Python 3.9+
>
> **Equipo de Desarrollo:**
> - **Yenci:** Investigación de la API de inteligencia artificial e instalación de dependencias necesarias
> - **Ulises:** Desarrollo de la interfaz del programa, implementación de la cámara y visualización en pantalla
> - **Raquel:** Integración de componentes y pruebas de funcionalidad del sistema
>
> **Tecnologías utilizadas:**
> - OpenCV para captura y procesamiento de imágenes
> - Hugging Face (facebook/detr-resnet-50) para detección de objetos
> - Flask para la interfaz web interactiva
> - Waitress como servidor WSGI de producción
>
> **Funcionalidades implementadas:**
> - Detección en tiempo real de personas y objetos
> - Interfaz web con visualización del video
> - Captura manual y automática configurable
> - Análisis con modelo de inteligencia artificial
> - Modo de prueba para entornos sin cámara
>
> **Descripción del Reto:**
> Desarrollo de un programa en Python que valide, utilizando la cámara de las laptops, si lo que se visualiza es una persona. En caso negativo, el sistema debe indicar que no es una persona y sugerir qué podría ser, de forma humorística.

---

## Proceso de Desarrollo

Para el desarrollo del proyecto, utilizamos un enfoque basado en IA generativa. Creamos un prompt específico que describiera nuestras necesidades y lo procesamos con el modelo DeepSeek para obtener una guía de implementación detallada:

**Prompt original:**
```
Necesito un programa en Python para que valide utilizando la cámara de las laptops, que analice si es una persona, si no es una persona que nos diga que no es y una idea de lo que el considera que es.

Vamos a utilizar vscode, python y una MacBook con m2. Necesito que tenga una iterfaz gráfica con html en local con acciones en el codigo para que no tengamos problemas con los puertos, la intergaz debe mostrar un apartado para la cámara al centro de la pantalla, un cuadro de texto que diga si es una persona y sino que muestre una sugerencia de lo que puede ser de forma graciosa y tenemos que ulitzar apis gratuitas que no sean las siguientes:

Utilizaremos:
* Hugging Face

Requisitos del Programa:
- Captura de Imagen: Desarrollar un programa que tome una foto a través de la cámara web de la computadora.
- Uso de API de IA: Investigar y seleccionar una API de inteligencia artificial que permita subir la imagen capturada y determinar si hay una persona en la foto.
- Resultados Esperados: El programa debe mostrar el resultado de la detección (si hay una persona o no).
- Opcional: Ampliar el programa para que describa otros objetos identificados en la imagen.
```

**Especificaciones técnicas obtenidas:**
```
Necesito un programa en Python (3.9+) para macOS (MacBook M2) que:

1. Captura de Imagen:
   - Usar la cámara integrada via OpenCV
   - Captura manual con botón o automática cada X segundos
   - Mostrar vista previa en tiempo real

2. Análisis con IA:
   - Usar Hugging Face (modelo ViT o similar) para:
     * Detectar si hay personas (con confianza >80%)
     * Identificar otros objetos principales
   - Manejar errores de API (reintentos, timeout)

3. Interfaz Web Local:
   - Flask con HTML/CSS responsive
   - Diseño centrado con:
     * Vista de cámara (640x480)
     * Resultados de detección destacados
     * Sugerencias humorísticas para no-humanos
   - Auto-gestión de puertos (evitar conflictos)

4. Requisitos Adicionales:
   - Mostrar bounding boxes opcionales
   - Historial de últimas 3 detecciones
   - Modo demo con imágenes de prueba
   - Configuración sencilla (intervalos, sensibilidad)

5. Exclusiones:
   - No usar otras APIs además de Hugging Face
   - Evitar dependencias complejas
```

Basados en estas especificaciones, desarrollamos el sistema utilizando el modelo `facebook/detr-resnet-50` de Hugging Face para la detección de objetos y personas.

---

## Documentación técnica

### Características

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