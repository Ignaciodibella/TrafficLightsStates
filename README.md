# Traffic Lights States
Este repositorio de GitHub contiene un conjunto de funciones y scripts en Python para el procesamiento de video en tiempo real, específicamente diseñado para la detección de semáforos y cambios en la escala de la imagen. El sistema utiliza algoritmos de visión por computadora, como SIFT para la correspondencia de características y el método RANSAC para calcular la homografía entre fotogramas consecutivos.

Las principales características incluyen:

1. **Detección de Semáforos:** Identifica y clasifica el color de los semáforos en una región de interés (ROI) definida por el usuario. La clasificación se basa en el análisis del contenido de píxeles y utiliza filtros de color en el espacio HSV.

2. **Corrección de Escala:** Detecta cambios en la escala entre fotogramas consecutivos, lo que puede indicar zoom in, zoom out o cambios en la resolución. Ajusta automáticamente las posiciones y dimensiones de las regiones de interés (ROIs) en consecuencia.

3. **Interfaz de Usuario para Selección de ROIs:** Permite al usuario seleccionar manualmente las regiones de interés en una imagen mediante una interfaz interactiva. También proporciona la opción de cargar coordenadas de ROIs desde un archivo de texto para un uso más rápido.

El repositorio incluye un script principal (`main.py`) que utiliza estas funciones para procesar un flujo de video en tiempo real y visualizar los resultados.

**Instrucciones de Uso:**
1. Ejecutar el script `main.py` y proporcionar la ruta al video de entrada o el 'id' de la cámara a emplear.
2. Seleccionar manualmente las ROIs o cargar coordenadas predefinidas.
3. Observar en tiempo real la detección de semáforos y ajuste automático de ROIs.

Este proyecto es ideal para aplicaciones de monitoreo de tráfico y análisis de video en entornos urbanos. Se recomienda revisar la documentación del código y las instrucciones para una comprensión detallada del funcionamiento y la configuración. ¡Disfruta explorando y mejorando este sistema de procesamiento de video en tiempo real!
