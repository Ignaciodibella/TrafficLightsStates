import pickle
import cv2 as cv
import numpy as np

def homografia(frame_previo, frame_actual): 
    """
    Realiza la correspondencia de características entre dos fotogramas utilizando SIFT y calcula la homografía.

    Parámetros:
    - frame_previo: Fotograma anterior en formato de imagen.
    - frame_actual: Fotograma actual en formato de imagen.

    Retorno:
    - scale_x: Relación de escala calculada en la dirección horizontal.
    - scale_y: Relación de escala calculada en la dirección vertical.

    La función utiliza el algoritmo SIFT para detectar keypoints y calcular descriptores en las imágenes de entrada.
    Luego, se utiliza el algoritmo FLANN para encontrar coincidencias entre los descriptores. Se aplican filtros
    para seleccionar buenas coincidencias y se calcula la homografía mediante el método RANSAC.

    Si el número de buenas coincidencias es suficiente, se calcula la relación de escala en las direcciones horizontal
    y vertical a partir de la matriz de homografía. Se muestra un mensaje indicando si hay un cambio en la escala.
    """

    img1 = cv.cvtColor(frame_previo, cv.IMREAD_GRAYSCALE)
    img2 = cv.cvtColor(frame_actual, cv.IMREAD_GRAYSCALE)

    sift = cv.SIFT_create()
    keypoints1, descriptor1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptor2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    indexParams = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks = 50)
    flann = cv.FlannBasedMatcher(indexParams, searchParams)
    nNeighbors = 2
    matches = flann.knnMatch(descriptor1,descriptor2,k=nNeighbors)

    goodMatches = []
    for m,n in matches:
        if m.distance < 0.2*n.distance:
            goodMatches.append(m)

    minGoodMatches = 10 # Al menos 4 - restricción del método

    if len(goodMatches) > minGoodMatches:
        srcPts = np.float32([ keypoints1[m.queryIdx].pt for m in goodMatches ]).reshape(-1,1,2) 
        dstPts = np.float32([ keypoints2[m.trainIdx].pt for m in goodMatches ]).reshape(-1,1,2)
        errorThreshold = 5
        M, mask = cv.findHomography(srcPts,dstPts,cv.RANSAC,errorThreshold)

        scale_x = M[0, 0]
        scale_y = M[1, 1]

        """
        Si la escala en la dirección horizontal es 1, significa que no hay cambio en la escala. 
        Un valor mayor que 1 indicaría un aumento en la escala (zoom-in), 
        mientras que un valor menor que 1 indicaría una disminución en la escala (zoom-out).
        """
        print("Relación de escala calculada:", scale_x, "|", scale_y) # Parametrizar Verbose (0|1)
    else:
        print("No se encontraron suficientes coincidencias.")

    return scale_x, scale_y

def seleccionarRois(imagen):
    """
    Permite al usuario seleccionar regiones de interés (ROIs) en una imagen.

    Parámetros:
    - imagen: numpy.ndarray
        La imagen en la que se seleccionarán las regiones de interés.

    Retorno:
    - rois: list
        Una lista que contiene las coordenadas (x, y, ancho, alto) de cada región de interés seleccionada.

    Uso:
    1. Se abre una ventana que muestra la imagen y permite al usuario seleccionar múltiples ROIs.
    2. Para seleccionar un ROI, se hace clic y arrastra para formar un rectángulo alrededor de la región de interés deseada.
    3. Se presiona la tecla 'Espacio' o 'Enter' para confirmar la selección de la ROI y pasar al siguiente.
    4. Presionar tecla 'C' para iniciar nueva selección.
    5. Para finalizar la selección, se cierra la ventana. -> Doble espacio/enter.
    """

    cv.namedWindow("Select ROIs", cv.WINDOW_NORMAL)
    cv.resizeWindow("Select ROIs", imagen.shape[1], imagen.shape[0])
    cv.setWindowProperty("Select ROIs", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    cv.imshow("Select ROIs", imagen)

    rois = []

    while True:
        roi = cv.selectROI("Select ROIs", imagen, fromCenter=False, showCrosshair=True)
        if roi[2] > 0 and roi[3] > 0:
            rois.append(roi)

            cv.rectangle(imagen, (int(roi[0]), int(roi[1])), (int(roi[0] + roi[2]), int(roi[1] + roi[3])), (0, 255, 0), 2)
            cv.imshow("Select ROIs", imagen)

        key = cv.waitKey(0) & 0xFF
        if key == ord(' ') or key == 13:
            break

    cv.destroyWindow("Select ROIs")

    return rois

def identificarEstadoSemaforo(roi, imagen):
    """
    Identifica el color de un semáforo en una región de interés (ROI) de una imagen.

    Parámetros:
    - roi: tupla
        Las coordenadas (x, y, ancho, alto) de la región de interés que contiene el semáforo.
    - imagen: numpy.ndarray
        La imagen completa que contiene la región de interés.

    Retorna:
    - str
        Un string indicando el color del semáforo identificado. Puede ser "RED" o "Not RED".

    Uso:
    1. Proporcionar las coordenadas de la ROI que abarca el semáforo y la imagen completa.
    2. La función calculará el color del semáforo en la ROI y devolverá la etiqueta correspondiente.
    """
    x, y, w, h = roi

    # Asegúrate de que las coordenadas estén dentro de los límites de la imagen
    x = max(0, x)
    y = max(0, y)
    w = min(w, imagen.shape[1] - x)
    h = min(h, imagen.shape[0] - y)

    # Trunca la ROI para que esté completamente dentro de la imagen
    selected_region = imagen[y:y+h, x:x+w]

    if selected_region.shape[0] == 0 or selected_region.shape[1] == 0:
        # La ROI está completamente fuera de la imagen, no se puede procesar
        return "Not RED"
    
    hsv = cv.cvtColor(selected_region, cv.COLOR_BGR2HSV)

    # Espectro de Color a Considerar - Ajustar si es necesario
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])

    mask1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv.inRange(hsv, lower_red2, upper_red2)

    combined_mask = cv.bitwise_or(mask1, mask2)

    red_pixel_count = np.count_nonzero(combined_mask)

    threshold_percentage = 0.05 # Porcentaje de la imagen que debe contener el color buscado para que sea identificado como "True"
    if red_pixel_count / (w * h) > threshold_percentage:
        return "RED", (0, 0, 255) #True
    else:
        return "Not RED", (0, 255, 0) # False

def ajustarPosicionRois(rois, frame, escala_x, escala_y):
    """
    Ajusta las posiciones y dimensiones de un conjunto de ROIs en relación con un frame,
    aplicando factores de escala en las dimensiones x e y.

    Parámetros:
    - rois (list): Lista de tuplas que representan las ROIs, donde cada tupla contiene 
                   las coordenadas (x, y, ancho, alto) de una ROI.
    - frame (numpy.ndarray): Imagen sobre la cual se basará el ajuste de las ROIs.
    - escala_x (float): Factor de escala para ajustar las dimensiones en el eje x.
    - escala_y (float): Factor de escala para ajustar las dimensiones en el eje y.

    Retorna:
    list: Lista de tuplas que representan las ROIs ajustadas después de aplicar los factores de escala.
    """

    for i in range(len(rois)):
                roi = rois[i]
                x, y, w, h = roi
                
                # Calcula el desplazamiento en x e y basándote en la posición original con respecto al centro
                centro_x = frame.shape[1] // 2
                centro_y = frame.shape[0] // 2

                delta_x = x - centro_x
                delta_y = y - centro_y

                # Aplica el factor de escala a los desplazamientos
                delta_x_scaled = int(delta_x * (escala_x - 1))
                delta_y_scaled = int(delta_y * (escala_y - 1))

                # Ajusta las coordenadas de la ROI
                x_ajustado = x + delta_x_scaled
                y_ajustado = y + delta_y_scaled
                w_ajustado = int(w * escala_x)
                h_ajustado = int(h * escala_y)
                
                rois[i] = (x_ajustado, y_ajustado, w_ajustado, h_ajustado)
    return rois

if __name__ == '__main__':

    # Para usar un video ya guardado:
    video_path = input("Indique la ruta al video de entrada: ")
    cap = cv.VideoCapture(video_path)

    # Para usar una cámara de video en tiempo real:
    # cap = cv2.VideoCapture(0)

    datos_guardados = input("Quiere usar las ROI definidas anteriormente: [s/n] ")
    frame_previo = None

    if datos_guardados.lower() == "s":
        # Intentar cargar las ROIs desde el archivo de texto
        try:
            with open("rois.txt", "r") as file:
                lines = file.readlines()
                rois = [tuple(map(int, line.strip().split(','))) for line in lines]       

        except FileNotFoundError:
            print("El archivo de coordenadas no existe. Seleccionando ROIs...")
            ret, frame = cap.read()
            rois = seleccionarRois(frame)

            # Persistir Frame de referencia y resolución
            frame_previo = frame
            cv.imwrite('frame_previo.jpg', frame)

            # Persistir coordenadas y dimensiones de las ROIs definidas
            with open("rois.txt", "w") as file:
                for roi in rois:
                    file.write(','.join(map(str, roi)) + '\n')
                    
    else:
        ret, frame = cap.read()
        rois = seleccionarRois(frame)

        # Persistir Frame de referencia
        frame_previo = frame
        cv.imwrite('frame_previo.jpg', frame)

        # Lectura de coordenadas y dimensiones de las ROIs
        with open("rois.txt", "w") as file:
            for roi in rois:
                file.write(','.join(map(str, roi)) + '\n')

    if frame_previo is None or np.array_equal(frame_previo, None):
        frame_previo = cv.imread('frame_previo.jpg')

    k_muestreo = 10 # Si k_muestreo = 1, entonces se evaluan todos los fotogramas
    # k = 1 --> 180.55 segundos
    # k = 5 --> 42.20 segundos
    # k = 10 --> 24.33 segundos

    contador_intervalos = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        contador_intervalos += 1
        # Verificar cambio de resolución - Esto puede ocurrir antes de la ejecución, pero no durante. Hay que guardarla con las ultimas ROI en el txt

        # Verificar cambio de zoom/escala - Esto puede ocurrir antes y/o durante la ejecución
        if contador_intervalos == k_muestreo:
    
            escala_x, escala_y = homografia(frame_previo, frame) # Cuello de Botella

            # Filtrado de ruido provocado por la estimación del cambio de escala
            sensibilidad_zoom_in = 1.10
            sensibilidad_zoom_out = 0.90

            if escala_x >= sensibilidad_zoom_in or escala_x <= sensibilidad_zoom_out:
                rois = ajustarPosicionRois(rois, frame, escala_x, escala_y)
                frame_previo = frame

            contador_intervalos = 0

        for roi in rois:
            result, color = identificarEstadoSemaforo(roi, frame)
            cv.rectangle(frame, (int(roi[0]), int(roi[1])), (int(roi[0] + roi[2]), int(roi[1] + roi[3])), color, 2)
            cv.putText(frame, result, (int(roi[0]), int(roi[1]) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv.namedWindow("Real-Time Result", cv.WINDOW_NORMAL)
        cv.resizeWindow("Real-Time Result", frame.shape[1], frame.shape[0])
        cv.setWindowProperty("Real-Time Result", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        cv.imshow("Real-Time Result", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.imwrite('frame_previo.jpg', frame)
            break

    cap.release()
    cv.destroyAllWindows()