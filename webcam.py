#En la parte final del codigo vienen comentadas las funciones detectar_y_predecir_emocionesCNN y detectar_y_predecir_emocionesMLP
#Estas funciones son las encargadas de detectar y predecir las emociones en tiempo real, la primera función detecta las emociones
#utilizando un modelo CNN y la segunda función detecta las emociones utilizando un modelo MLP
#El usuario puede seleccionar cual de las dos usar descomentando la función que desee utilizar esto con la finalidad de mantener un
#Frame mas limpio y no saturar la pantalla con información innecesaria

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import face_recognition
from PIL import Image

# Cargar el modelo previamente entrenado para la detección de emociones
modelo_emociones = load_model("modeloFERCNN.h5")
modelomlp = load_model("7MLPFER.h5")

# Etiquetas de las imágenes del dataset
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad','surprise']

# Función para detectar caras y predecir emociones en tiempo real
def detectar_y_predecir_emocionesCNN(frame, modelo):
    try:
        # Convertir el frame a escala de grises
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar las ubicaciones de las caras en el frame
        face_locations = face_recognition.face_locations(gray)

        if face_locations:
            for (top, right, bottom, left) in face_locations:
                # Recortar la región de interés (ROI) correspondiente a la cara
                roi = gray[top:bottom, left:right]
                
                # Redimensionar la ROI a 150x150
                resized_roi = cv2.resize(roi, (48, 48))
                
                # Normalizar la ROI (dividiendo por 255)
                resized_roi_normalized = resized_roi / 255.0

                # Expandir las dimensiones para que coincidan con las dimensiones de entrada del modelo CNN
                resized_roi_normalized_expanded = np.expand_dims(resized_roi_normalized, axis=-1)
                resized_roi_normalized_expanded = np.expand_dims(resized_roi_normalized_expanded, axis=0)

                # Realizar la predicción de emociones en la ROI
                predictions = modelo.predict(resized_roi_normalized_expanded)[0]

                # Obtener el índice de la etiqueta con mayor probabilidad
                idx_etiqueta = np.argmax(predictions)

                # Obtener la etiqueta correspondiente
                etiqueta = labels[idx_etiqueta]

                # Mostrar la etiqueta en la pantalla
                etiqueta_texto = "Prediccion CNN: " + etiqueta
                cv2.putText(frame, etiqueta_texto, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
               

                # Dibujar un rectángulo alrededor de la cara detectada
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Mostrar todas las etiquetas con sus probabilidades en la parte superior del frame
                for i, (label, prob) in enumerate(zip(labels, predictions)):
                    text = f"{label}: {prob:.2f}"
                    cv2.putText(frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    except Exception as e:
        print(e)            


def detectar_y_predecir_emocionesMLP(frame, modelo):
    # Convertir el frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    resized_image = cv2.resize(gray, (48, 48))

    caras = face_recognition.face_locations(frame)

    if caras is not None:

        face_landmarks_list = face_recognition.face_landmarks(resized_image)

        #recorrer los puntos de la cara y añadirlos a una lista
        transformed_landmarks = []
        for landmarks in face_landmarks_list:
            for point in landmarks.values():
                transformed_landmarks.extend(point)

        #rellenar con ceros si no se detectan todos los puntos
        while len(transformed_landmarks) < 144:
            transformed_landmarks.append([0, 0])
        #recortar la lista a 144 puntos en caso de que se detecten más
        transformed_landmarks = transformed_landmarks[:144]

        
        transformed_landmarks = np.array(transformed_landmarks).tolist()

        # Realizar la predicción de emociones en la ROI
        roi_predicted = modelo.predict(np.expand_dims(transformed_landmarks, axis=0))

        # Obtener el índice de la etiqueta con mayor probabilidad
        idx_etiqueta = np.argmax(roi_predicted)

        # Obtener la etiqueta correspondiente
        etiqueta = labels[idx_etiqueta]

        texto = " prediccion MLP: " + etiqueta

        # Mostrar la etiqueta en la pantalla
        cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

           
# Iniciar la captura de video desde la cámara web
video_capture = cv2.VideoCapture(0)


while True:
    # Capturar un frame de la cámara
    ret, frame = video_capture.read()

    # Detectar y predecir emociones en el frame
    detectar_y_predecir_emocionesCNN(frame, modelo_emociones)

    #detectar_y_predecir_emocionesMLP(frame, modelomlp)

    # Mostrar el frame resultante
    cv2.imshow('Emociones en tiempo real', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de video y cerrar las ventanas
video_capture.release()
cv2.destroyAllWindows()