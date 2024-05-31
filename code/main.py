import numpy as py #pip install numpy
import cv2 #pip install opencv-python
import pytesseract #pip install pytesseract
import skimage #python -m pip install scikit-image

def divideColor(img):
        if not(len(img.shape) == 3 and img.shape[2] == 3):
            raise ValueError('La imagen no contiene 3 canales de color')
        azul = img.copy()
        azul[:, :, 1] = 0  # Eliminar el canal verde
        azul[:, :, 2] = 0  # Eliminar el canal rojo
        
        verde = img.copy()
        verde[:, :, 0] = 0  # Eliminar el canal azul
        verde[:, :, 2] = 0  # Eliminar el canal rojo
        
        rojo = img.copy()
        rojo[:, :, 0] = 0  # Eliminar el canal azul
        rojo[:, :, 1] = 0  # Eliminar el canal verde
        
        # Mostrar las imágenes resultantes
        cv2.imshow('Canal Rojo', rojo)
        cv2.imshow('Canal Verde', verde)
        cv2.imshow('Canal Azul', azul)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

img = cv2.imread("./dataset/coche1.png")
print(img.shape)
divideColor(img)