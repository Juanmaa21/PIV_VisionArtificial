import numpy as py #pip install numpy
import cv2 #pip install opencv-python
import matplotlib.pyplot as plt #pip install matplotlib
import pytesseract #pip install pytesseract
import skimage #python -m pip install scikit-image

def divideColor(img):
        if not(len(img.shape) == 3 and img.shape[2] == 3):
            raise ValueError('La imagen no contiene 3 canales de color')
        
        #Canal 0: Azul, canal 1: Verde, canal 2: Rojo
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
        cv2.imwrite('./images/coche1_rojo.png', rojo)
        cv2.imwrite('./images/coche1_verde.png', verde)
        cv2.imwrite('./images/coche1_azul.png', azul)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

def plot_image(img, grayscale=True):
    plt.axis('off')
    if grayscale:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

def img_to_BYN(img):
    # Convertir la imagen a escala de grises
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

def umbralizacion(img):
    thresh = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY_INV)[1]
    return thresh

def detectarFormas(img):
    contours = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    return contours

def filtrarFormas(contours):
    proporcion_matricula = 520/110
    min_w = 80
    max_w = 110
    min_h = 25
    max_h = 52

    candidatos = []
    for contador in contours:
        x, y, w, h = cv2.boundingRect(contador)
        proporcion_de_aspecto = float(w) / h
        if(py.isclose(proporcion_de_aspecto, proporcion_matricula, atol=0.7) and (max_w > w > min_w) and (max_h > h > min_h)):
            candidatos.append(contador)

    return candidatos


img = cv2.imread("./dataset/coche1.png")
print(img.shape)

# divideColor(img)

gray = img_to_BYN(img)
print(gray.shape)
cv2.imshow('Escala de grises', gray)
cv2.imwrite('./images/coche1_gray.png', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()


thresh = umbralizacion(gray)
cv2.imshow('Imagen umbralizada',thresh)
cv2.imwrite('./images/coche1_umbral.png', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

contornos = detectarFormas(thresh)
canvas = py.zeros_like(img)
cv2.drawContours(canvas, contornos, -1, (0, 255, 0), 2)
plt.axis('off')
plt.imshow(canvas)

candidatos = filtrarFormas(contornos)
canvas = py.zeros_like(img)
cv2.drawContours(canvas, candidatos, -1, (0, 255, 0), 2)
plt.axis('off')
plt.imshow(canvas)


