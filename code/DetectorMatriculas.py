import numpy as np
import cv2
import matplotlib.pyplot as plt
import pytesseract
import skimage

class DetectorMatriculas:
    def __init__(self):
        # Configurar la ruta de Tesseract
        pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files (x86)/Tesseract-OCR/tesseract.exe'
        self.ratio = 520/110 # Dimensiones matricula europea
        self.min_w = 64
        self.max_w = 256
        self.min_h = 16
        self.max_h = 64
    
    # Muestra una imagen cualquiera
    def mostrarImagen(self, img):
        cv2.imshow('Imagen',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Muestra los 3 canales de color de una imagen
    def mostrarRGB(self, img):
        if not(len(img.shape) == 3 and img.shape[2] == 3):
            raise ValueError('La imagen no contiene 3 canales de color')
        
        # Canal 0: Azul, canal 1: Verde, canal 2: Rojo
        azul = img.copy()
        azul[:, :, 1] = 0  # Eliminar el canal verde
        azul[:, :, 2] = 0  # Eliminar el canal rojo
        
        verde = img.copy()
        verde[:, :, 0] = 0  # Eliminar el canal azul
        verde[:, :, 2] = 0  # Eliminar el canal rojo
        
        rojo = img.copy()
        rojo[:, :, 0] = 0  # Eliminar el canal azul
        rojo[:, :, 1] = 0  # Eliminar el canal verde

        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(cv2.cvtColor(rojo, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Canal Rojo')
        axes[0].axis('off')
        
        axes[1].imshow(cv2.cvtColor(verde, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Canal Verde')
        axes[1].axis('off')
        
        axes[2].imshow(cv2.cvtColor(azul, cv2.COLOR_BGR2RGB))
        axes[2].set_title('Canal Azul')
        axes[2].axis('off')
        
        plt.show()

    # Convertir la imagen a escala de grises
    def toEscalaDeGrises(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img_gray
    
    # Umbrealizar una imagen con un valor de referencia
    def aplicarUmbrealizacion(self, img, value):
        return cv2.threshold(img, value, 255, cv2.THRESH_BINARY_INV)[1]
    
    # Umbrealizar una imagen con un valor de referencia
    def aplicarUmbrealizacionAdaptativa(self, img):
        return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 13)
    
    def encontrarContornos(self, img):
        return cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    def mostrarContornos(self, img, contornos):
        canvas = np.zeros_like(img)
        cv2.drawContours(canvas, contornos, -1, (0, 255, 0), 2)
        plt.axis('off')
        plt.imshow(canvas)
        plt.show()

    def filtrarCandidatos(self, contornos):
        candidatos = []
        for c in contornos:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = float(w) / h
            if (np.isclose(aspect_ratio, self.ratio, atol=0.7) and
               (self.max_w > w > self.min_w) and
               (self.max_h > h > self.min_h)):
                candidatos.append(c)
        return candidatos
    
    def mostrarCandidatos(self, img, candidatos):
        canvas = np.zeros_like(img)
        cv2.drawContours(canvas, candidatos, -1, (0, 255, 0), 2)
        plt.axis('off')
        plt.imshow(canvas)
        plt.show()

    def filtrarMenorCandidato(self, candidatos):
        ys = []
        for c in candidatos:
            x, y, w, h = cv2.boundingRect(c)
            ys.append(y)
        return candidatos[np.argmax(ys)]
    
    def mostrarMenorCandidato(self, img, candidato):
        canvas = np.zeros_like(img)
        cv2.drawContours(canvas, [candidato], -1, (0, 255, 0), 2)
        plt.axis('off')
        plt.imshow(canvas)
        plt.show()

    def recortarMatricula(self, img, candidato):
        x, y, w, h = cv2.boundingRect(candidato)
        return img[y:y+h,x:x+w]
    
    def eliminarBordes(self, img):
        return skimage.segmentation.clear_border(img)
    
    def invertirImagen(self, img):
        return cv2.bitwise_not(img)
    
    def hallarMatricula(self, img):
        alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        options = "-c tessedit_char_whitelist={}".format(alphanumeric)
        options += " -- psm 7"
        txt = pytesseract.image_to_string(img, config=options)
        return txt[1:]