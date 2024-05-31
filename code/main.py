import numpy as np #pip install numpy
import cv2 #pip install opencv-python
import matplotlib.pyplot as plt #pip install matplotlib
import pytesseract #pip install pytesseract
import skimage #python -m pip install scikit-image
import DetectorMatriculas as DM

dm = DM.DetectorMatriculas()
img = img = cv2.imread("./dataset/coche1.png")
print(img.shape)
dm.mostrarImagen(img, "Imagen original",)

dm.mostrarRGB(img)

imgBW = dm.toEscalaDeGrises(img)
dm.mostrarImagen(imgBW, "Escala de grises")

imgBIN = dm.aplicarUmbrealizacionAdaptativa(imgBW)
dm.mostrarImagen(imgBIN, "Umbrealizacion")

contornos = dm.encontrarContornos(imgBIN)
dm.mostrarContornos(img, contornos)

candidatos = dm.filtrarCandidatos(contornos)
dm.mostrarCandidatos(img, candidatos)

candidatoMenor = dm.filtrarMenorCandidato(candidatos)
dm.mostrarMenorCandidato(img, candidatoMenor)