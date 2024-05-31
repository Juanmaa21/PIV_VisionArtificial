import numpy as np #pip install numpy
import cv2 #pip install opencv-python
import matplotlib.pyplot as plt #pip install matplotlib
import pytesseract #pip install pytesseract
import skimage #python -m pip install scikit-image
import DetectorMatriculas as DM

dm = DM.DetectorMatriculas()
img = img = cv2.imread("./dataset/coche1.png")
print(img.shape)
dm.mostrarImagen(img)

#dm.mostrarRGB(img)

imgBW = dm.toEscalaDeGrises(img)
#dm.mostrarImagen(imgBW)

imgBIN = dm.aplicarUmbrealizacionAdaptativa(imgBW)
#dm.mostrarImagen(imgBIN)

contornos = dm.encontrarContornos(imgBIN)
#dm.mostrarContornos(img, contornos)

candidatos = dm.filtrarCandidatos(contornos)
#dm.mostrarCandidatos(img, candidatos)

candidatoMenor = dm.filtrarMenorCandidato(candidatos)
#dm.mostrarMenorCandidato(img, candidatoMenor)

matricula = dm.recortarMatricula(img, candidatoMenor)
dm.mostrarImagen(matricula)

matriculaBIN = dm.aplicarUmbrealizacionAdaptativa(dm.toEscalaDeGrises(matricula))
dm.mostrarImagen(matriculaBIN)

matriculaBINSinBordes = dm.eliminarBordes(matriculaBIN)
dm.mostrarImagen(matriculaBINSinBordes)

matriculaFinal = dm.invertirImagen(matriculaBINSinBordes)
dm.mostrarImagen(matriculaFinal)

print(dm.hallarMatricula(matriculaFinal))