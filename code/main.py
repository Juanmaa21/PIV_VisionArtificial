import cv2 #pip install opencv-python
import DetectorMatriculas as DM

dm = DM.DetectorMatriculas()
# coches: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
# coches cuya matricula coincide y solo 1 forma: 1, 2
# coches cuya matricula coincide y mas de 1 forma: 6, 7, 8, 10
# coches cuya matricula casi coincide y solo 1 forma:
# coches cuya matricula casi coincide y mas de 1 forma: 3, 4, 5, 9
img = img = cv2.imread("./dataset/coche7.png")
print(img.shape)
dm.mostrarImagen(img)

dm.mostrarRGB(img)

imgBW = dm.toEscalaDeGrises(img)
dm.mostrarImagen(imgBW)

imgBIN = dm.aplicarUmbrealizacionAdaptativa(imgBW)
dm.mostrarImagen(imgBIN)

contornos = dm.encontrarContornos(imgBIN)
dm.mostrarContornos(img, contornos)

candidatos = dm.filtrarCandidatos(contornos)
dm.mostrarCandidatos(img, candidatos)

candidatoMenor = dm.filtrarMenorCandidato(candidatos)
dm.mostrarMenorCandidato(img, candidatoMenor)

matricula = dm.recortarMatricula(img, candidatoMenor)
dm.mostrarImagen(matricula)

matriculaBIN = dm.aplicarUmbrealizacionAdaptativa(dm.toEscalaDeGrises(matricula))
dm.mostrarImagen(matriculaBIN)

matriculaBINSinBordes = dm.eliminarBordes(matriculaBIN)
dm.mostrarImagen(matriculaBINSinBordes)

matriculaFinal = dm.invertirImagen(matriculaBINSinBordes)
dm.mostrarImagen(matriculaFinal)

print(dm.hallarNumerosYLetras(matriculaFinal))