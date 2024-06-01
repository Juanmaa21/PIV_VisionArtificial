import cv2
import DetectorMatriculas as DM
import time

def compararSimilitud(s, t):
    nFallos = 0
    for i in range(len(s)):
        if s[i] != t[i] :
            nFallos += 1
    return 1-(nFallos/(len(s) +.0))

dm = DM.DetectorMatriculas()

matriculas = ["8735 BBT", "7206 KDF", "5683 JZG", "4386 LJP", "4991 KXN",
              "9723 LCP", "8586 KBY", "8206 MCS", "0378 LKF", "0798 MNC"]

nAciertos = 0
porcentajeError = 0

print("ORIGINAL ----  RESULT  ----- SIMILITUD(%)")
for i in range(1,11):
    img = cv2.imread(f"./dataset/coche{i}.png")
    similitud = compararSimilitud(matriculas[i-1],dm.hallarMatricula(img))
    if similitud == 1:
        nAciertos += 1
    porcentajeError += 1-similitud

    print(matriculas[i-1] + " ---- " + dm.hallarMatricula(img) + " ----- " + str(similitud*100) + "%")

print()
print("Coches totales:               " + str(len(matriculas)))
print("Numero de aciertos:           " + str(nAciertos))
print("Numero de fallos:             " + str(len(matriculas)-nAciertos))
print("Porcentaje simbolos erroneos: " + str((porcentajeError/len(matriculas))*100) + "%")