import cv2
import time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if face_cascade.empty():
    print("Erro ao carregar o classificador.")
else:
    print("Classificador carregado com sucesso.")

imagem = cv2.imread('alanturing.jpg')

img_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(

    img_cinza, 
    scaleFactor=1.1, 
    minNeighbors=5,
    minSize=(30, 30)

)

for (x, y, w, h) in faces:
    cv2.rectangle(imagem, (x,y), (x + w, y + h), (255, 0, 0), 2)

print('O contorno da região que o modelo detecta como "Rosto" será incrementado na imagem.')
time.sleep(2)
cv2.imshow('Detecção', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
