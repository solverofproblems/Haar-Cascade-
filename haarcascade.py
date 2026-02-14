import cv2
import time

face_cascade = cv2.CascadeClassifier(
    # É importante que você mantenha exatamente dessa forma!
    # Quando instalamos a biblioteca OpenCV, ela já vem com esse classificador pré-treinado.
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

)

#Essa parte você pode remover se desejar, mas recomendo por uma questão de segurança.
if face_cascade.empty():
    print("Erro ao carregar o classificador.")
else:
    print("Classificador carregado com sucesso.")

#Aqui estamos carregando a imagem desejada. Note que você pode alterá-la se quiser.
imagem = cv2.imread('alanturing.jpg')

#Aqui transformamos a imagem para escala cinza, facilitando a análise do classificador.
#Lembre-se que o classificador foi treinado para trabalhar com imagens nessa escala, logo, é boa prática preservá-la assim.
img_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)


#Aqui são configurações gerais sobre a forma como o modelo classificatório deve operar, além dos cuidados que ele deve ter.
faces = face_cascade.detectMultiScale(

    img_cinza, #Aqui estamos simplesmente carregando a imagem que desejamos.

    scaleFactor=1.1, #Aqui estamos definindo o quanto a imagem será reduzida conforme o andamento dos estágios.

    minNeighbors=5, #Controla o rigor de detecção. O modelo encontra diversos pontos que se parecem com um rosto e, no final, agrupam-se para formar o retângulo que contorna aquilo que o modelo classifica como "Rosto".

    minSize=(30, 30) #Aqui é definido o tamanho mínimo que o determinado rosto deve conter. Se o rosto da imagem for menor que o que está sendo definido, o classificador simplesmente não consegue identificá-lo. Logo, é importante saber se as imagens que você deseja avaliar possuem um rosto em grande ou pequena escala.

)

#Aqui você consegue construir o retângulo que contorna a região que o modelo detecta como rosto. Você pode mudar a cor e grossura.

for (x, y, w, h) in faces:
    cv2.rectangle(imagem, (x,y), (x + w, y + h), (0, 0, 255), 2)

print('O contorno da região que o modelo detecta como "Rosto" será incrementado na imagem.')
time.sleep(2)
cv2.imshow('Detecção', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
