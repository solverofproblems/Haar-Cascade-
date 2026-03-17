import cv2
import os
import sys
import time

def _try_load_cascade(path: str):
    """Tenta criar um CascadeClassifier para o caminho informado.

    Retorna o CascadeClassifier carregado com sucesso ou None caso falhe.
    """
    if not os.path.isfile(path):
        return None

    cc = cv2.CascadeClassifier(path)
    return cc if not cc.empty() else None


def _short_path(path: str) -> str:
    """Retorna o caminho curto (8.3) no Windows para evitar problemas com caracteres.

    Alguns builds do OpenCV não lidam bem com caminhos que contêm acentos ou caracteres Unicode.
    """
    if os.name != 'nt':
        return path

    try:
        import ctypes

        buf = ctypes.create_unicode_buffer(260)
        if ctypes.windll.kernel32.GetShortPathNameW(path, buf, len(buf)):
            return buf.value
    except Exception:
        pass

    return path


# Tenta carregar o classificador Haar Cascade do OpenCV e de caminhos alternativos.
cascade_names = ['haarcascade_frontalface_default.xml']
primary_dir = cv2.data.haarcascades

cascade_paths = []
for name in cascade_names:
    cascade_paths.append(os.path.join(primary_dir, name))
    cascade_paths.append(_short_path(os.path.join(primary_dir, name)))
    cascade_paths.append(os.path.join(os.path.dirname(__file__), name))

face_cascade = None
loaded_path = None
for p in cascade_paths:
    face_cascade = _try_load_cascade(p)
    if face_cascade is not None:
        loaded_path = p
        break

if face_cascade is None:
    tried = '\n'.join(f" - {p!r}" for p in cascade_paths)
    sys.exit(
        "Erro: não foi possível carregar o classificador Haar Cascade.\n"
        "Verifique se o arquivo existe em um destes caminhos (e se o OpenCV consegue lê-lo):\n"
        f"{tried}\n\n"
        "Dica: caminhos com acentos ou caracteres Unicode podem causar esse problema no Windows.\n"
        "Considere mover o projeto para uma pasta sem acentos."
    )

print(f"Classificador carregado com sucesso (usando: {loaded_path})")

# Aqui estamos carregando a imagem desejada. Note que você pode alterá-la se quiser.
imagem = cv2.imread('alanturing.jpg')
if imagem is None:
    sys.exit(
        "Erro: não foi possível carregar a imagem 'alanturing.jpg'.\n"
        "Verifique se o arquivo existe no diretório atual."
    )

# Aqui transformamos a imagem para escala cinza, facilitando a análise do classificador.
# Lembre-se que o classificador foi treinado para trabalhar com imagens nessa escala, logo, é boa prática preservá-la assim.
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
