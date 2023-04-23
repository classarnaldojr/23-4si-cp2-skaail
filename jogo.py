import cv2
from matplotlib import pyplot as plt
import numpy as np


cap = cv2.VideoCapture('pedra-papel-tesoura.mp4')

placar_1 = 0
placar_2 = 0

pedra_template_1 = cv2.imread('pedra_1.png', 0)
papel_template_1 = cv2.imread('papel_1.png', 0)
tesoura_template_1 = cv2.imread('tesoura_1.png', 0)

pedra_template_2 = cv2.imread('pedra_2.png', 0)
papel_template_2 = cv2.imread('papel_2.png', 0)
tesoura_template_2 = cv2.imread('tesoura_2.png', 0)

def detect_jogada_2(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    res_pedra = cv2.matchTemplate(gray, pedra_template_2, cv2.TM_CCOEFF_NORMED)
    res_papel = cv2.matchTemplate(gray, papel_template_2, cv2.TM_CCOEFF_NORMED)
    res_tesoura = cv2.matchTemplate(gray, tesoura_template_2, cv2.TM_CCOEFF_NORMED)

    max_pedra = cv2.minMaxLoc(res_pedra)[1]
    max_papel = cv2.minMaxLoc(res_papel)[1]
    max_tesoura = cv2.minMaxLoc(res_tesoura)[1]

    vars_dict = {'jogador 2 pedra': max_pedra, 'jogador 2 papel': max_papel, 'jogador 2 tesoura': max_tesoura}
    max_var = max(vars_dict, key=vars_dict.get)
    return max_var


def detect_jogada(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    res_pedra = cv2.matchTemplate(gray, pedra_template_1, cv2.TM_CCOEFF_NORMED)
    res_papel = cv2.matchTemplate(gray, papel_template_1, cv2.TM_CCOEFF_NORMED)
    res_tesoura = cv2.matchTemplate(gray, tesoura_template_1, cv2.TM_CCOEFF_NORMED)

    max_pedra = cv2.minMaxLoc(res_pedra)[1]
    max_papel = cv2.minMaxLoc(res_papel)[1]
    max_tesoura = cv2.minMaxLoc(res_tesoura)[1]

    vars_dict = {'jogador 1 pedra': max_pedra, 'jogador 1 papel': max_papel, 'jogador 1 tesoura': max_tesoura}
    max_var = max(vars_dict, key=vars_dict.get)
    return max_var

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (700, 500))

    jogada_1 = detect_jogada(frame)
    jogada_2 = detect_jogada_2(frame)
    
    if jogada_1 == 'jogador 1 papel' and jogada_2 == 'jogador 2 pedra':
        placar_1 += 1

    if jogada_1 is not None:
        cv2.putText(frame, jogada_1, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    
    if jogada_2 is not None:
        cv2.putText(frame, jogada_2, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    print(placar_1, placar_2)

    cv2.imshow('frame', frame)

    cv2.waitKey(25)

cap.release()
cv2.destroyAllWindows()
