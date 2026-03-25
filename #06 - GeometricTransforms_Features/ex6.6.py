"""
ex6.6.py

Permite selecionar manualmente os 4 cantos de um objeto retangular (ex: livro)
e corrige a homografia para gerar uma visão frontal (retificação).

Uso:
    python ex6.6.py -i path/to/image.jpg

Clique nos 4 cantos na ordem: top-left, top-right, bottom-right, bottom-left.
Pressione 'r' para reiniciar a seleção ou qualquer tecla na janela resultante para sair.
"""
import argparse
import os
import sys

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Ex6.6 - Homography manual selection and rectification")
    p.add_argument("-i", "--image", required=True, help="Caminho para a imagem de entrada")
    p.add_argument("--out_width", type=int, default=700, help="Largura (px) da imagem retificada (default 700)")
    return p.parse_args()


def make_transformed(img):
    # Gera uma versão transformada para testar (igual aos exercícios anteriores)
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 25, 1.0)
    M[0, 2] += -50
    M[1, 2] += 30
    return cv2.warpAffine(img, M, (cols, rows))


def main():
    args = parse_args()
    if not os.path.exists(args.image):
        print(f"Arquivo não encontrado: {args.image}")
        sys.exit(1)

    img = cv2.imread(args.image)
    if img is None:
        print("Não foi possível ler a imagem.")
        sys.exit(1)

    # Usamos a versão transformada (simula já ter uma perspectiva)
    src_img = make_transformed(img)
    clone = src_img.copy()

    pts = []


    def on_mouse(event, x, y, flags, param):
        nonlocal pts, clone
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(clone, str(len(pts)), (x + 8, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Selecione 4 cantos", clone)


    cv2.namedWindow("Selecione 4 cantos")
    cv2.setMouseCallback("Selecione 4 cantos", on_mouse)

    print("Clique nos 4 cantos na ordem: top-left, top-right, bottom-right, bottom-left")
    print("Pressione 'r' para reiniciar a seleção")

    while True:
        cv2.imshow("Selecione 4 cantos", clone)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            pts = []
            clone = src_img.copy()
            cv2.imshow("Selecione 4 cantos", clone)
            print("Seleção reiniciada")
        if len(pts) == 4:
            break

    cv2.destroyWindow("Selecione 4 cantos")

    src_pts = np.array(pts, dtype=np.float32)

    # Calcula tamanho de saída mantendo a razão real do livro 17.5 x 23.5 cm
    aspect = 23.5 / 17.5
    w = args.out_width
    h = int(round(w * aspect))

    dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(src_img, H, (w, h))

    base, ext = os.path.splitext(os.path.basename(args.image))
    out_warp = f"{base}_homography_warp{ext}"
    cv2.imwrite(out_warp, warped)

    print(f"Homography aplicada. Saída: {out_warp} (size {w}x{h})")

    cv2.imshow("Retificado", warped)
    print("Pressione qualquer tecla na janela para sair...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
