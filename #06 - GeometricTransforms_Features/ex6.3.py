import argparse
import os
import sys

import cv2
import numpy as np


def make_transformed(img):
    rows, cols = img.shape[:2]
    # Rotação em torno do centro e pequena translação
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 25, 1.0)
    M[0, 2] += -50
    M[1, 2] += 30
    transformed = cv2.warpAffine(img, M, (cols, rows))
    return transformed


def create_sift():
    try:
        sift = cv2.SIFT_create()
    except AttributeError:
        # Fallback for older OpenCV builds
        try:
            sift = cv2.xfeatures2d.SIFT_create()
        except Exception:
            print("SIFT não disponível. Instale opencv-contrib-python.")
            sys.exit(1)
    return sift


def draw_and_save(img, keypoints, out_path):
    drawn = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(out_path, drawn)
    return drawn


def main():
    p = argparse.ArgumentParser(description="Ex6.3 - SIFT keypoints on original and transformed image")
    p.add_argument("--image", "-i", required=True, help="Caminho para a imagem de entrada")
    args = p.parse_args()

    if not os.path.exists(args.image):
        print(f"Arquivo não encontrado: {args.image}")
        sys.exit(1)

    img = cv2.imread(args.image)
    if img is None:
        print("Não foi possível ler a imagem. Verifique o formato/arquivo.")
        sys.exit(1)

    transformed = make_transformed(img)

    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)

    sift = create_sift()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    base, ext = os.path.splitext(os.path.basename(args.image))
    out1 = f"{base}_kp{ext}"
    out2 = f"{base}_tf_kp{ext}"

    drawn1 = draw_and_save(img, kp1, out1)
    drawn2 = draw_and_save(transformed, kp2, out2)

    print(f"Keypoints (original): {len(kp1)} -> {out1}")
    print(f"Keypoints (transformada): {len(kp2)} -> {out2}")

    cv2.imshow("Original - keypoints", drawn1)
    cv2.imshow("Transformada - keypoints", drawn2)
    print("Pressione qualquer tecla na janela para sair...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
