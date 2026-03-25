import argparse
import os
import sys

import cv2
import numpy as np


def make_transformed(img):
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 25, 1.0)
    M[0, 2] += -50
    M[1, 2] += 30
    return cv2.warpAffine(img, M, (cols, rows))


def create_sift():
    try:
        return cv2.SIFT_create()
    except AttributeError:
        try:
            return cv2.xfeatures2d.SIFT_create()
        except Exception:
            print("SIFT não disponível. Instale opencv-contrib-python.")
            sys.exit(1)


def match_descriptors(des1, des2, cross_check=True):
    # SIFT usa descritores float -> NORM_L2
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=cross_check)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def draw_and_save_matches(img1, kp1, img2, kp2, matches, out_path):
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(out_path, img_matches)
    return img_matches


def parse_args():
    p = argparse.ArgumentParser(description="Ex6.4 - Correspondências com BFMatcher")
    p.add_argument("-i", "--image", required=True, help="Caminho para a imagem de entrada")
    p.add_argument("--num_matches", type=int, default=None, help="Número fixo de matches a manter")
    p.add_argument("--percent", type=float, default=0.1, help="Percentual das melhores matches (default 0.1)")
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.image):
        print(f"Arquivo não encontrado: {args.image}")
        sys.exit(1)

    img = cv2.imread(args.image)
    if img is None:
        print("Não foi possível ler a imagem.")
        sys.exit(1)

    img_t = make_transformed(img)

    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img_t, cv2.COLOR_BGR2GRAY)

    sift = create_sift()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        print("Descritores não encontrados em uma das imagens.")
        sys.exit(1)

    matches = match_descriptors(des1, des2, cross_check=True)
    if len(matches) == 0:
        print("Nenhuma match encontrada.")
        sys.exit(1)

    # Seleção de matches
    if args.num_matches is not None:
        num = min(args.num_matches, len(matches))
    else:
        num = max(1, int(len(matches) * args.percent))

    good_matches = matches[:num]

    base, ext = os.path.splitext(os.path.basename(args.image))
    out_matches = f"{base}_matches{ext}"

    img_matches = draw_and_save_matches(img, kp1, img_t, kp2, good_matches, out_matches)

    print(f"Total matches encontradas: {len(matches)}")
    print(f"Usando matches: {len(good_matches)} -> {out_matches}")

    cv2.imshow("Matches", img_matches)
    print("Pressione qualquer tecla na janela para sair...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
