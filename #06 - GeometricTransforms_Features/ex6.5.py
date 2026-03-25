"""
ex6.5.py

Avalia a transformação automaticamente usando correspondências SIFT + BFMatcher
e estima uma matriz afim com RANSAC. Salva a imagem warpeada e a diferença.

Uso:
    python ex6.5.py -i path/to/image.jpg [--num_matches N] [--percent P] [--no-ransac]

"""
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
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=cross_check)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def parse_args():
    p = argparse.ArgumentParser(description="Ex6.5 - Avaliação automática da transformação")
    p.add_argument("-i", "--image", required=True, help="Caminho para a imagem de entrada")
    p.add_argument("--num_matches", type=int, default=None, help="Número fixo de matches a usar")
    p.add_argument("--percent", type=float, default=0.1, help="Percentual das melhores matches (default 0.1)")
    p.add_argument("--no-ransac", action="store_true", help="Desativa RANSAC e usa solução direta quando possível")
    return p.parse_args()


def compute_transform(src_pts, dst_pts, use_ransac=True, ransac_thresh=3.0):
    # src_pts, dst_pts devem ter shape (N,1,2)
    src = src_pts.reshape(-1, 2)
    dst = dst_pts.reshape(-1, 2)
    if use_ransac and len(src) >= 3:
        M, inliers = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
        return M, inliers
    # fallback: if exactly 3 pontos, usar getAffineTransform
    if len(src) >= 3:
        M = cv2.getAffineTransform(src[:3].astype(np.float32), dst[:3].astype(np.float32))
        return M, None
    return None, None


def decompose_affine(M):
    a, c, tx = M[0]
    b, d, ty = M[1]
    sx = np.sign(a) * np.sqrt(a * a + b * b)
    sy = np.sign(d) * np.sqrt(c * c + d * d)
    psi = np.degrees(np.arctan2(b, a))
    return {"a": a, "b": b, "c": c, "d": d, "tx": tx, "ty": ty, "sx": sx, "sy": sy, "psi_deg": psi}


def draw_and_save_matches(img1, kp1, img2, kp2, matches, out_path):
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(out_path, img_matches)
    return img_matches


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

    if args.num_matches is not None:
        num = min(args.num_matches, len(matches))
    else:
        num = max(1, int(len(matches) * args.percent))

    good_matches = matches[:num]

    # converter para np array conforme README
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, inliers = compute_transform(src_pts, dst_pts, use_ransac=not args.no_ransac)
    if M is None:
        print("Não foi possível estimar a transformação (poucos pontos).")
        sys.exit(1)

    base, ext = os.path.splitext(os.path.basename(args.image))
    out_matches = f"{base}_matches{ext}"
    out_warp = f"{base}_auto_warp{ext}"
    out_diff = f"{base}_warp_diff{ext}"

    draw_and_save_matches(img, kp1, img_t, kp2, good_matches, out_matches)

    warped = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    cv2.imwrite(out_warp, warped)

    diff = cv2.absdiff(img_t, warped)
    cv2.imwrite(out_diff, diff)

    params = decompose_affine(M)
    print("Matriz estimada (2x3):")
    print(M)
    print(f"Parâmetros: tx={params['tx']:.2f}, ty={params['ty']:.2f}, psi={params['psi_deg']:.2f} deg, sx={params['sx']:.3f}, sy={params['sy']:.3f}")
    if inliers is not None:
        print(f"Inliers RANSAC: {int(np.sum(inliers))} / {len(inliers)}")

    print(f"Saídas: {out_matches}, {out_warp}, {out_diff}")

    # Mostrar janelas
    win_matches = cv2.imread(out_matches)
    cv2.imshow("Matches", win_matches)
    cv2.imshow("Warped (estimado)", warped)
    cv2.imshow("Diff (transf - warped)", diff)
    print("Pressione qualquer tecla na janela para sair...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
