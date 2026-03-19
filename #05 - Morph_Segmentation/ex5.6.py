import cv2
import numpy as np
import os

# Images to test (relative to this script)
IMAGE_LIST = [
    os.path.join('..', 'images', 'wdg2.bmp'),
    os.path.join('..', 'images', 'tools_2.png'),
    os.path.join('..', 'images', 'lena.jpg'),
]

current_idx = 0
img = None

def load_image(idx):
    path = IMAGE_LIST[idx]
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f'Erro no caminho da imagem: {path}')
        return None
    return img

def on_mouse(event, x, y, flags, param):
    global img
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    h, w = img.shape[:2]
    if x < 0 or x >= w or y < 0 or y >= h:
        return

    seed_point = (x, y)
    # red fill for visibility
    newVal = (0, 0, 255)
    # allow ±5 intensity per channel
    loDiff = (5, 5, 5)
    upDiff = (5, 5, 5)
    flags = 8 | cv2.FLOODFILL_FIXED_RANGE

    mask = np.zeros((h + 2, w + 2), np.uint8)
    img_copy = img.copy()
    retval, img_out, mask_out, rect = cv2.floodFill(img_copy, mask, seed_point, newVal, loDiff, upDiff, flags)
    print('Seed:', seed_point, 'Filled pixels:', retval, 'Bounding rect:', rect)

    # show marker on original and the filled result
    orig_marker = img.copy()
    cv2.drawMarker(orig_marker, seed_point, (0,255,0), markerType=cv2.MARKER_CROSS, thickness=2)
    cv2.imshow('Original (click to seed)', orig_marker)
    cv2.imshow('FloodFilled', img_out)


def main():
    global img, current_idx
    img = load_image(current_idx)
    if img is None:
        return

    cv2.namedWindow('Original (click to seed)')
    cv2.setMouseCallback('Original (click to seed)', on_mouse)

    # Initial display
    cv2.imshow('Original (click to seed)', img)
    print('Instructions: Left-click to choose seed. Keys: n=next, p=prev, r=reset, q=quit')

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            current_idx = (current_idx + 1) % len(IMAGE_LIST)
            img = load_image(current_idx)
            if img is None:
                break
            cv2.imshow('Original (click to seed)', img)
            cv2.destroyWindow('FloodFilled') if cv2.getWindowProperty('FloodFilled', cv2.WND_PROP_VISIBLE) >= 0 else None
            print('Switched to:', IMAGE_LIST[current_idx])
        elif key == ord('p'):
            current_idx = (current_idx - 1) % len(IMAGE_LIST)
            img = load_image(current_idx)
            if img is None:
                break
            cv2.imshow('Original (click to seed)', img)
            cv2.destroyWindow('FloodFilled') if cv2.getWindowProperty('FloodFilled', cv2.WND_PROP_VISIBLE) >= 0 else None
            print('Switched to:', IMAGE_LIST[current_idx])
        elif key == ord('r'):
            # reset display of current image
            img = load_image(current_idx)
            if img is None:
                break
            cv2.imshow('Original (click to seed)', img)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()