import cv2

image = cv2.imread('../images/lena.jpg', cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image not found")
    exit(-1)

def select_src(event, x, y, flags, params):
    global  srcPts
    if event == cv2.EVENT_LBUTTONDOWN:
        srcPts.append((x,y))
        cv2.circle(src, (x, y), 2, (255, 0, 0), 2)
        cv2.putText(src,str(len(srcPts)), (x+10,y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
        cv2.imshow("orginal", src)



cv2.imshow("Original", image)