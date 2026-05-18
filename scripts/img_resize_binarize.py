import cv2

img = cv2.imread("../figures/originals/neko.jpeg", 0)

ret2, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

cv2.imwrite("../figures/edited/neko_edited.jpeg", img)
cv2.imshow("otsu", img)
cv2.waitKey()
cv2.destroyAllWindows()