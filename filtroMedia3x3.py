import cv2
import numpy as np

image = cv2.imread("imageruido.jpg", 0)

altura, largura = image.shape
newimage = np.zeros((altura,largura), dtype='uint8')

#flitro media 3x3
for y in range(1, altura -1):
    for x in range(1, largura -1):
        media = int(image[y-1][x-1]) + int(image[y - 1][x]) + int(image[y-1][x+1]) #1 linha
        media += int(image[y][x+1]) + int(image[y][x]) + int(image[y][x + 1]) # 2 linha
        media += int(image[y + 1][x  - 1]) + int(image[y + 1][x]) + int(image[y + 1][x + 1]) # 3 linha

        media = int(media/9)

        newimage[y][x] = media

cv2.imshow("original image", image)
cv2.imshow("Filtred image", newimage)

cv2.waitKey(0)
cv2.destroyAllWindows()