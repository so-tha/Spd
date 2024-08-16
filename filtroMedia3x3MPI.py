import cv2
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
nproces = comm.Get_size()
myrank = comm.Get_rank()

localimage = None
newimage = None
image = None
n = 0
largura = 0

if myrank == 0:
    image = cv2.imread("imageruido.jpg", 0)
    altura, largura = image.shape
    n = int(altura/largura)
    newimage = np.zeros((altura,largura), dtype='uint8')


(n,largura) = comm.bcast((n,largura), root = 0)
localimage = np.zeros((n,largura), dtype ='uint8')

#Distribuindo entre os processos
comm.Scatterv(image, localimage, root = 0)
cv2.imshow("corte", localimage) # imagens partidas


#Recolhendo entre os processos
comm.Gatherv(localimage, newimage, root = 0)

if myrank == 0:
    cv2.imshow("original image", image)
    cv2.imshow("Filtred image", newimage)
    cv2.waitKey(0)

cv2.destroyAllWindows()