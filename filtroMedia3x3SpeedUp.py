import cv2
import numpy as np
from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
nproces = comm.Get_size()
myrank = comm.Get_rank()

localimage = None
newimage = None
image = None
n = 0
largura = 0

if myrank == 0:
    image = cv2.imread("maspcomruido.jpg", 0)
    if image is None:
        raise FileNotFoundError("Não foi possível ler a imagem.")
    altura, largura = image.shape
    n = int(altura / nproces)
    newimage = np.zeros((altura, largura), dtype='uint8')

(n, largura) = comm.bcast((n, largura), root=0)
localimage = np.zeros((n, largura), dtype='uint8')

# Dividindo a imagem entre os processos
counts = [n * largura for _ in range(nproces)]
displs = [i * n * largura for i in range(nproces)]

# Início da contagem do MPI
start_time = MPI.Wtime()

# Distribuindo entre os processos
comm.Scatterv([image, counts, displs, MPI.UINT8_T], localimage, root=0)

# Convolução aplicada localmente em cada processo
localimage = cv2.blur(localimage, (3, 3))

# Recolhendo
comm.Gatherv(localimage, [newimage, counts, displs, MPI.UINT8_T], root=0)

# Fim da contagem do MPI
end_time = MPI.Wtime()
parallel_duration = end_time - start_time

# Tempo paralelo máximo entre os processos
max_parallel_duration = comm.reduce(parallel_duration, op=MPI.MAX, root=0)

if myrank == 0:
    serial_start_time = time.time()
    serial_image = cv2.blur(image, (3, 3))
    serial_end_time = time.time()
    serial_duration = serial_end_time - serial_start_time

    # Calcular speedup
    speedup = serial_duration / max_parallel_duration

    print(f"Tempo de execução em série: {serial_duration:.4f} segundos")
    print(f"Tempo de execução em paralelo: {max_parallel_duration:.4f} segundos")
    print(f"Speedup: {speedup:.2f}")
