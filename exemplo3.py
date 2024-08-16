import numpy
from mpi4py import MPI

#função soma
def soma(v, i , f):
    resultadoSoma = numpy.zeros(1)
    for x in range(i,f):
        resultadoSoma = resultadoSoma + v[x]
    return resultadoSoma

#criando o vetor
tamvetor = 10000
vetor = numpy.arange(tamvetor) # cada processo cria na sua memoria reservada essas posições, sem troca de msgs

#criando o comunicador do MPI
comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

qtp = tamvetor//nprocs
inicio = qtp * rank
fim = qtp * (rank+1)

rparcial = soma(vetor, inicio, fim) #cada um possui a sua soma

#print("Resultado: ", rparcial)

total = numpy.zeros(1)
comm.Reduce(rparcial, total, op=MPI.SUM, root = 0)

if rank == 0:
    print(total, rank)