from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

tamanho = comm.Get_size()

if rank == 0:
    print("Sou o processo inicial")
    print("Informo que há", tamanho, "processos em execução")
else:
    print("Demais processos: ",rank)
