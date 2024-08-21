#include <stdio.h>
#include <mpi.h>
#include <mpi-ext.h> /* Needed for CUDA-aware check */
#include <stdbool.h>


int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    bool happy = false;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    happy = (bool) MPIX_Query_cuda_support();
#endif

    if (happy) {
        printf("This Open MPI installation has CUDA-aware support.\n");
    } else {
        printf("This Open MPI installation does not have CUDA-aware support.\n");
    }

    MPI_Finalize();
    return 0;
}
