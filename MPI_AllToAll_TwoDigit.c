#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int main(void)
{
    int rank, size;

    MPI_Init(NULL, NULL);                   // Initialize the MPI environment.
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // Get this process's rank (ID).
    MPI_Comm_size(MPI_COMM_WORLD, &size);   // Get the total number of processes.

    if (size > 10) {                        // At most 10 processes are allowed.
        if (rank == 0) {
            fprintf(stderr,
                    "ERROR: This task requires size <= 10 so that each rank fits into one decimal digit.\n"
                    "You started %d processes.\n",
                    size);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);       // Abort all processes.
    }

    // Set a different random seed for each process.
    srand((unsigned)time(NULL) ^ (unsigned)(rank * 2654435761u));

    int *sendbuf = (int *)malloc((size_t)size * sizeof(int));  // Allocate the send buffer.
    int *recvbuf = (int *)malloc((size_t)size * sizeof(int));  // Allocate the receive buffer.
    if (!sendbuf || !recvbuf) {
        fprintf(stderr, "Rank %d: malloc failed\n", rank);     // Check the allocation.
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    for (int dest = 0; dest < size; ++dest) {                  // Populate the send buffer.
        if (dest == rank) {
            sendbuf[dest] = -1;                                // Do not send to itself.
        } else {
            int tens = rank;                                   // Tens digit = process rank.
            int ones = rand() % 10;                            // Ones digit = random value from 0 to 9.
            sendbuf[dest] = tens * 10 + ones;                  // Form the two-digit number.
        }
    }

    // Each process sends one int to every other process.
    MPI_Alltoall(sendbuf, 1, MPI_INT, recvbuf, 1, MPI_INT, MPI_COMM_WORLD);

    for (int r = 0; r < size; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);                           // Synchronize output.
        if (r == rank) {
            printf("Process %d received:", rank);              // Print the received values.
            for (int src = 0; src < size; ++src) {
                if (src == rank) continue;                     // Skip this process's own value.
                printf(" %d", recvbuf[src]);
            }
            printf("\n");
            fflush(stdout);                                    // Flush standard output.
        }
    }

    free(sendbuf);                                             // Free allocated memory.
    free(recvbuf);

    MPI_Finalize();                                            // Shut down the MPI environment.
    return 0;
}
