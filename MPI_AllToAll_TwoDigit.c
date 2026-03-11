#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int main(void)
{
    int rank, size;

    MPI_Init(NULL, NULL);                   // Inicijalizacija MPI okruzenja
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // Uzimanje ranga (ID procesa)
    MPI_Comm_size(MPI_COMM_WORLD, &size);   // Ukupan broj procesa

    if (size > 10) {                        // Dozvoljeno je najvise 10 procesa
        if (rank == 0) {
            fprintf(stderr,
                    "ERROR: This task requires size <= 10 so that each rank fits into one decimal digit.\n"
                    "You started %d processes.\n",
                    size);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);       // Zaustavljanje svih procesa
    }

    // Postavljanje razlicitog seed-a za svaki proces
    srand((unsigned)time(NULL) ^ (unsigned)(rank * 2654435761u));

    int *sendbuf = (int *)malloc((size_t)size * sizeof(int));  // Alokacija send buffera
    int *recvbuf = (int *)malloc((size_t)size * sizeof(int));  // Alokacija recv buffera
    if (!sendbuf || !recvbuf) {
        fprintf(stderr, "Rank %d: malloc failed\n", rank);     // Provera alokacije
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    for (int dest = 0; dest < size; ++dest) {                  // Popunjavanje send buffera
        if (dest == rank) {
            sendbuf[dest] = -1;                                // Ne salje se sebi
        } else {
            int tens = rank;                                   // Desetice = rank procesa
            int ones = rand() % 10;                            // Jedinice = nasumican broj 0–9
            sendbuf[dest] = tens * 10 + ones;                  // Formiranje broja XY
        }
    }

    // Svaki proces salje jedan int svakom drugom procesu
    MPI_Alltoall(sendbuf, 1, MPI_INT, recvbuf, 1, MPI_INT, MPI_COMM_WORLD);

    for (int r = 0; r < size; ++r) {
        MPI_Barrier(MPI_COMM_WORLD);                           // Sinhronizacija ispisa
        if (r == rank) {
            printf("Process %d received:", rank);              // Ispis primljenih vrednosti
            for (int src = 0; src < size; ++src) {
                if (src == rank) continue;                     // Preskace se sopstvena vrednost
                printf(" %d", recvbuf[src]);
            }
            printf("\n");
            fflush(stdout);                                    // Forsiranje ispisa
        }
    }

    free(sendbuf);                                             // Oslobadjanje memorije
    free(recvbuf);

    MPI_Finalize();                                            // Zatvaranje MPI okruzenja
    return 0;
}
