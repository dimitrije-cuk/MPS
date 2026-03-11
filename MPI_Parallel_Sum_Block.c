#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;
    long long N = 0;    // Gornja granica sume: 1 + 2 + ... + N

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Ulaz: proces 0 određuje vrednost N
    if (rank == 0) {
        printf("Enter N (non-negative integer): ");
        fflush(stdout);

        if (scanf("%lld", &N) != 1 || N < 0) {
            fprintf(stderr,
                    "Invalid input. N must be a non-negative integer.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Distribucija vrednosti N ka svim procesima
    MPI_Bcast(&N, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    // Izračunavanje ravnomerne raspodele posla
    long long q = N / size;
    long long r = N % size;

    // Prvih r procesa dobija po (q+1) elemenata, ostali dobijaju po q elemenata
    long long local_count = (rank < r) ? (q + 1) : q;

    // Izračunavanje prefiksa — koliko elemenata dolazi pre ovog procesa.
    long long prefix;
    if (rank < r) {
        prefix = rank * (q + 1);               // Ako je rank < r, pre njega ima rank*(q+1) elemenata
    } else {
        prefix = r * (q + 1) + (rank - r) * q; // Ako je rank >= r, pre njega ima r*(q+1) + (rank-r)*q
    }

    // Lokalni segment je raspon [local_start, local_end]
    long long local_start = 1 + prefix;                    // Prvi broj u segmentu
    long long local_end   = local_start + local_count - 1; // Poslednji broj u segmentu

    long long local_sum = 0;
    if (local_count > 0) {
        long long a = local_start;
        long long b = local_end;
        long long cnt = local_count;
        local_sum = (a + b) * cnt / 2;   // Računanje formule za aritmetički niz bez petlje
    }

    // Sabiramo sve local_sum vrednosti na proces 0
    long long global_sum = 0;
    MPI_Reduce(&local_sum, &global_sum,
               1, MPI_LONG_LONG, MPI_SUM,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Sum(1..%lld) = %lld\n", N, global_sum);
    }

    MPI_Finalize();
    return 0;
}
