#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size;
    long long N = 0;    // Upper bound of the sum: 1 + 2 + ... + N.

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Input: process 0 determines the value of N.
    if (rank == 0) {
        printf("Enter N (non-negative integer): ");
        fflush(stdout);

        if (scanf("%lld", &N) != 1 || N < 0) {
            fprintf(stderr,
                    "Invalid input. N must be a non-negative integer.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Distribute N to all processes.
    MPI_Bcast(&N, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    // Compute an even distribution of the work.
    long long q = N / size;
    long long r = N % size;

    // The first r processes receive q+1 elements each; the rest receive q.
    long long local_count = (rank < r) ? (q + 1) : q;

    // Compute the prefix: the number of elements assigned before this process.
    long long prefix;
    if (rank < r) {
        prefix = rank * (q + 1);               // If rank < r, there are rank*(q+1) preceding elements.
    } else {
        prefix = r * (q + 1) + (rank - r) * q; // If rank >= r, there are r*(q+1) + (rank-r)*q preceding elements.
    }

    // The local segment is the range [local_start, local_end].
    long long local_start = 1 + prefix;                    // First number in the segment.
    long long local_end   = local_start + local_count - 1; // Last number in the segment.

    long long local_sum = 0;
    if (local_count > 0) {
        long long a = local_start;
        long long b = local_end;
        long long cnt = local_count;
        local_sum = (a + b) * cnt / 2;   // Compute the arithmetic-series sum without a loop.
    }

    // Sum all local_sum values on process 0.
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
