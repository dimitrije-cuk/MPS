#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

// Print an error on rank 0 and abort all processes.
static void die_rank0_abort(MPI_Comm comm, int rank, const char *msg)
{
    if (rank == 0) fprintf(stderr, "ERROR: %s\n", msg);
    MPI_Abort(comm, 1);
}

// Parse N from argv[1] (a non-negative integer).
static long long parse_N_or_abort(int argc, char **argv, int rank)
{
    long long N = -1;

    if (rank == 0) {
        if (argc < 2) {
            fprintf(stderr, "Usage: %s <N>\n", argv[0]);
            fprintf(stderr, "  N must be a positive integer (vector length)\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        char *end = NULL;
        long long tmp = strtoll(argv[1], &end, 10);

        if (end == argv[1] || *end != '\0' || tmp < 1) {
            fprintf(stderr, "Usage: %s <N>\n", argv[0]);
            fprintf(stderr, "  N must be a positive integer (vector length)\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        N = tmp;
    }

    return N;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    long long N = parse_N_or_abort(argc, argv, rank);        // Only rank 0 parses N; the other processes receive it below.
    MPI_Bcast(&N, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);      // All processes receive N.

    // Root allocates the full vectors (only root holds complete A and B).
    double *A = NULL;
    double *B = NULL;

    // Root generates random data (the same input is used for sequential and parallel calculations).
    if (rank == 0) {
        A = (double *)malloc((size_t)N * sizeof(double));
        B = (double *)malloc((size_t)N * sizeof(double));
        if (!A || !B) die_rank0_abort(MPI_COMM_WORLD, rank, "out of memory for full vectors");

        // Seed based on the current time (one seed is enough because only rank 0 generates data).
        srand((unsigned)time(NULL));

        // Generate random values in the range [-1, 1].
        for (long long i = 0; i < N; i++) {
            A[i] = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
            B[i] = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
        }

    }

    // Measure the sequential execution time on root only.
    double seq_sum = 0.0;
    double seq_time = 0.0;

    if (rank == 0) {
        double t0 = MPI_Wtime();
        for (long long i = 0; i < N; i++) {
            seq_sum += A[i] * B[i];
        }
        double t1 = MPI_Wtime();
        seq_time = t1 - t0;
    }

    // Distribute elements evenly, including when N is not divisible by p.
    long long q = N / p;
    long long r = N % p;

    int local_n = (int)(q + (rank < r ? 1 : 0));    // The first r processes receive q+1 elements each; the rest receive q.

    // Root prepares sendcounts/displs for Scatterv.
    int *sendcounts = NULL;
    int *displs     = NULL;

    if (rank == 0) {
        sendcounts = (int *)malloc((size_t)p * sizeof(int));
        displs     = (int *)malloc((size_t)p * sizeof(int));
        if (!sendcounts || !displs) die_rank0_abort(MPI_COMM_WORLD, rank, "out of memory for sendcounts/displs");

        long long disp = 0;

        for (int i = 0; i < p; i++) {
            long long rows_i = q + (i < r ? 1 : 0);
            if (rows_i > (long long)INT_MAX)
                die_rank0_abort(MPI_COMM_WORLD, rank, "N too large for Scatterv int counts");
            if (disp > (long long)INT_MAX)
                die_rank0_abort(MPI_COMM_WORLD, rank, "N too large for Scatterv int displs");

            sendcounts[i] = (int)rows_i;                     // Number of elements sent to process i.
            displs[i]     = (int)disp;                       // Starting index of process i's block in A/B.
            disp += rows_i;                                  // Advance the offset to the next block.
        }
    }

    // Each process allocates its local portions of the vectors.
    double *local_A = NULL;
    double *local_B = NULL;

    if (local_n > 0) {
        local_A = (double *)malloc((size_t)local_n * sizeof(double));
        local_B = (double *)malloc((size_t)local_n * sizeof(double));
        if (!local_A || !local_B) die_rank0_abort(MPI_COMM_WORLD, rank, "out of memory for local vectors");
    }

    // Synchronize before timing the parallel section for a fair measurement.
    MPI_Barrier(MPI_COMM_WORLD);

    double par_t0 = MPI_Wtime();                              // Start the parallel timing interval.

    // Scatter A: root sends blocks of different sizes; the other processes receive them.
    MPI_Scatterv(
        A, sendcounts, displs, MPI_DOUBLE,                    // Root: global array and distribution metadata.
        local_A, local_n, MPI_DOUBLE,                         // All processes: local buffer and local size.
        0, MPI_COMM_WORLD
    );

    // Scatter B using the same distribution as for A.
    MPI_Scatterv(
        B, sendcounts, displs, MPI_DOUBLE,                    // Root: global array and distribution metadata.
        local_B, local_n, MPI_DOUBLE,                         // All processes: local buffer and local size.
        0, MPI_COMM_WORLD
    );

    // Each process computes the partial sum for its segment.
    double local_sum = 0.0;
    for (int i = 0; i < local_n; i++) {
        local_sum += local_A[i] * local_B[i];
    }

    // Reduce combines all local_sum values on root.
    double par_sum = 0.0;
    MPI_Reduce(&local_sum, &par_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);                              // Ensure all processes finish before stopping the timer.
    double par_t1 = MPI_Wtime();                              // End the parallel timing interval.

    double par_time = par_t1 - par_t0;                        // Total time for the parallel section.

    // Root compares the results and calculates the speedup.
    if (rank == 0) {
        double diff = par_sum - seq_sum;
        if (diff < 0) diff = -diff;                           // Absolute value without math.h.

        double speedup = (par_time > 0.0) ? (seq_time / par_time) : 0.0;

        printf("------------------------------------------------------------\n");
        printf("Dot product benchmark (MPI)\n");
        printf("N          = %lld\n", N);
        printf("Processes  = %d\n", p);
        printf("------------------------------------------------------------\n");
        printf("Sequential sum = %.10f\n", seq_sum);
        printf("Parallel   sum = %.10f\n", par_sum);
        printf("Abs diff       = %.10e\n", diff);
        printf("------------------------------------------------------------\n");
        printf("Sequential time = %.6f s\n", seq_time);
        printf("Parallel   time = %.6f s\n", par_time);
        printf("Speedup         = %.3f x\n", speedup);
        printf("------------------------------------------------------------\n");

        // Check whether the difference exceeds the allowed floating-point tolerance
        // (relative tolerance 1e-9 * |seq_sum| + absolute tolerance 1e-12).
        if (diff > 1e-9 * (seq_sum >= 0 ? seq_sum : -seq_sum) + 1e-12) {
            printf("WARNING: Results differ more than expected (floating-point order effects).\n");
        } else {
            printf("OK: Results match within tolerance.\n");
        }
    }

    // Clean up local buffers.
    free(local_A);
    free(local_B);

    // Clean up root's global buffers and Scatterv metadata.
    if (rank == 0) {
        free(A);
        free(B);
        free(sendcounts);
        free(displs);
    }

    MPI_Finalize();
    return 0;
}
