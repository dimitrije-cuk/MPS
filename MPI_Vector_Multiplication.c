#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

// Ispis greške na rank 0 i prekid svih procesa
static void die_rank0_abort(MPI_Comm comm, int rank, const char *msg)
{
    if (rank == 0) fprintf(stderr, "ERROR: %s\n", msg);
    MPI_Abort(comm, 1);
}

// Parsiranje N iz argv[1] (nenegativan ceo broj)
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

    long long N = parse_N_or_abort(argc, argv, rank);        // Samo rank0 parsira, ostali dobiju kasnije
    MPI_Bcast(&N, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);      // Svi procesi dobijaju N

    // Root alocira pune vektore (samo root drži kompletne A i B)
    double *A = NULL;
    double *B = NULL;

    // Root generiše random podatke (isti ulaz i za seq i za parallel)
    if (rank == 0) {
        A = (double *)malloc((size_t)N * sizeof(double));
        B = (double *)malloc((size_t)N * sizeof(double));
        if (!A || !B) die_rank0_abort(MPI_COMM_WORLD, rank, "out of memory for full vectors");

        // Seed na osnovu vremena (jedan seed je dovoljan jer samo rank0 generiše)
        srand((unsigned)time(NULL));

        // Random vrednosti u opsegu [-1, 1]
        for (long long i = 0; i < N; i++) {
            A[i] = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
            B[i] = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
        }

    }

    // Sekvencijalno vreme merimo samo na root-u
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

    // Ravnomerna, ali neujednačena raspodela elemenata kada N nije deljivo sa p
    long long q = N / p;
    long long r = N % p;

    int local_n = (int)(q + (rank < r ? 1 : 0));    // Prvih r procesa dobija po (q+1) elemenata, ostali dobijaju po q elemenata

    // Root priprema sendcounts/displs za Scatterv
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

            sendcounts[i] = (int)rows_i;                     // Koliko elemenata šaljemo procesu i
            displs[i]     = (int)disp;                       // Od kog indeksa u A/B kreće blok za proces i
            disp += rows_i;                                  // Pomeri offset na sledeći blok
        }
    }

    // Svaki proces alocira svoje lokalne delove vektora
    double *local_A = NULL;
    double *local_B = NULL;

    if (local_n > 0) {
        local_A = (double *)malloc((size_t)local_n * sizeof(double));
        local_B = (double *)malloc((size_t)local_n * sizeof(double));
        if (!local_A || !local_B) die_rank0_abort(MPI_COMM_WORLD, rank, "out of memory for local vectors");
    }

    // Sinhronizacija pre merenja paralelnog dela (da merenje bude fer)
    MPI_Barrier(MPI_COMM_WORLD);

    double par_t0 = MPI_Wtime();                              // Start paralelnog merenja

    // Scatterv za A: root šalje različite veličine blokova, ostali samo primaju
    MPI_Scatterv(
        A, sendcounts, displs, MPI_DOUBLE,                    // Root: globalni niz + raspodela
        local_A, local_n, MPI_DOUBLE,                         // Svi: lokalni bafer + lokalna veličina
        0, MPI_COMM_WORLD
    );

    // Scatterv za B: ista raspodela kao za A
    MPI_Scatterv(
        B, sendcounts, displs, MPI_DOUBLE,                    // Root: globalni niz + raspodela
        local_B, local_n, MPI_DOUBLE,                         // Svi: lokalni bafer + lokalna veličina
        0, MPI_COMM_WORLD
    );

    // Svaki proces računa parcijalnu sumu za svoj segment
    double local_sum = 0.0;
    for (int i = 0; i < local_n; i++) {
        local_sum += local_A[i] * local_B[i];
    }

    // Reduce sabira sve local_sum vrednosti na root
    double par_sum = 0.0;
    MPI_Reduce(&local_sum, &par_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);                              // Barijera da svi završe pre stop vremena
    double par_t1 = MPI_Wtime();                              // Kraj paralelnog merenja

    double par_time = par_t1 - par_t0;                        // Ukupno vreme paralelnog dela

    // Root upoređuje rezultate i računa ubrzanje
    if (rank == 0) {
        double diff = par_sum - seq_sum;
        if (diff < 0) diff = -diff;                           // Apsolutna vrednost bez math.h

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

        // Provera da li je razlika veća od dozvoljene floating-point tolerancije
        // (relativna tolerancija 1e-9 * |seq_sum| + apsolutna 1e-12).
        if (diff > 1e-9 * (seq_sum >= 0 ? seq_sum : -seq_sum) + 1e-12) {
            printf("WARNING: Results differ more than expected (floating-point order effects).\n");
        } else {
            printf("OK: Results match within tolerance.\n");
        }
    }

    // Cleanup lokalnih bafera
    free(local_A);
    free(local_B);

    // Cleanup root globalnih bafera i meta-podataka za Scatterv
    if (rank == 0) {
        free(A);
        free(B);
        free(sendcounts);
        free(displs);
    }

    MPI_Finalize();
    return 0;
}
