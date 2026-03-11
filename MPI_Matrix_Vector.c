#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Rank 0 ispisuje grešku i prekida sve procese
static void die_rank0_abort(MPI_Comm comm, int rank, const char *msg)
{
    if (rank == 0) {
        fprintf(stderr, "ERROR: %s\n", msg);
    }
    MPI_Abort(comm, 1);
}

// Prebrojava koliko double vrednosti postoji u fajlu
static int count_doubles_in_file(const char *fname)
{
    FILE *f = fopen(fname, "r");
    if (!f) return -1;

    int count = 0;
    double tmp;
    // Čitaj double po double
    while (fscanf(f, "%lf", &tmp) == 1) count++;

    fclose(f);
    return count;
}

// Učitava vektor x dužine n iz fajla
static double *load_vector(const char *fname, int n)
{
    FILE *f = fopen(fname, "r");
    if (!f) return NULL;

    double *x = (double *)malloc((size_t)n * sizeof(double));
    if (!x) { fclose(f); return NULL; }

    for (int i = 0; i < n; i++) {             // Tačno n brojeva mora da postoji
        if (fscanf(f, "%lf", &x[i]) != 1) {
            free(x);
            fclose(f);
            return NULL;
        }
    }

    fclose(f);
    return x;
}

// Učitava matricu A dimenzija n×n iz fajla (row-major: redovi jedan za drugim)
static double *load_matrix(const char *fname, int n)
{
    FILE *f = fopen(fname, "r");
    if (!f) return NULL;

    size_t m = (size_t)n * (size_t)n;         // Ukupan broj elemenata matrice
    double *A = (double *)malloc(m * sizeof(double));
    if (!A) { fclose(f); return NULL; }

    for (size_t i = 0; i < m; i++) {          // Učitavamo m = n*n brojeva
        if (fscanf(f, "%lf", &A[i]) != 1) {
            free(A);
            fclose(f);
            return NULL;
        }
    }

    fclose(f);
    return A;                                 // A[i*n + j] je element u i-tom redu i j-toj koloni - row-major format
}

// Upisuje rezultat y[0..n-1] u fajl kao listu brojeva
static void write_result(const char *fname, const double *y, int n)
{
    FILE *f = fopen(fname, "w");
    if (!f) return;

    for (int i = 0; i < n; i++) {
        fprintf(f, "%lf%s", y[i], (i + 1 == n) ? "" : " "); // Razdvajanje razmakom osim nakon poslednjeg elementa
    }

    fprintf(f, "\n");
    fclose(f);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);       // Rank procesa
    MPI_Comm_size(MPI_COMM_WORLD, &p);          // Broj procesa

    // Program očekuje: ime_fajla_vektora i ime_fajla_matrice
    if (argc != 3) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s <vector_file> <matrix_file>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    const char *vec_file = argv[1];           // Putanja do vektora x
    const char *mat_file = argv[2];           // Putanja do matrice A

    int n = 0;  // Dimenzija vektora x(n), matrice A(n×n), i rezultata y(n)

    // Rank 0 određuje dimenziju n
    if (rank == 0) {
        n = count_doubles_in_file(vec_file);
        if (n <= 0) {
            fprintf(stderr, "ERROR: cannot read vector size from '%s'\n", vec_file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int q = n / p;                            // Osnovni broj redova po procesu
    int r = n % p;                            // Prvih r procesa dobija +1 red

    int local_rows = q + (rank < r ? 1 : 0);  // Broj redova za ovaj proces

    int *sendcountsA = NULL;
    int *displsA     = NULL;
    int *recvcountsY = NULL;
    int *displsY     = NULL;

    if (rank == 0) {
        sendcountsA = (int *)malloc((size_t)p * sizeof(int));
        displsA     = (int *)malloc((size_t)p * sizeof(int));
        recvcountsY = (int *)malloc((size_t)p * sizeof(int));
        displsY     = (int *)malloc((size_t)p * sizeof(int));

        if (!sendcountsA || !displsA || !recvcountsY || !displsY)
            die_rank0_abort(MPI_COMM_WORLD, rank, "allocation failed");

        int dispA = 0;                        // Offset unutar učitane matrice Afull
        int dispY = 0;                        // Offset unutar rezultata y

        for (int i = 0; i < p; i++) {
            int rows_i = q + (i < r ? 1 : 0); // Koliko redova dobija proces i

            sendcountsA[i] = rows_i * n;      // rows_i redova * n kolona
            displsA[i]     = dispA;           // Gde u Afull počinje blok

            recvcountsY[i] = rows_i;          // rows_i rezultata
            displsY[i]     = dispY;           // Gde u y počinje blok

            dispA += sendcountsA[i];          // Sledeći blok matrice počinje posle ovog
            dispY += recvcountsY[i];          // Sledeći blok rezultata počinje posle ovog
        }
    }

    // Alokacija vektora x na svim procesima
    double *x = (double *)malloc((size_t)n * sizeof(double));
    if (!x) die_rank0_abort(MPI_COMM_WORLD, rank, "out of memory for vector x");

    // Rank 0 učitava x
    if (rank == 0) {
        double *tmp = load_vector(vec_file, n);
        if (!tmp) {
            free(x);
            die_rank0_abort(MPI_COMM_WORLD, rank, "vector read failed");
        }
        for (int i = 0; i < n; i++) x[i] = tmp[i];
        free(tmp);
    }

    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Rank 0 učitava celu matricu A
    double *Afull = NULL;
    if (rank == 0) {
        Afull = load_matrix(mat_file, n);
        if (!Afull) {
            free(x);
            die_rank0_abort(MPI_COMM_WORLD, rank, "failed to read matrix file (format/size mismatch)");
        }
    }

    // Alokacija lokalnog dela matrice: local_rows × n
    double *Alocal = NULL;
    if (local_rows > 0) {
        Alocal = (double *)malloc((size_t)local_rows * (size_t)n * sizeof(double));
        if (!Alocal) {
            free(x);
            if (rank == 0) free(Afull);
            die_rank0_abort(MPI_COMM_WORLD, rank, "out of memory for local matrix chunk");
        }
    }

    // Scatterv šalje različit broj redova svakom procesu
    MPI_Scatterv(
        Afull, sendcountsA, displsA, MPI_DOUBLE,
        Alocal, local_rows * n, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    // Alokacija lokalnog rezultata ylocal
    double *ylocal = NULL;
    if (local_rows > 0) {
        ylocal = (double *)malloc((size_t)local_rows * sizeof(double));
        if (!ylocal) {
            free(x);
            if (rank == 0) free(Afull);
            free(Alocal);
            die_rank0_abort(MPI_COMM_WORLD, rank, "out of memory for local result chunk");
        }

        for (int i = 0; i < local_rows; i++) {
            double sum = 0.0;
            const double *row = &Alocal[(size_t)i * (size_t)n]; // Početak i-tog reda
            for (int j = 0; j < n; j++) {
                sum += row[j] * x[j];                           // Skalarni proizvod reda i vektora
            }
            ylocal[i] = sum;                                    // Rezultat za taj red
        }
    }

    // Rank 0 alocira ceo rezultat y (dužine n) jer samo on skuplja sve parcijalne delove
    double *y = NULL;
    if (rank == 0) {
        y = (double *)malloc((size_t)n * sizeof(double));
        if (!y) {
            free(x);
            free(Afull);
            die_rank0_abort(MPI_COMM_WORLD, rank, "out of memory for full result y");
        }
    }

    MPI_Gatherv(
        ylocal, local_rows, MPI_DOUBLE,
        y, recvcountsY, displsY, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    // Rank 0 upisuje rezultat u fajl
    if (rank == 0)
        write_result("MPI_Matrix_Vector-Result.txt", y, n);

    // Oslobađanje memorije
    free(x);
    free(Alocal);
    free(ylocal);

    if (rank == 0) {
        free(Afull);
        free(y);
        free(sendcountsA);
        free(displsA);
        free(recvcountsY);
        free(displsY);
    }

    MPI_Finalize();
    return 0;
}
