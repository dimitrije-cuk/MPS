#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Rank 0 prints the error and aborts all processes.
static void die_rank0_abort(MPI_Comm comm, int rank, const char *msg)
{
    if (rank == 0) {
        fprintf(stderr, "ERROR: %s\n", msg);
    }
    MPI_Abort(comm, 1);
}

// Counts the number of double values in the file.
static int count_doubles_in_file(const char *fname)
{
    FILE *f = fopen(fname, "r");
    if (!f) return -1;

    int count = 0;
    double tmp;
    // Read one double value at a time.
    while (fscanf(f, "%lf", &tmp) == 1) count++;

    fclose(f);
    return count;
}

// Loads a vector x of length n from a file.
static double *load_vector(const char *fname, int n)
{
    FILE *f = fopen(fname, "r");
    if (!f) return NULL;

    double *x = (double *)malloc((size_t)n * sizeof(double));
    if (!x) { fclose(f); return NULL; }

    for (int i = 0; i < n; i++) {             // The file must contain exactly n numbers.
        if (fscanf(f, "%lf", &x[i]) != 1) {
            free(x);
            fclose(f);
            return NULL;
        }
    }

    fclose(f);
    return x;
}

// Loads an n×n matrix A from a file (row-major: rows stored consecutively).
static double *load_matrix(const char *fname, int n)
{
    FILE *f = fopen(fname, "r");
    if (!f) return NULL;

    size_t m = (size_t)n * (size_t)n;         // Total number of matrix elements.
    double *A = (double *)malloc(m * sizeof(double));
    if (!A) { fclose(f); return NULL; }

    for (size_t i = 0; i < m; i++) {          // Read m = n*n numbers.
        if (fscanf(f, "%lf", &A[i]) != 1) {
            free(A);
            fclose(f);
            return NULL;
        }
    }

    fclose(f);
    return A;                                 // A[i*n + j] is the element in row i, column j (row-major format).
}

// Writes the result y[0..n-1] to a file as a list of numbers.
static void write_result(const char *fname, const double *y, int n)
{
    FILE *f = fopen(fname, "w");
    if (!f) return;

    for (int i = 0; i < n; i++) {
        fprintf(f, "%lf%s", y[i], (i + 1 == n) ? "" : " "); // Separate values with spaces, except after the last element.
    }

    fprintf(f, "\n");
    fclose(f);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);       // Process rank.
    MPI_Comm_size(MPI_COMM_WORLD, &p);          // Number of processes.

    // The program expects a vector filename and a matrix filename.
    if (argc != 3) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s <vector_file> <matrix_file>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    const char *vec_file = argv[1];           // Path to vector x.
    const char *mat_file = argv[2];           // Path to matrix A.

    int n = 0;  // Dimension of vector x(n), matrix A(n×n), and result y(n).

    // Rank 0 determines the dimension n.
    if (rank == 0) {
        n = count_doubles_in_file(vec_file);
        if (n <= 0) {
            fprintf(stderr, "ERROR: cannot read vector size from '%s'\n", vec_file);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int q = n / p;                            // Base number of rows per process.
    int r = n % p;                            // The first r processes receive one extra row.

    int local_rows = q + (rank < r ? 1 : 0);  // Number of rows assigned to this process.

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

        int dispA = 0;                        // Offset within the loaded matrix Afull.
        int dispY = 0;                        // Offset within result y.

        for (int i = 0; i < p; i++) {
            int rows_i = q + (i < r ? 1 : 0); // Number of rows assigned to process i.

            sendcountsA[i] = rows_i * n;      // rows_i rows * n columns.
            displsA[i]     = dispA;           // Starting index of the block in Afull.

            recvcountsY[i] = rows_i;          // rows_i results.
            displsY[i]     = dispY;           // Starting index of the block in y.

            dispA += sendcountsA[i];          // The next matrix block starts after this one.
            dispY += recvcountsY[i];          // The next result block starts after this one.
        }
    }

    // Allocate vector x on all processes.
    double *x = (double *)malloc((size_t)n * sizeof(double));
    if (!x) die_rank0_abort(MPI_COMM_WORLD, rank, "out of memory for vector x");

    // Rank 0 loads x.
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

    // Rank 0 loads the entire matrix A.
    double *Afull = NULL;
    if (rank == 0) {
        Afull = load_matrix(mat_file, n);
        if (!Afull) {
            free(x);
            die_rank0_abort(MPI_COMM_WORLD, rank, "failed to read matrix file (format/size mismatch)");
        }
    }

    // Allocate the local matrix block: local_rows × n.
    double *Alocal = NULL;
    if (local_rows > 0) {
        Alocal = (double *)malloc((size_t)local_rows * (size_t)n * sizeof(double));
        if (!Alocal) {
            free(x);
            if (rank == 0) free(Afull);
            die_rank0_abort(MPI_COMM_WORLD, rank, "out of memory for local matrix chunk");
        }
    }

    // Scatterv sends a different number of rows to each process.
    MPI_Scatterv(
        Afull, sendcountsA, displsA, MPI_DOUBLE,
        Alocal, local_rows * n, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    // Allocate the local result ylocal.
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
            const double *row = &Alocal[(size_t)i * (size_t)n]; // Start of row i.
            for (int j = 0; j < n; j++) {
                sum += row[j] * x[j];                           // Dot product of row i and vector x.
            }
            ylocal[i] = sum;                                    // Result for this row.
        }
    }

    // Rank 0 allocates the full result y (length n) because it gathers all partial results.
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

    // Rank 0 writes the result to a file.
    if (rank == 0)
        write_result("MPI_Matrix_Vector-Result.txt", y, n);

    // Free allocated memory.
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
