# MPS domaci zadaci

## OpenMPI Windows instalacija

Instalirati **Microsoft MPI v10.1.3** sa [Microsoft Download Center](https://www.microsoft.com/en-us/download/details.aspx?id=105289).

Instalirati dve komponente:

* `msmpisetup.exe` – MPI runtime
* `msmpisdk.msi` – headers and libraries (SDK)

```bash
set "MSMPI_INC=C:\Program Files (x86)\Microsoft SDKs\MPI\Include"
set "MSMPI_LIB64=C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64"
```

## Zadatak 2.4

```bash
# Windows build
gcc MPI_AllToAll_TwoDigit.c -I"%MSMPI_INC%" -L"%MSMPI_LIB64%" -lmsmpi -o MPI_AllToAll_TwoDigit

# Run
mpiexec -n 4 MPI_AllToAll_TwoDigit
```

## Zadatak 3.2

```bash
# Windows build
gcc MPI_Parallel_Sum_Block.c -I"%MSMPI_INC%" -L"%MSMPI_LIB64%" -lmsmpi -o MPI_Parallel_Sum_Block

# Run
mpiexec -n 4 MPI_Parallel_Sum_Block
```

## Zadatak 3.3

```bash
# Windows build
gcc MPI_Vector_Multiplication.c -I"%MSMPI_INC%" -L"%MSMPI_LIB64%" -lmsmpi -o MPI_Vector_Multiplication

# Run
mpiexec -n 4 MPI_Vector_Multiplication 100000000
```

## Zadatak 4.1

```bash
# Windows build
gcc MPI_Matrix_Vector.c -I"%MSMPI_INC%" -L"%MSMPI_LIB64%" -lmsmpi -o MPI_Matrix_Vector

# Run
mpiexec -n 4 MPI_Matrix_Vector MPI_Matrix_Vector-Vector.txt MPI_Matrix_Vector-Matrix.txt
```

## Zadatak 5.5

```bash
# Windows build
gcc -O2 -Wall -Wextra -fopenmp OMP_Parallel_Sum_Comparison.c -o OMP_Parallel_Sum_Comparison

# Run
OMP_Parallel_Sum_Comparison 1000000000
```

## Zadatak 6.2

```bash
# Windows build
gcc -O2 -Wall -Wextra -fopenmp OMP_Sieve_of_Eratosthenes.c -o OMP_Sieve_of_Eratosthenes

# Run
OMP_Sieve_of_Eratosthenes 10000000 primes.txt
```
