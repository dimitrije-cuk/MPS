# MPS Homework Assignments

This repository contains MPI and OpenMP programming assignments.

## Installing Microsoft MPI on Windows

Install **Microsoft MPI v10.1.3** from the [Microsoft Download Center](https://www.microsoft.com/en-us/download/details.aspx?id=105289).

Install both components:

* `msmpisetup.exe` — MPI runtime
* `msmpisdk.msi` — headers and libraries (SDK)

Set the SDK paths in the command prompt:

```bat
set "MSMPI_INC=C:\Program Files (x86)\Microsoft SDKs\MPI\Include"
set "MSMPI_LIB64=C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64"
```

## Assignment 2.4 — MPI All-to-All

```bat
gcc MPI_AllToAll_TwoDigit.c -I"%MSMPI_INC%" -L"%MSMPI_LIB64%" -lmsmpi -o MPI_AllToAll_TwoDigit
mpiexec -n 4 MPI_AllToAll_TwoDigit
```

## Assignment 3.2 — MPI Block Sum

```bat
gcc MPI_Parallel_Sum_Block.c -I"%MSMPI_INC%" -L"%MSMPI_LIB64%" -lmsmpi -o MPI_Parallel_Sum_Block
mpiexec -n 4 MPI_Parallel_Sum_Block
```

## Assignment 3.3 — MPI Vector Dot Product

```bat
gcc MPI_Vector_Multiplication.c -I"%MSMPI_INC%" -L"%MSMPI_LIB64%" -lmsmpi -o MPI_Vector_Multiplication
mpiexec -n 4 MPI_Vector_Multiplication 100000000
```

## Assignment 4.1 — MPI Matrix-Vector Multiplication

```bat
gcc MPI_Matrix_Vector.c -I"%MSMPI_INC%" -L"%MSMPI_LIB64%" -lmsmpi -o MPI_Matrix_Vector
mpiexec -n 4 MPI_Matrix_Vector MPI_Matrix_Vector-Vector.txt MPI_Matrix_Vector-Matrix.txt
```

## Assignment 5.5 — OpenMP Sum Comparison

```bat
gcc -O2 -Wall -Wextra -fopenmp OMP_Parallel_Sum_Comparison.c -o OMP_Parallel_Sum_Comparison
OMP_Parallel_Sum_Comparison 1000000000
```

## Assignment 6.2 — OpenMP Sieve of Eratosthenes

```bat
gcc -O2 -Wall -Wextra -fopenmp OMP_Sieve_of_Eratosthenes.c -o OMP_Sieve_of_Eratosthenes
OMP_Sieve_of_Eratosthenes 10000000 primes.txt
```
