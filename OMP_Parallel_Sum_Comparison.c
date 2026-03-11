#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

/* Paralelna suma koristeći OpenMP reduction mehanizam. */
static long long sum_reduction(long long n, double *elapsed_sec)
{
    double t0 = omp_get_wtime();

    long long sum = 0;

    /* Svaka nit ima privatnu sumu, OpenMP ih automatski sabira na kraju. */
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (long long i = 1; i <= n; ++i) {
        sum += i;
    }

    double t1 = omp_get_wtime();
    *elapsed_sec = t1 - t0;
    return sum;
}

/* Ručno računanje parcijalnih suma — ono što reduction radi interno. */
static long long sum_manual_partials(long long n, double *elapsed_sec)
{
    int T = omp_get_max_threads();

    /* Po jedna parcijalna suma po niti. */
    long long *partial = (long long*)calloc((size_t)T, sizeof(long long));
    if (!partial) {
        fprintf(stderr, "Allocation failed.\n");
        exit(1);
    }

    double t0 = omp_get_wtime();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        long long local = 0;

        /* Svaka nit sabira svoj deo opsega. */
        #pragma omp for schedule(static)
        for (long long i = 1; i <= n; ++i) {
            local += i;
        }

        partial[tid] = local;   // Upis lokalne sume
    }

    /* Završna serijska akumulacija parcijalnih suma. */
    long long sum = 0;
    for (int t = 0; t < T; ++t) {
        sum += partial[t];
    }

    double t1 = omp_get_wtime();
    *elapsed_sec = t1 - t0;

    free(partial);
    return sum;
}

/* Matematička formula za sumu 1..n — služi za proveru tačnosti. */
static long long sum_closed_form(long long n)
{
    return (n * (n + 1)) / 2;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <n> [repetitions]\n", argv[0]);
        return 1;
    }

    long long n = atoll(argv[1]);
    if (n < 0) {
        fprintf(stderr, "Error: n must be >= 0.\n");
        return 1;
    }

    printf("OpenMP max threads: %d\n", omp_get_max_threads());
    printf("n = %lld\n\n", n);

    long long expected = sum_closed_form(n);

    double best_red = 1e300, best_man = 1e300;
    long long sum_red_best = 0, sum_man_best = 0;

    double t_red = 0.0, t_man = 0.0;

    /* Jedno merenje za obe metode. */
    long long s_red = sum_reduction(n, &t_red);
    long long s_man = sum_manual_partials(n, &t_man);

    if (t_red < best_red) { best_red = t_red; sum_red_best = s_red; }
    if (t_man < best_man) { best_man = t_man; sum_man_best = s_man; }

    /* Osnovna provera korektnosti. */
    int ok_red = (sum_red_best == expected);
    int ok_man = (sum_man_best == expected);

    printf("Expected (closed form) : %lld\n\n", expected);

    printf("[A] reduction(+:sum)\n");
    printf("    sum    : %lld   (%s)\n", sum_red_best, ok_red ? "OK" : "MISMATCH");
    printf("    time   : %.6f s\n\n", best_red);

    printf("[B] manual partial sums + final accumulation\n");
    printf("    sum    : %lld   (%s)\n", sum_man_best, ok_man ? "OK" : "MISMATCH");
    printf("    time   : %.6f s\n\n", best_man);

    if (best_man > 0.0) {
        printf("Speed ratio (A/B): %.3f\n", best_red / best_man);
    }

    return 0;
}
