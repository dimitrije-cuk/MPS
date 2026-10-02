#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <omp.h>

/* Parses a long integer from a string; requires an integer >= 2. */
static long parse_long(const char *s)
{
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);

    // Reject conversion errors, missing digits, trailing characters, or v < 2.
    if (errno != 0 || end == s || *end != '\0' || v < 2) {
        fprintf(stderr, "Invalid N: '%s' (must be integer >= 2)\n", s);
        exit(EXIT_FAILURE);
    }

    return v;
}

/* Runs the parallel Sieve of Eratosthenes up to N; optionally returns the flags and prime count. */
static double sieve_run(long N, unsigned char **out_is_prime, long *out_count)
{
    // Allocate N+1 flags indicating whether each number is prime (1) or composite (0).
    unsigned char *is_prime = (unsigned char*)malloc((size_t)(N + 1));
    if (!is_prime) {
        fprintf(stderr, "Allocation failed for N=%ld\n", N);
        exit(EXIT_FAILURE);
    }

    // Assume all numbers are prime (1), except 0 and 1, which are not (0).
    memset(is_prime, 1, (size_t)(N + 1));
    is_prime[0] = 0;
    is_prime[1] = 0;

    // Start timing.
    double t0 = omp_get_wtime();

    #pragma omp parallel
    {
        for (long p = 2; p * p <= N; ++p) {
            // If p is marked prime, mark all its multiples as composite.
            if (is_prime[p]) {
                // All numbers below p*p have already been processed.
                long start = p * p;

                // There is an implicit barrier at the start of the omp for loop.
                #pragma omp for schedule(runtime)
                for (long m = start; m <= N; m += p) {
                    is_prime[m] = 0;
                } // There is an implicit barrier at the end of the omp for loop.
            }
        }
    }

    // Stop timing.
    double t1 = omp_get_wtime();

    // Count prime numbers after the sieve finishes.
    long count = 0;
    for (long i = 2; i <= N; ++i)
        if (is_prime[i]) ++count;

    // Return requested results, or free the allocation if it was not requested.
    if (out_count) *out_count = count;
    if (out_is_prime) *out_is_prime = is_prime;
    else free(is_prime);

    // Return the elapsed time.
    return t1 - t0;
}

/* Writes the prime numbers (from the is_prime array) to a file, one per line. */
static void write_primes_to_file(const char *path, const unsigned char *is_prime, long N)
{
    // Open the file for writing (binary mode) and check that it opened successfully.
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "Cannot open output file '%s'\n", path);
        exit(EXIT_FAILURE);
    }

    // Set a 1 MB buffer to speed up I/O.
    setvbuf(fp, NULL, _IOFBF, 1 << 20);

    // Iterate from 2 through N and write only numbers marked as prime.
    for (long i = 2; i <= N; ++i)
        if (is_prime[i]) fprintf(fp, "%ld\n", i);

    // Close the file and flush the buffer.
    fclose(fp);
}

/* Converts an omp_sched_t value to a printable name. */
static const char *sched_name(omp_sched_t s)
{
    // Select the name based on the OpenMP enum value.
    switch (s) {
        case omp_sched_static:  return "static";  // Static scheduling.
        case omp_sched_dynamic: return "dynamic"; // Dynamic scheduling.
        case omp_sched_guided:  return "guided";  // Guided scheduling.
        case omp_sched_auto:    return "auto";    // Automatic runtime selection.
        default:                return "unknown"; // Unknown value.
    }
}

/* Stores benchmark results for a scheduling policy. */
typedef struct
{
    omp_sched_t sched;                          // Scheduling policy (static/dynamic/guided).
    int chunk;                                  // Chunk size.
    double time_s;                              // Elapsed time in seconds.
    long prime_count;                           // Number of primes found.
} bench_result_t;

int main(int argc, char **argv)
{
    // Require N and optionally an output filename.
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <N> [output_file]\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Parse N from the command line and select the output file.
    long N = parse_long(argv[1]);
    const char *out_path = (argc == 3) ? argv[2] : "primes.txt";

    // Disable dynamic adjustment of the number of OpenMP threads.
    omp_set_dynamic(0);

    // Print N and the maximum number of threads supported by OpenMP on this system.
    printf("N = %ld\n", N);
    printf("OpenMP max threads = %d\n\n", omp_get_max_threads());

    // Define chunk sizes for the scheduling policies.
    const int chunk_static  = 0;
    const int chunk_dynamic = 1024;
    const int chunk_guided  = 1024;

    // Store results for the three scheduling policies.
    bench_result_t results[3];

    // Set the scheduling policy and chunk size for each of the three tests.
    results[0].sched = omp_sched_static;
    results[0].chunk = chunk_static;
    results[1].sched = omp_sched_dynamic;
    results[1].chunk = chunk_dynamic;
    results[2].sched = omp_sched_guided;
    results[2].chunk = chunk_guided;

    // Benchmark each scheduling policy and store its results.
    for (int i = 0; i < 3; ++i) {
        // Set the policy and corresponding chunk size for this test.
        omp_set_schedule(results[i].sched, results[i].chunk);

        // Run the sieve without returning the flags; get the elapsed time and prime count.
        long count = 0;
        double t = sieve_run(N, NULL, &count);

        // Store the elapsed time and prime count.
        results[i].time_s = t;
        results[i].prime_count = count;

        // Print the policy name, chunk size, elapsed time, and prime count.
        printf("Schedule: %-7s  chunk: %-5d  time: %.6f s  primes: %ld\n",
               sched_name(results[i].sched),
               results[i].chunk,
               t,
               count);
    }

    // Find the fastest scheduling policy.
    int best = 0;
    for (int i = 1; i < 3; ++i)
        if (results[i].time_s < results[best].time_s) best = i;

    // Print the fastest scheduling policy.
    printf("\nBest schedule: %s (chunk=%d), time=%.6f s\n",
           sched_name(results[best].sched),
           results[best].chunk,
           results[best].time_s);

    // Use the fastest policy for the final run, which returns the prime flags.
    omp_set_schedule(results[best].sched, results[best].chunk);

    // Run the sieve once more to get the is_prime array and prime count.
    unsigned char *is_prime = NULL;
    long count = 0;
    double t_final = sieve_run(N, &is_prime, &count);

    // Print the elapsed time and prime count for the final run.
    printf("Final run: %.6f s, primes=%ld\n", t_final, count);

    // Write the prime numbers to a file and free the allocated memory.
    write_primes_to_file(out_path, is_prime, N);
    free(is_prime);

    printf("Primes written to: %s\n", out_path);
    return EXIT_SUCCESS;
}
