#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <omp.h>

/* Parsira long iz stringa; zahteva ceo broj >= 2. */
static long parse_long(const char *s)
{
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);

    // Provera: greška konverzije, nema cifara, višak znakova ili v < 2
    if (errno != 0 || end == s || *end != '\0' || v < 2) {
        fprintf(stderr, "Invalid N: '%s' (must be integer >= 2)\n", s);
        exit(EXIT_FAILURE);
    }

    return v;
}

/* Pokreće paralelno Eratostenovo sito do N; opciono vraća niz i broj prostih. */
static double sieve_run(long N, unsigned char **out_is_prime, long *out_count)
{
    // Alokacija niza dužine N+1 koji označava da li je broj prost (1) ili složen (0).
    unsigned char *is_prime = (unsigned char*)malloc((size_t)(N + 1));
    if (!is_prime) {
        fprintf(stderr, "Allocation failed for N=%ld\n", N);
        exit(EXIT_FAILURE);
    }

    // Pretpostavimo da su svi brojevi prosti (1), osim 0 i 1 koji nisu (0).
    memset(is_prime, 1, (size_t)(N + 1));
    is_prime[0] = 0;
    is_prime[1] = 0;

    // Početak merenja vremena.
    double t0 = omp_get_wtime();

    #pragma omp parallel
    {
        for (long p = 2; p * p <= N; ++p) {
            // Ako je p označen kao prost, označi sve brojeve deljive sa p kao složene.
            if (is_prime[p]) {
                // Svi brojevi manji od p*p su već obrađeni.
                long start = p * p;

                // Na početku omp for petlje je implicitna barijera.
                #pragma omp for schedule(runtime)
                for (long m = start; m <= N; m += p) {
                    is_prime[m] = 0;
                } // Na kraju omp for petlje je implicitna barijera.
            }
        }
    }

    // Kraj merenja vremena.
    double t1 = omp_get_wtime();

    // Brojanje prostih brojeva nakon završetka sita.
    long count = 0;
    for (long i = 2; i <= N; ++i)
        if (is_prime[i]) ++count;

    // Vraćanje rezultata ako je traženo, ili oslobađanje memorije ako nije.
    if (out_count) *out_count = count;
    if (out_is_prime) *out_is_prime = is_prime;
    else free(is_prime);

    // Vraćanje izmerenog vremena.
    return t1 - t0;
}

/* Upisuje proste brojeve (na osnovu is_prime niza) u fajl, po jedan u redu. */
static void write_primes_to_file(const char *path, const unsigned char *is_prime, long N)
{
    // Otvara fajl za pisanje (binarni mod) i proverava uspešnost otvaranja.
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "Cannot open output file '%s'\n", path);
        exit(EXIT_FAILURE);
    }

    // Postavlja bafer veličine 1 MB za brže I/O operacije.
    setvbuf(fp, NULL, _IOFBF, 1 << 20);

    // Prolaz kroz sve brojeve od 2 do N i upisuje samo one koji su označeni kao prosti.
    for (long i = 2; i <= N; ++i)
        if (is_prime[i]) fprintf(fp, "%ld\n", i);

    // Zatvara fajl (flush bafera).
    fclose(fp);
}

/* Pretvara omp_sched_t u tekstualni naziv za ispis. */
static const char *sched_name(omp_sched_t s)
{
    // Izbor prema OpenMP enum vrednosti.
    switch (s) {
        case omp_sched_static:  return "static";  // Statičko raspoređivanje.
        case omp_sched_dynamic: return "dynamic"; // Dinamičko raspoređivanje.
        case omp_sched_guided:  return "guided";  // Vođeno raspoređivanje.
        case omp_sched_auto:    return "auto";    // Automatski izbor runtime-a.
        default:                return "unknown"; // Nepoznata vrednost.
    }
}

/* Struktura za čuvanje rezultata benchmark-a po rasporedu. */
typedef struct
{
    omp_sched_t sched;                          // Tip rasporeda (static/dynamic/guided).
    int chunk;                                  // Veličina chunk-a.
    double time_s;                              // Izmereno vreme u sekundama.
    long prime_count;                           // Broj pronađenih prostih.
} bench_result_t;

int main(int argc, char **argv)
{
    // Provera broja argumenata: očekuje se barem N i opciono ime fajla.
    if (argc < 2 || argc > 3) {
        fprintf(stderr, "Usage: %s <N> [output_file]\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Parsiranje N iz komandne linije i izbor izlaznog fajla.
    long N = parse_long(argv[1]);
    const char *out_path = (argc == 3) ? argv[2] : "primes.txt";

    // Isključivanje dinamičkog menjanja broja niti od strane OpenMP-a.
    omp_set_dynamic(0);

    // Ispis vrednosti N i maksimalnog broja niti koje OpenMP podržava na trenutnom sistemu.
    printf("N = %ld\n", N);
    printf("OpenMP max threads = %d\n\n", omp_get_max_threads());

    // Definisanje chunk veličina za različite rasporede.
    const int chunk_static  = 0;
    const int chunk_dynamic = 1024;
    const int chunk_guided  = 1024;

    // Niz za čuvanje rezultata tri rasporeda.
    bench_result_t results[3];

    // Postavljanje rasporeda i chunk-a za svaki od tri testirana rasporeda.
    results[0].sched = omp_sched_static;
    results[0].chunk = chunk_static;
    results[1].sched = omp_sched_dynamic;
    results[1].chunk = chunk_dynamic;
    results[2].sched = omp_sched_guided;
    results[2].chunk = chunk_guided;

    // Benchmarking: pokretanje sita za svaki raspored i čuvanje rezultata.
    for (int i = 0; i < 3; ++i) {
        // Postavljanje rasporeda za trenutni test: static, dynamic ili guided sa odgovarajućim chunk-om.
        omp_set_schedule(results[i].sched, results[i].chunk);

        // Pokretanje sita bez vraćanja niza i dobijanje vremena i broja prostih brojeva.
        long count = 0;
        double t = sieve_run(N, NULL, &count);

        // Čuvanje rezultata: vreme i broj prostih brojeva.
        results[i].time_s = t;
        results[i].prime_count = count;

        // Ispis rezultata za trenutni raspored: naziv, chunk, vreme i broj prostih brojeva.
        printf("Schedule: %-7s  chunk: %-5d  time: %.6f s  primes: %ld\n",
               sched_name(results[i].sched),
               results[i].chunk,
               t,
               count);
    }

    // Pronalaženje najbržeg rasporeda.
    int best = 0;
    for (int i = 1; i < 3; ++i)
        if (results[i].time_s < results[best].time_s) best = i;

    // Ispis najbržeg rasporeda.
    printf("\nBest schedule: %s (chunk=%d), time=%.6f s\n",
           sched_name(results[best].sched),
           results[best].chunk,
           results[best].time_s);

    // Postavljanje najboljeg rasporeda za finalno pokretanje koje vraća niz prostih brojeva.
    omp_set_schedule(results[best].sched, results[best].chunk);

    // Finalno pokretanje koje vraća niz is_prime i broj prostih brojeva.
    unsigned char *is_prime = NULL;
    long count = 0;
    double t_final = sieve_run(N, &is_prime, &count);

    // Ispis vremena i broja prostih brojeva za finalno pokretanje.
    printf("Final run: %.6f s, primes=%ld\n", t_final, count);

    // Upisivanje prostih brojeva u fajl i oslobađanje memorije.
    write_primes_to_file(out_path, is_prime, N);
    free(is_prime);

    printf("Primes written to: %s\n", out_path);
    return EXIT_SUCCESS;
}
