/* Program 2B: Distributed prime counter using MPI_Bcast and MPI_Gather
 * Usage: mpirun -np 4 ./p2b_primes 2 100000
 * Root reads a and b from the command line and broadcasts both in ONE
 * MPI_Bcast (packed in a 2-element array). Process r checks
 * [a + r*chunk, a + (r+1)*chunk - 1]; the last process also takes any
 * leftover numbers up to b. Local counts are collected with MPI_Gather.
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define CHECK(call) do { int _rc = (call); if (_rc != MPI_SUCCESS) { \
    fprintf(stderr, "MPI error %d at line %d\n", _rc, __LINE__); \
    MPI_Abort(MPI_COMM_WORLD, _rc); } } while (0)

/* Trial-division primality test */
static int is_prime(int n) {
    if (n < 2) return 0;
    if (n % 2 == 0) return n == 2;
    for (int d = 3; (long long)d * d <= n; d += 2)
        if (n % d == 0) return 0;
    return 1;
}

int main(int argc, char *argv[]) {
    int rank, p;
    int ab[2];                       /* ab[0] = a, ab[1] = b */

    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) { fprintf(stderr, "MPI_Init failed\n"); return 1; }
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CHECK(MPI_Comm_size(MPI_COMM_WORLD, &p));

    /* Only root reads the command-line arguments */
    if (rank == 0) {
        if (argc != 3) {
            fprintf(stderr, "Usage: mpirun -np <p> %s <a> <b>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        ab[0] = atoi(argv[1]);
        ab[1] = atoi(argv[2]);
        if (ab[0] > ab[1]) {
            fprintf(stderr, "Error: a must be <= b\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    /* ONE broadcast sends both values (2-element array) to all processes */
    CHECK(MPI_Bcast(ab, 2, MPI_INT, 0, MPI_COMM_WORLD));
    int a = ab[0], b = ab[1];

    /* Split [a, b] into p sub-ranges */
    int total_numbers = b - a + 1;
    int chunk = total_numbers / p;
    int lo = a + rank * chunk;
    int hi = (rank == p - 1) ? b : lo + chunk - 1;   /* last rank takes the remainder */

    /* Count primes in my sub-range */
    int local_count = 0;
    for (int n = lo; n <= hi; n++)
        if (is_prime(n)) local_count++;

    /* Collect one count from every process into counts[] at root */
    int *counts = NULL;
    if (rank == 0) counts = malloc(p * sizeof(int));
    CHECK(MPI_Gather(&local_count, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD));

    if (rank == 0) {
        long total = 0;
        for (int r = 0; r < p; r++) {
            int rlo = a + r * chunk;
            int rhi = (r == p - 1) ? b : rlo + chunk - 1;
            printf("Process %d: range [%d, %d] -> %d primes\n", r, rlo, rhi, counts[r]);
            total += counts[r];
        }
        printf("Total primes between %d and %d: %ld\n", a, b, total);
        free(counts);
    }

    MPI_Finalize();
    return 0;
}