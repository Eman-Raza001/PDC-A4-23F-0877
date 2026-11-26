/* Program 2A: Parallel array sum with MPI_Scatter and MPI_Reduce
 * Root creates arr[i] = i+1 for N = 1,000,000 elements.
 * MPI_Scatter gives each process an equal chunk, each process sums its chunk,
 * and MPI_Reduce(MPI_SUM) combines the partial sums at root.
 * Root checks the result against the formula N*(N+1)/2 and prints PASS/FAIL.
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 1000000

#define CHECK(call) do { int _rc = (call); if (_rc != MPI_SUCCESS) { \
    fprintf(stderr, "MPI error %d at line %d\n", _rc, __LINE__); \
    MPI_Abort(MPI_COMM_WORLD, _rc); } } while (0)

int main(int argc, char *argv[]) {
    int rank, p;
    int *arr = NULL;                 /* full array, only allocated at root */
    long long local_sum = 0, global_sum = 0;

    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) { fprintf(stderr, "MPI_Init failed\n"); return 1; }
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CHECK(MPI_Comm_size(MPI_COMM_WORLD, &p));

    /* MPI_Scatter sends equal chunks, so N must divide evenly by p */
    if (N % p != 0) {
        if (rank == 0) fprintf(stderr, "Error: N=%d is not divisible by p=%d\n", N, p);
        MPI_Finalize();
        return 1;
    }
    int chunk = N / p;               /* elements per process */
    int *local = malloc(chunk * sizeof(int));
    if (!local) { fprintf(stderr, "malloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

    /* Root builds the array arr[i] = i + 1 */
    if (rank == 0) {
        arr = malloc(N * sizeof(int));
        if (!arr) { fprintf(stderr, "malloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        for (int i = 0; i < N; i++) arr[i] = i + 1;
    }

    /* Make all processes start the timed section together */
    CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double t0 = MPI_Wtime();

    /* Distribute equal chunks of arr to every process (including root) */
    CHECK(MPI_Scatter(arr, chunk, MPI_INT, local, chunk, MPI_INT, 0, MPI_COMM_WORLD));

    /* Each process sums its own chunk (long long: total ~5e11 overflows int) */
    for (int i = 0; i < chunk; i++) local_sum += local[i];

    /* Add all partial sums together; result lands at root only */
    CHECK(MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD));

    double t1 = MPI_Wtime();         /* end of the parallel section */

    if (rank == 0) {
        long long expected = (long long)N * (N + 1) / 2;   /* N(N+1)/2 */
        printf("Processes p = %d, N = %d\n", p, N);
        printf("Global sum = %lld, expected = %lld -> %s\n",
               global_sum, expected, global_sum == expected ? "PASS" : "FAIL");
        printf("Elapsed time (scatter + sum + reduce) = %.6f seconds\n", t1 - t0);
        free(arr);
    }

    free(local);
    MPI_Finalize();
    return 0;
}