/* Program 2C: Parallel matrix-vector multiplication y = A*x
 * A is M x M (M = 8) with A[i][j] = i + j ; x[i] = i + 1.
 * MPI_Bcast sends x to everyone, MPI_Scatter gives each process M/p rows of A,
 * each process multiplies its rows by x, MPI_Gather collects the pieces of y.
 * Root also computes y serially and compares.
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define M 8

#define CHECK(call) do { int _rc = (call); if (_rc != MPI_SUCCESS) { \
    fprintf(stderr, "MPI error %d at line %d\n", _rc, __LINE__); \
    MPI_Abort(MPI_COMM_WORLD, _rc); } } while (0)

int main(int argc, char *argv[]) {
    int rank, p;
    int A[M * M];                    /* matrix stored row by row (only filled at root) */
    int x[M];                        /* vector, needed by every process */
    int y[M], y_serial[M];

    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) { fprintf(stderr, "MPI_Init failed\n"); return 1; }
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CHECK(MPI_Comm_size(MPI_COMM_WORLD, &p));

    /* Rows must divide evenly among the processes */
    if (M % p != 0) {
        if (rank == 0) fprintf(stderr, "Error: M=%d is not divisible by p=%d\n", M, p);
        MPI_Finalize();
        return 1;
    }
    int rows = M / p;                /* rows of A handled by each process */

    /* Root initializes A[i][j] = i + j and x[i] = i + 1 */
    if (rank == 0) {
        for (int i = 0; i < M; i++) {
            x[i] = i + 1;
            for (int j = 0; j < M; j++) A[i * M + j] = i + j;
        }
    }

    /* Send the whole vector x from root to every process */
    CHECK(MPI_Bcast(x, M, MPI_INT, 0, MPI_COMM_WORLD));

    /* Give each process its block of 'rows' consecutive rows (rows*M ints) */
    int local_A[M * M];              /* big enough for any p; only rows*M used */
    CHECK(MPI_Scatter(A, rows * M, MPI_INT, local_A, rows * M, MPI_INT, 0, MPI_COMM_WORLD));

    /* Local multiply: each of my rows dot x */
    int local_y[M];
    for (int r = 0; r < rows; r++) {
        local_y[r] = 0;
        for (int j = 0; j < M; j++) local_y[r] += local_A[r * M + j] * x[j];
    }

    /* Collect all local result pieces into y at root (in rank order) */
    CHECK(MPI_Gather(local_y, rows, MPI_INT, y, rows, MPI_INT, 0, MPI_COMM_WORLD));

    if (rank == 0) {
        /* Serial reference result */
        for (int i = 0; i < M; i++) {
            y_serial[i] = 0;
            for (int j = 0; j < M; j++) y_serial[i] += A[i * M + j] * x[j];
        }
        printf("p = %d, result y = [", p);
        for (int i = 0; i < M; i++) printf("%d%s", y[i], i < M - 1 ? ", " : "");
        printf("]\n");
        printf("serial  y = [");
        for (int i = 0; i < M; i++) printf("%d%s", y_serial[i], i < M - 1 ? ", " : "");
        printf("]\n");
        int same = 1;
        for (int i = 0; i < M; i++) if (y[i] != y_serial[i]) same = 0;
        printf("Parallel result matches serial: %s\n", same ? "YES" : "NO");
    }

    MPI_Finalize();
    return 0;
}