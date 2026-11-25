/* Program 1A: Point-to-point message chain (needs exactly 4 processes)
 * P0: x = 42 -> P1: *2 -> P2: +10 -> P3: prints final value
 * Expected chain: 42 -> 84 -> 94
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/* Error-handling helper: abort the whole job if an MPI call fails */
#define CHECK(call) do { int _rc = (call); if (_rc != MPI_SUCCESS) { \
    fprintf(stderr, "MPI error %d at line %d\n", _rc, __LINE__); \
    MPI_Abort(MPI_COMM_WORLD, _rc); } } while (0)

#define TAG_01 1   /* tag for P0 -> P1 */
#define TAG_12 2   /* tag for P1 -> P2 */
#define TAG_23 3   /* tag for P2 -> P3 */

int main(int argc, char *argv[]) {
    int rank, size, x;
    MPI_Status status;

    /* Start the MPI environment */
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Init failed\n");
        return 1;
    }
    /* Return error codes instead of aborting, so CHECK can handle them */
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    /* Get this process's id (rank) */
    CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    /* Get the total number of processes */
    CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));

    /* The program is defined for exactly 4 processes */
    if (size != 4) {
        if (rank == 0) fprintf(stderr, "Error: run with exactly 4 processes (mpirun -np 4)\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        x = 42;
        /* Send x to P1 (1 int, tag TAG_01) */
        CHECK(MPI_Send(&x, 1, MPI_INT, 1, TAG_01, MPI_COMM_WORLD));
        printf("P0: initialized x = %d, sent %d to P1\n", x, x);
    } else if (rank == 1) {
        /* Receive from P0 with matching tag */
        CHECK(MPI_Recv(&x, 1, MPI_INT, 0, TAG_01, MPI_COMM_WORLD, &status));
        int recvd = x;
        x = x * 2;
        /* Send the doubled value to P2 */
        CHECK(MPI_Send(&x, 1, MPI_INT, 2, TAG_12, MPI_COMM_WORLD));
        printf("P1: received %d from P0, multiplied by 2, sent %d to P2\n", recvd, x);
    } else if (rank == 2) {
        /* Receive from P1 */
        CHECK(MPI_Recv(&x, 1, MPI_INT, 1, TAG_12, MPI_COMM_WORLD, &status));
        int recvd = x;
        x = x + 10;
        /* Send the incremented value to P3 */
        CHECK(MPI_Send(&x, 1, MPI_INT, 3, TAG_23, MPI_COMM_WORLD));
        printf("P2: received %d from P1, added 10, sent %d to P3\n", recvd, x);
    } else { /* rank == 3 */
        /* Receive from P2 */
        CHECK(MPI_Recv(&x, 1, MPI_INT, 2, TAG_23, MPI_COMM_WORLD, &status));
        printf("P3: received %d from P2. Final value = %d\n", x, x);
    }

    /* Shut down MPI */
    MPI_Finalize();
    return 0;
}