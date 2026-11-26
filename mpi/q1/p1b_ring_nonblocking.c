/* Program 1B (Version 2): Ring with NON-BLOCKING MPI_Isend / MPI_Irecv
 * Both operations are only *posted*, so no process waits while it still has
 * its other operation to post -> no deadlock regardless of order or p.
 * MPI_Waitall then completes both requests.
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define CHECK(call) do { int _rc = (call); if (_rc != MPI_SUCCESS) { \
    fprintf(stderr, "MPI error %d at line %d\n", _rc, __LINE__); \
    MPI_Abort(MPI_COMM_WORLD, _rc); } } while (0)

int main(int argc, char *argv[]) {
    int rank, p, sent, received;
    MPI_Request reqs[2];
    MPI_Status stats[2];

    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) { fprintf(stderr, "MPI_Init failed\n"); return 1; }
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CHECK(MPI_Comm_size(MPI_COMM_WORLD, &p));

    int next = (rank + 1) % p;
    int prev = (rank - 1 + p) % p;
    sent = rank;

    /* Post the receive from prev (returns immediately) */
    CHECK(MPI_Irecv(&received, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, &reqs[0]));
    /* Post the send to next (returns immediately) */
    CHECK(MPI_Isend(&sent, 1, MPI_INT, next, 0, MPI_COMM_WORLD, &reqs[1]));
    /* Wait until BOTH operations are complete; only then are buffers safe to use */
    CHECK(MPI_Waitall(2, reqs, stats));

    printf("Process %d: sent %d to P%d, received %d from P%d\n",
           rank, sent, next, received, prev);

    MPI_Finalize();
    return 0;
}