/* Program 1B (Version 1): Ring with BLOCKING MPI_Send / MPI_Recv
 * Each process sends its rank to (rank+1)%p and receives from (rank-1+p)%p.
 *
 * Deadlock avoidance: if every process called MPI_Send first, all would wait
 * for a matching receive and could deadlock (a circular wait). So we order the
 * calls: EVEN ranks send first then receive, ODD ranks receive first then send.
 * Note: this pairing needs an even p (4, 6, ...). With odd p, ranks p-1 and 0
 * are both even and both send first; small messages are usually buffered so it
 * still runs, but it is not guaranteed safe.
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define CHECK(call) do { int _rc = (call); if (_rc != MPI_SUCCESS) { \
    fprintf(stderr, "MPI error %d at line %d\n", _rc, __LINE__); \
    MPI_Abort(MPI_COMM_WORLD, _rc); } } while (0)

int main(int argc, char *argv[]) {
    int rank, p, sent, received;
    MPI_Status status;

    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) { fprintf(stderr, "MPI_Init failed\n"); return 1; }
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));   /* my rank */
    CHECK(MPI_Comm_size(MPI_COMM_WORLD, &p));      /* number of processes */

    int next = (rank + 1) % p;          /* neighbour I send to */
    int prev = (rank - 1 + p) % p;      /* neighbour I receive from */
    sent = rank;                        /* the value I send is my rank */

    if (rank % 2 == 0) {
        /* Even rank: send first, then receive */
        CHECK(MPI_Send(&sent, 1, MPI_INT, next, 0, MPI_COMM_WORLD));
        CHECK(MPI_Recv(&received, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, &status));
    } else {
        /* Odd rank: receive first, then send */
        CHECK(MPI_Recv(&received, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, &status));
        CHECK(MPI_Send(&sent, 1, MPI_INT, next, 0, MPI_COMM_WORLD));
    }

    printf("Process %d: sent %d to P%d, received %d from P%d\n",
           rank, sent, next, received, prev);

    MPI_Finalize();
    return 0;
}