/* Program 1C: All-to-all exchange using only MPI_Send / MPI_Recv
 * local_data[i] = rank*10 + i ; process r sends local_data[j] to process j
 * and stores what it receives from j in recv_buf[j].
 * Therefore on process j: recv_buf[i] should equal i*10 + j.
 *
 * Deadlock-free strategy: we handle the partners j = 0..p-1 in the same
 * increasing order on every process. For each PAIR (rank, j) the LOWER rank
 * sends first and then receives, the HIGHER rank receives first and then
 * sends. So the two sides of every pair are always complementary (one is
 * sending while the other is receiving), and no circular wait can form,
 * even if MPI_Send blocks until the receive is posted.
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define CHECK(call) do { int _rc = (call); if (_rc != MPI_SUCCESS) { \
    fprintf(stderr, "MPI error %d at line %d\n", _rc, __LINE__); \
    MPI_Abort(MPI_COMM_WORLD, _rc); } } while (0)

int main(int argc, char *argv[]) {
    int rank, p;
    MPI_Status status;

    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) { fprintf(stderr, "MPI_Init failed\n"); return 1; }
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CHECK(MPI_Comm_size(MPI_COMM_WORLD, &p));

    int *local_data = malloc(p * sizeof(int));
    int *recv_buf   = malloc(p * sizeof(int));
    if (!local_data || !recv_buf) { fprintf(stderr, "malloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

    for (int i = 0; i < p; i++) local_data[i] = rank * 10 + i;
    recv_buf[rank] = local_data[rank];   /* own element: no message needed */

    for (int j = 0; j < p; j++) {
        if (j == rank) continue;
        if (rank < j) {
            /* lower rank: send first, then receive */
            CHECK(MPI_Send(&local_data[j], 1, MPI_INT, j, 0, MPI_COMM_WORLD));
            CHECK(MPI_Recv(&recv_buf[j], 1, MPI_INT, j, 0, MPI_COMM_WORLD, &status));
        } else {
            /* higher rank: receive first, then send */
            CHECK(MPI_Recv(&recv_buf[j], 1, MPI_INT, j, 0, MPI_COMM_WORLD, &status));
            CHECK(MPI_Send(&local_data[j], 1, MPI_INT, j, 0, MPI_COMM_WORLD));
        }
    }

    /* Verify: recv_buf[i] must equal i*10 + rank */
    int ok = 1;
    for (int i = 0; i < p; i++)
        if (recv_buf[i] != i * 10 + rank) ok = 0;

    /* Print the buffer (one single printf per process keeps lines unmixed) */
    char line[256]; int off = 0;
    off += snprintf(line + off, sizeof line - off, "Process %d recv_buf = [", rank);
    for (int i = 0; i < p; i++)
        off += snprintf(line + off, sizeof line - off, "%d%s", recv_buf[i], i < p - 1 ? ", " : "");
    snprintf(line + off, sizeof line - off, "]  verification: %s", ok ? "OK" : "FAILED");
    printf("%s\n", line);

    free(local_data);
    free(recv_buf);
    MPI_Finalize();
    return 0;
}