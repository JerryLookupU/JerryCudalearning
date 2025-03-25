#pragma once

#include <mpi.h>
#include <string>
#include <vector>

class MPIUtils {
public:
    static void Initialize(int* argc, char*** argv) {
        MPI_Init(argc, argv);
    }

    static void Finalize() {
        MPI_Finalize();
    }

    static int GetRank() {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        return rank;
    }

    static int GetSize() {
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        return size;
    }

    static void SendParameters(const std::vector<float>& params, int dest, int tag) {
        MPI_Send(params.data(), params.size(), MPI_FLOAT, dest, tag, MPI_COMM_WORLD);
    }

    static void RecvParameters(std::vector<float>& params, int src, int tag) {
        MPI_Status status;
        MPI_Recv(params.data(), params.size(), MPI_FLOAT, src, tag, MPI_COMM_WORLD, &status);
    }

    static void BroadcastParameters(std::vector<float>& params, int root) {
        MPI_Bcast(params.data(), params.size(), MPI_FLOAT, root, MPI_COMM_WORLD);
    }
};