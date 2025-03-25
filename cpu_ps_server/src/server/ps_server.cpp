#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include "mpi_utils.h"



class Parameter {
public:
    std::vector<float> data;
    int version = 0;
};

class PSService {
    std::unordered_map<std::string, Parameter> parameters_;
    
    void PullParameters(const std::string& param_name, std::vector<float>& params, int client_rank) {
        auto it = parameters_.find(param_name);
        if (it != parameters_.end()) {
            MPIUtils::SendParameters(it->second.data, client_rank, 0);
        }
    }
    
    void PushGradients(const std::string& param_name, const std::vector<float>& grads, int client_rank) {
        auto it = parameters_.find(param_name);
        if (it != parameters_.end()) {
            // 简单的梯度更新：参数 = 参数 - 学习率 * 梯度
            float learning_rate = 0.01f;
            for (size_t i = 0; i < it->second.data.size(); ++i) {
                it->second.data[i] -= learning_rate * grads[i];
            }
            it->second.version++;
        }
    }
};

void RunServer(int argc, char** argv) {
    MPIUtils::Initialize(&argc, &argv);
    
    int rank = MPIUtils::GetRank();
    int size = MPIUtils::GetSize();
    
    if (rank == 0) {
        std::cout << "Parameter server started with " << size << " processes" << std::endl;
        PSService service;
        
        while (true) {
            MPI_Status status;
            int flag;
            MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
            
            if (flag) {
                if (status.MPI_TAG == 0) { // Pull请求
                    std::string param_name;
                    MPI_Get_count(&status, MPI_CHAR, &flag);
                    param_name.resize(flag);
                    MPI_Recv(&param_name[0], flag, MPI_CHAR, status.MPI_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    std::vector<float> params;
                    auto it = service.parameters_.find(param_name);
                    if (it != service.parameters_.end()) {
                        params = it->second.data;
                    }
                    MPIUtils::SendParameters(params, status.MPI_SOURCE, 1);
                } else if (status.MPI_TAG == 1) { // Push请求
                    std::string param_name;
                    MPI_Get_count(&status, MPI_CHAR, &flag);
                    param_name.resize(flag);
                    MPI_Recv(&param_name[0], flag, MPI_CHAR, status.MPI_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    std::vector<float> grads;
                    MPI_Get_count(&status, MPI_FLOAT, &flag);
                    grads.resize(flag);
                    MPI_Recv(grads.data(), flag, MPI_FLOAT, status.MPI_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    service.PushGradients(param_name, grads, status.MPI_SOURCE);
                }
            }
        }
    }
    
    MPIUtils::Finalize();
}

int main(int argc, char** argv) {
    RunServer(argc, argv);
    return 0;
}