#include <iostream>
#include <vector>
#include <fstream>
#include <omp.h>
#include <cstdlib>
#include <ctime>
using namespace std;

const int INF = 1e9; // Велике значення для відсутності ребра між вершинами

void floydWarshall(vector<vector<int>>& graph) {
    int V = graph.size();
#pragma omp parallel num_threads(8)
    {
        for (int k = 0; k < V; k++) {
#pragma omp for collapse(2)
            for (int i = 0; i < V; i++) {
                for (int j = 0; j < V; j++) {
                    if (graph[i][k] != INF && graph[k][j] != INF && graph[i][k] + graph[k][j] < graph[i][j]) {
                        graph[i][j] = graph[i][k] + graph[k][j];
                    }
                }
            }
        }
    }
}

void floydWarshallSingleThread(vector<vector<int>>& graph) {
    int V = graph.size();
    for (int k = 0; k < V; k++) {
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                if (graph[i][k] != INF && graph[k][j] != INF && graph[i][k] + graph[k][j] < graph[i][j]) {
                    graph[i][j] = graph[i][k] + graph[k][j];
                }
            }
        }
    }
}

void writeMatrixToFile(const vector<vector<int>>& matrix, const string& filename) {
    ofstream file(filename);
    if (file.is_open()) {
        for (const auto& row : matrix) {
            for (int value : row) {
                file << value << " ";
            }
            file << endl;
        }
        file.close();
        cout << "Matrix has been written to " << filename << endl << endl;
    } else {
        cout << "Unable to open file " << filename << " for writing." << endl;
    }
}

int main() {
    long long V = 1000; // Кількість вершин

    srand(time(0));
    vector graph(V, vector(V, INF));
    for (long long i = 0; i < V; i++) {
        for (long long j = 0; j < V; j++) {
            if (i == j) {
                graph[i][j] = 0; // Діагональні елементи рівні 0
            } else {
                int w = rand() % 1000 + 1; // Випадкова вага ребра від 1 до 1000
                graph[i][j] = w;
            }
        }
    }

    vector<vector<int>> graphCopy = graph;
    writeMatrixToFile(graph, "initial_matrix_OpenMP.txt");

    auto start = omp_get_wtime();

    floydWarshallSingleThread(graphCopy);

    writeMatrixToFile(graphCopy, "single_thread_shortest_paths_OpenMP.txt");

    auto stop = omp_get_wtime();

    auto durationSingleThread = stop - start;
    cout << "Single-threaded execution time: " << durationSingleThread << " seconds" << endl << endl;

    start = omp_get_wtime();

    floydWarshall(graph);

    writeMatrixToFile(graphCopy, "multi_thread_shortest_paths_OpenMP.txt");

    stop = omp_get_wtime();

    auto durationMultiThread = stop - start;
    cout << "Multi-threaded execution time: " << durationMultiThread << " seconds" << endl;

    return 0;
}
