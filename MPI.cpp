#include <cmath>
#include <iostream>
#include <vector>
#include <fstream>
#include <mpi.h>
#include <cstdlib>
#include <ctime>

using namespace std;

const int INF = 1e9; // Визначення значення нескінченності для представлення відсутності ребра

void floydWarshallSingleThread(vector<vector<int>>& graph) {
    int V = graph.size();
    // Проходимо по всіх вершинах
    for (int k = 0; k < V; k++) {
        // Проходимо по всіх парах вершин
        for (int i = 0; i < V; i++) {
            for (int j = 0; j < V; j++) {
                // Оновлюємо найкоротший шлях, якщо вершина k знаходиться на найкоротшому шляху від i до j
                if (graph[i][k] != INF && graph[k][j] != INF && graph[i][k] + graph[k][j] < graph[i][j]) {
                    graph[i][j] = graph[i][k] + graph[k][j];
                }
            }
        }
    }
}

void floydWarshall(vector<vector<int>>& graph, int rank, int size) {
    int V = graph.size(); // Кількість вершин у графі
    vector<int> kRow(V); // Зберігаємо k-ий рядок матриці відстаней
    vector rows(V, vector<int>(V)); // Тимчасове зберігання рядків матриці відстаней

    // Проходимо по всіх вершинах
    for(int k = 0; k < V; k++) {
        if(rank == 0) {
            kRow = graph[k]; // Отримуємо k-ий рядок матриці відстаней
            // Надсилаємо k-ий рядок усім іншим процесам
            for(int p = 1; p < size; p++) {
                MPI_Send(&kRow[0], V, MPI_INT, p, 0, MPI_COMM_WORLD);
            }
            // Надсилаємо відповідні рядки матриці відстаней кожному процесу
            for(int i = 0; i < V; i++) {
                if(i % size != 0) { // Пропускаємо відсилання самому собі
                    MPI_Send(&graph[i][0], V, MPI_INT, i % size, 0, MPI_COMM_WORLD);
                }
            }
            // Отримуємо оновлені рядки від інших процесів
            for(int i = 0; i < V; i++) {
                if(i % size != 0) { // Пропускаємо отримання від себе
                    MPI_Recv(&graph[i][0], V, MPI_INT, i % size, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
            // Оновлюємо власні рядки
            for(int i = 0; i < V; i += size) {
                for(int j = 0; j < V; j++) {
                    graph[i][j] = min(graph[i][j], graph[i][k] + kRow[j]);
                }
            }
        } else {
            // Отримуємо k-ий рядок від процесу 0
            MPI_Recv(&kRow[0], V, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // Проходимо по призначеним цьому процесу рядках
            for(int i = rank; i < V; i += size) {
                // Отримуємо відповідний рядок від процесу 0
                MPI_Recv(&rows[i][0], V, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // Оновлюємо рядки
                for(int j = 0; j < V; j++) {
                    rows[i][j] = min(rows[i][j], rows[i][k] + kRow[j]);
                }
                // Надсилаємо оновлений рядок назад процесу 0
                MPI_Send(&rows[i][0], V, MPI_INT, 0, 0, MPI_COMM_WORLD);
            }
        }
    }
}

// Функція для запису матриці у файл
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

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv); // Ініціалізація MPI
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Отримуємо ранг процесу
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Отримуємо загальну кількість процесів

    long long V = 1000; // Кількість вершин у графі

    // Ініціалізуємо матрицю відстаней значенням INF
    vector graph(V, vector(V, INF));
    vector<vector<int>> graphCopy;

    if (rank == 0) {
        srand(time(0) + rank);

        // Генеруємо випадкові ваги ребер
        for (long long i = 0; i < V; i++) {
            for (long long j = 0; j < V; j++) {
                if (i == j) {
                    graph[i][j] = 0; // Відстань від вершини до себе - 0
                } else {
                    int w = rand() % 1000 + 1; // Випадкова вага між 1 та 1000
                    graph[i][j] = w;
                }
            }
        }
        graphCopy = graph;
        writeMatrixToFile(graph, "initial_matrix_MPI.txt");

        double start = MPI_Wtime();

        floydWarshallSingleThread(graphCopy);

        double stop = MPI_Wtime();

        writeMatrixToFile(graphCopy, "single_threaded_shortest_paths_MPI.txt");
        cout << "Single-threaded execution time: " << stop - start << " seconds" << endl << endl;
    }

    double start = MPI_Wtime();

    floydWarshall(graph, rank, size);

    double stop = MPI_Wtime();

    if (rank == 0) {
        writeMatrixToFile(graph, "multi_threaded_shortest_paths_MPI.txt");
        cout << "Multi-threaded execution time: " << stop - start << " seconds" << endl << endl;
    }

    MPI_Finalize();
    return 0;
}
