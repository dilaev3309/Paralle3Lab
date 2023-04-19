/*
Программа решает уравнение Пуассона на прямоугольной области с помощью метода Якоби.Ускорение происодит с помощью OpenACC.
Сначала задаются параметры модли, после выделяется память.
Алгоритм выполняется пока значение переменной error не станет меньше заданного.
*/
#include "/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/math_libs/11.8/targets/x86_64-linux/include/cublas_v2.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define BILLION 1000000000

int main(int argc, char** argv) {

    struct timespec start, stop;
    clock_gettime(CLOCK_REALTIME, &start);

    int size, iter_max;
    double tol;
    sscanf(argv[1], "%d", &size);
    sscanf(argv[2], "%d", &iter_max);
    sscanf(argv[3], "%lf", &tol);
    double *buf;
    cublasHandle_t handle;
    cublasCreate(&handle);
    const double alpha = -1;
    double step1 = 10.0 / (size - 1);
    double* A = (double*)calloc(size*size, sizeof(double));
    double* Anew = (double*)calloc(size*size, sizeof(double));
    double x1 = 10.0;
    double x2 = 20.0;
    double y1 = 20.0;
    double y2 = 30.0;
    //Заполнение угловых значений сетки
    A[0] = Anew[0] = x1;
    A[size] = Anew[size] = x2;
    A[size * (size - 1) + 1] = Anew[size * (size - 1) + 1] = y1;
    A[size * size] = Anew[size * size] = y2;
//Создание копии массивов на устройстве
#pragma acc enter data create(A[0:size*size], Anew[0:size*size]) copyin(size, step1)
#pragma acc kernels
    {
#pragma acc loop independent
        //Инициализация граничных значений массивов
        for (int i = 0; i < size; i++) {
            A[i*size] = Anew[i*size] = x1 + i * step1;
            A[i] = Anew[i] = x1 + i * step1;
            A[(size - 1) * size + i] = Anew[(size - 1) * size + i] = y1 + i * step1;
            A[i * size + (size - 1)] = Anew[i * size + (size - 1)] = x2 + i * step1;        }
    }

    int itter = 0;
    double error = 1.0;
    {
    while (itter < iter_max && error > tol) {
        itter++;
        if (itter % 100 == 0 || itter == 1) {
            //Снова создаем копии массивов
#pragma acc data present(A[0:size*size], Anew[0:size*size])
#pragma acc kernels async(1)
            {
#pragma acc loop independent collapse(2)
                for (int i = 1; i < size - 1; i++) {
                    for (int j = 1; j < size - 1; j++) {
                        Anew[i * size + j] =
                                0.25 * (A[(i + 1) * size + j] + A[(i - 1) * size + j] + A[i * size + j - 1] + A[i * size + j + 1]);
                    }
                }
            }
            int id = 0;
            //Ожидание завершения асинхронных операций
#pragma acc wait
#pragma acc host_data use_device(A, Anew)
            {
                cublasDaxpy(handle, size * size, &alpha, Anew, 1, A, 1);
                cublasIdamax(handle, size * size, A, 1, &id);

            }
#pragma acc update self(A[id-1:1])
            error = fabs(A[id - 1]);
#pragma acc host_data use_device(A, Anew)
            cublasDcopy(handle, size * size, Anew, 1, A, 1);

        } else {
#pragma acc data present(A[0:size*size], Anew[0:size*size])
#pragma acc kernels async(1)
            {
#pragma acc loop independent collapse(2)
                for (int i = 1; i < size - 1; i++) {
                    for (int j = 1; j < size - 1; j++) {
                        Anew[i * size + j] =
                                0.25 * (A[(i + 1) * size + j] + A[(i - 1) * size + j] + A[i * size + j - 1] + A[i * size + j + 1]);
                    }
                }
            }
        }
        buf = A;
        A = Anew;
        Anew = buf;

        if (itter % 100 == 0 || itter == 1)
#pragma acc wait(1)
            printf("%d %e\n", itter, error);
        //Отслеживаем прогресс вычислений

    }
}
    //выводим результат
    clock_gettime(CLOCK_REALTIME, &stop);
    double delta = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec)/(double)BILLION;

    printf("%d\n", itter);
    printf("%0.6lf\n", error);
    printf("time %lf\n", delta);

    cublasDestroy(handle);
    return 0;
}
