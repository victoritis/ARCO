///////////////////////////////////////////////////////////////////////////
// PROGRAMACIÓN EN CUDA C/C++
// Curso Basico
// PRACTICA 1: "Dispositivos CUDA + NÚMEROS ALEATORIOS"
//
// SEPTIEMBRE 2021
///////////////////////////////////////////////////////////////////////////
// includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
///////////////////////////////////////////////////////////////////////////
// defines
#define KIBI (1<<10) // KiB = 2^10
#define MEBI (1<<20) // MiB = 2^20
#define N 12
///////////////////////////////////////////////////////////////////////////
// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv)
{
	// declaraciones
	float* hst_A, * hst_B;
	float* dev_A, * dev_B;

	// buscando dispositivos
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0)
	{
		printf("!!!!!No se han encontrado dispositivos CUDA!!!!!\n");
		printf("<pulsa [INTRO] para finalizar>");
		getchar();
		return 1;
	}
	else
	{
		printf("Se han encontrado <%d> dispositivos CUDA:\n", deviceCount);
		for (int deviceID = 0; deviceID < deviceCount; deviceID++)
		{
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, deviceID);
			// calculo del numero de cores (SP)
			int cudaCores = 0;
			int SM = deviceProp.multiProcessorCount;
			int major = deviceProp.major;
			int minor = deviceProp.minor;
			switch (major)
			{
			case 1:
				//TESLA
				cudaCores = 8;
				break;
			case 2:
				//FERMI
				if (minor == 0)
					cudaCores = 32;
				else
					cudaCores = 48;
				break;
			case 3:
				//KEPLER
				cudaCores = 192;
				break;
			case 5:
				//MAXWELL
				cudaCores = 128;
				break;
			case 6:
				//PASCAL
				cudaCores = 64;
				break;
			case 7:
				//VOLTA (7.0) TURING (7.5)
				cudaCores = 64;
				break;
			case 8:
				//AMPERE
				cudaCores = 64;
				break;
			default:
				//ARQUITECTURA DESCONOCIDA
				cudaCores = 0;
				printf("!!!!!dispositivo desconocido!!!!!\n");
			}
			// presentacion de propiedades
			printf("*****************\n");
			printf("DEVICE %d: %s\n", deviceID, deviceProp.name);
			printf("*****************\n");
			printf("> Capacidad de Computo            \t: %d.%d\n", major, minor);
			printf("> No. de MultiProcesadores        \t: %d \n", SM);
			printf("> No. de CUDA Cores (%dx%d)       \t: %d \n", cudaCores, SM, cudaCores * SM);
			printf("> Memoria Global (total)          \t: %zu MiB\n", deviceProp.totalGlobalMem / MEBI);
			printf("> Memoria Compartida (por bloque) \t: %zu KiB\n", deviceProp.sharedMemPerBlock / KIBI);
			printf("> Memoria Constante  (total)      \t: %zu KiB\n", deviceProp.totalConstMem / KIBI);
			printf("*****************\n");
		}
	}

	// reserva en el host
		hst_A = (float*)malloc(N * sizeof(float));
		hst_B = (float*)malloc(N * sizeof(float));

	// reserva en el device
		cudaMalloc((void**)&dev_A, N * sizeof(float));
		cudaMalloc((void**)&dev_B, N * sizeof(float));

	// inicializacion
		srand ( (int)time(NULL) );
		for (int i = 0; i < N; i++)
		{
			hst_A[i] = (float)rand() / RAND_MAX;
			hst_B[i] = 0;
		}

	// copia de datos
		cudaMemcpy(dev_A, hst_A, N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_B, dev_A, N * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(hst_B, dev_B, N * sizeof(float), cudaMemcpyDeviceToHost);

	// muestra de resultados
		printf("ENTRADA (hst_A):\n");
		for (int i = 0; i < N; i++)
			printf("%.2f ", hst_A[i]);
		printf("\n");
		printf("SALIDA (hst_B):\n");
		for (int i = 0; i < N; i++)
			printf("%.2f ", hst_B[i]);
		printf("\n");

	// liberacion de recursos
		cudaFree(dev_A);
		cudaFree(dev_B);

	// salida del programa
		time_t fecha;
		time(&fecha);
		printf("*****************\n");
		printf("Programa ejecutado el: %s\n", ctime(&fecha));

	// capturamos un INTRO para que no se cierre la consola de MSVS
		printf("<pulsa [INTRO] para finalizar>");
		getchar();
	return 0;
}
///////////////////////////////////////////////////////////////////////////