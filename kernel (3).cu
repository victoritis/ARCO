// includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h> 
// declaracion de funciones
// GLOBAL: funcion llamada desde el host y ejecutada en el device (kernel)
__global__ void par_gpu(int* A, int* B)
{
	// indice de fila
	int fila = threadIdx.y;
	// indice de columna
	int columna = threadIdx.x;

	// Calculamos la suma:
	// C[fila][columna] = A[fila][columna] + B[fila][columna]
	// Para ello convertimos los indices de 'fila' y 'columna' en un indice lineal
	int myID = columna + fila * blockDim.x;
	if (columna % 2 == 0) {
		B[myID] = A[myID];
	}
	else {
		B[myID] = 0;
	}
}
__host__ void propiedades_Device(int deviceID)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceID);
	// calculo del numero de cores (SP)
	int cudaCores = 0;
	int SM = deviceProp.multiProcessorCount;
	int major = deviceProp.major;
	int minor = deviceProp.minor;
	const char* archName;
	switch (major)
	{
	case 1:
		//TESLA 
		archName = "TESLA";
		cudaCores = 8;
		break;
	case 2:
		//FERMI
		archName = "FERMI";
		if (minor == 0)
			cudaCores = 32;
		else
			cudaCores = 48;
		break;
	case 3:
		//KEPLER
		archName = "KEPLER";
		cudaCores = 192;
		break;
	case 5:
		//MAXWELL
		archName = "MAXWELL";
		cudaCores = 128;
		break;
	case 6:
		//PASCAL
		archName = "PASCAL";
		cudaCores = 64;
		break;
	case 7:
		//VOLTA(7.0) //TURING(7.5) 
		cudaCores = 64;
		if (minor == 0)
			archName = "VOLTA";
		else
			archName = "TURING";
		break;
	case 8:
		// AMPERE
		archName = "AMPERE";
		cudaCores = 64;
		break;
	default:
		//ARQUITECTURA DESCONOCIDA
		archName = "DESCONOCIDA";
	}
	int rtV;
	cudaRuntimeGetVersion(&rtV);
	// presentacion de propiedades
	printf("***************************************************\n");
	printf("DEVICE %d: %s\n", deviceID, deviceProp.name);
	printf("***************************************************\n");
	printf("> CUDA Toolkit \t\t\t\t: %d.%d\n", rtV / 1000, (rtV % 1000) / 10);
	printf("> Arquitectura CUDA \t\t\t: %s\n", archName);
	printf("> Capacidad de Computo \t\t\t: %d.%d\n", major, minor);
	printf("> No. MultiProcesadores \t\t: %d\n", SM);
	printf("> No. Nucleos CUDA (%dx%d) \t\t: %d\n", cudaCores, SM, cudaCores * SM);
	printf("> Memoria Global (total) \t\t: %u MiB\n",
		deviceProp.totalGlobalMem / (1024 * 1024));
	printf(">No. maximo de Hilos (por bloque):\t: %d\n", deviceProp.maxThreadsPerBlock);
	printf(" [x -> %d]\n [y -> %d]\n [z -> %d]\n", deviceProp.maxThreadsDim[0],
		deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
	printf(">No. maximo de Bloques (por eje):\n");
	printf(" [x -> %d]\n [y -> %d]\n [z -> %d]\n", deviceProp.maxGridSize[0],
		deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);

	printf("***************************************************\n");
}
// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv)
{
	// declaraciones
	int* hst_A, * hst_B;
	int* dev_A, * dev_B;
	int fila = 0;
	int columna = 0;


	printf("Kernel de 1 bloque y 126 hilos");
	fila = 7;
	columna = 21;

	printf("\n%d\n", fila);
	printf("\n%d\n",columna);

	// reserva en el host
	hst_A = (int*)malloc(fila * columna * sizeof(int));
	hst_B = (int*)malloc(fila * columna * sizeof(int));
	
	// reserva en el device
	cudaMalloc((void**)&dev_A, fila * columna * sizeof(int));
	cudaMalloc((void**)&dev_B, fila * columna * sizeof(int));
	
	// incializacion
	srand((int)time(NULL));
	for (int i = 0; i < fila * columna; i++)
	{
		hst_A[i]= rand() % 9;
		hst_B[i] = 0;
	}
	// copia de datos
	cudaMemcpy(dev_A, hst_A, fila * columna * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, hst_B, fila * columna * sizeof(int), cudaMemcpyHostToDevice);
	
	// dimensiones del kernel
	dim3 hilosB(columna, fila);
	
	// llamada al kernel bidimensional de NxN hilos
	par_gpu <<<1, hilosB >>> (dev_A, dev_B);
	
	// recogida de datos
	cudaMemcpy(hst_B, dev_B, fila * columna * sizeof(int), cudaMemcpyDeviceToHost);
	
	// impresion de resultados
	printf("MATRIZ A:\n");
	for (int i = 0; i < fila; i++)
	{
		for (int j = 0; j < columna; j++)
		{
			printf("%d ", hst_A[j + i * columna]);
		}
		printf("\n");
	}
	printf("MATRIZ B:\n");
	for (int i = 0; i < fila; i++)
	{
		for (int j = 0; j < columna; j++)
		{
			printf("%d ", hst_B[j + i * columna]);
		}
		printf("\n");
	}

	
	// salida
	time_t fecha;
	time(&fecha);
	printf("***************************************************\n");
	printf("Programa ejecutado el: %s\n", ctime(&fecha));
	printf("<pulsa [INTRO] para finalizar>");
	getchar();
	return 0;
}