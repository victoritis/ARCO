///////////////////////////////////////////////////////////////////////////
// PROGRAMACIÓN EN CUDA C/C++
// Curso Basico
// PRACTICA 2: "LANZAMIENTO DE KERNEL"
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


///////////////////////////////////////////////////////////////////////////
// Prototipos de funciones
int elementosVector(int);

//////////////////////////////////////////////////////////////////////////
// FUNCION GLOBAL QUE TRABAJA CON EL DEVICE
__global__ void suma(int* vector1, int* vector2, int* resultado, int n) {
	//identificador de hilo
	int myID = threadIdx.x;
	//obtenemos el vector 2 invirtiendo el vector 1
	vector2[myID] = vector1[n - 1 - myID];
	//sumamos los dos vectores y escribimos el resultado
	resultado[myID] = vector1[myID] + vector2[myID];
}
/////////////////////////////////////////////////////////////////////////

//pedida de elementos del vector
int elementosVector(int MaxThreads) {
	int num;
	do {
		printf("Introduce el numero de elementos de los vectores a sumar \n");
		scanf("%d", &num);
		if (num > MaxThreads){
			printf("El numero de elementos introducido es mayor que el numero de hilos.\n");
		}
	} while (num > MaxThreads);
	getchar();// capturamos un INTRO para que no se cierre la consola de MSVS
	printf("\n");
	return num;
}

// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv)
{
	// buscando dispositivos
	int deviceCount;
	int MaxThreads;
	int n;
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
			MaxThreads = deviceProp.maxThreadsPerBlock;
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
			printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
			printf("*****************\n");
		}
	}
	n=elementosVector(MaxThreads);

	// declaraciones
	int* hst_vector;//, * hst_vector2, * hst_resultado
	int* dev_vector1, * dev_vector2, * dev_resultado;

	// reserva en el host
	hst_vector = (int*)malloc(n * sizeof(int));

	// reserva en el device 
	cudaMalloc((void**)&dev_vector1, n * sizeof(int));
	cudaMalloc((void**)&dev_vector2, n * sizeof(int));
	cudaMalloc((void**)& dev_resultado, n * sizeof(int));


	// creamos el primer vector en el host y el segundo en el device
	srand((int)time(NULL));
	for (int i = 0; i < n; i++)
	{
		hst_vector[i] = rand() % 10;
	}

	// copia de datos CPU -> GPU 
	cudaMemcpy(dev_vector1, hst_vector, n * sizeof(int), cudaMemcpyHostToDevice);

	//lanzamiento del Kernel
	printf("Vector de %d elementos \n", n);
	printf("Lanzamiento con %d bloque de %d hilos \n", 1, n);
	suma <<<1,n>>> (dev_vector1, dev_vector2, dev_resultado, n);

	//recogida de datos desde el device (GPU -> CPU) e impresion de resultados
	cudaMemcpy(hst_vector, dev_vector1, n * sizeof(int), cudaMemcpyDeviceToHost);
	printf("\nVECTOR 1: \n");
	for (int i = 0; i < n; i++) {
		printf("%2d ", hst_vector[i]);
	}
	printf("\n");

	cudaMemcpy(hst_vector, dev_vector2, n * sizeof(int), cudaMemcpyDeviceToHost);
	printf("\nVECTOR 2: \n");
	for (int i = 0; i < n; i++) {
		printf("%2d ", hst_vector[i]);
	}
	printf("\n");

	cudaMemcpy(hst_vector, dev_resultado, n * sizeof(int), cudaMemcpyDeviceToHost);
	printf("\nVECTOR RESULTADO: \n");
	for (int i = 0; i < n; i++) {
		printf("%2d ", hst_vector[i]);
	}

	// salida del programa
	time_t fecha;
	time(&fecha);
	printf("\n*****************\n");
	printf("\nPrograma ejecutado el: %s\n", ctime(&fecha));

	// capturamos un INTRO para que no se cierre la consola de MSVS
	printf("<pulsa [INTRO] para finalizar>");
	getchar();
	return 0;
}
///////////////////////////////////////////////////////////////////////////