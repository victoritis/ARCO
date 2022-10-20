// includes 
#include <stdio.h> 
#include <stdlib.h> 
#include <time.h> 
#include <cuda_runtime.h> 

#define BLOCK 10

// declaracion de funciones 
// GLOBAL: funcion llamada desde el host y ejecutada en el device (kernel)
__global__ void suma_GPU(int* dev_A, int* dev_B, int* dev_resultado, int n)
{
	int myID = threadIdx.x + blockDim.x * blockIdx.x;
	if (myID < n)
	{
		dev_B[myID] = dev_A[(n - 1) - myID];
		dev_resultado[myID] = dev_A[myID] + dev_B[myID];
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
__host__ int maximoHilos(int deviceID )
{

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceID);
	int maximoHilosBloque = deviceProp.maxThreadsPerBlock;
	int maximoBloques = deviceProp.maxGridSize[3];
	int maximoHilos = maximoBloques * maximoHilosBloque;
	return maximoHilos;
}

// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv)
{
	// declaraciones
	int* hst_salida1, * hst_salida2, * hst_resultado;
	int* dev_salida1, * dev_salida2, * dev_resultado;
	int bloques = 0;
	int m = 0;
	int elementos = 0;
	int maximodeHilos = maximoHilos(0);
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
		for (int id = 0; id < deviceCount; id++)
		{
			propiedades_Device(id);
		}
	}
	do {
		printf("Introduzca el numero de elementos: ");
		scanf("%d", &elementos);
		if (elementos > maximodeHilos)
			printf("\nERROR: numero maximo de hilos superado! [%d hilos]\n", maximodeHilos);
	} while (elementos > maximodeHilos);
	bloques = BLOCK;
	m = elementos / BLOCK;
	if (elementos % bloques != 0) {
		m = m + 1;
	}

	printf("Lanzamiento con %d bloques de %d hilos (%d hilos)",m,bloques,bloques*m);
	
	// reserva en el host
	hst_salida1 = (int*)malloc(elementos * sizeof(int));
	hst_salida2 = (int*)malloc(elementos * sizeof(int));
	hst_resultado = (int*)malloc(elementos * sizeof(int));
	// reserva en el device
	cudaMalloc((void**)&dev_salida1, elementos * sizeof(int));
	cudaMalloc((void**)&dev_salida2, elementos * sizeof(int));
	cudaMalloc((void**)&dev_resultado, elementos * sizeof(int));

	srand((int)time(NULL));
	for (int i = 0; i < elementos; i++)
	{
		hst_salida1[i] = rand() % 9;
		hst_salida2[i] = 0;
	}

	cudaMemcpy(dev_salida1, hst_salida1,elementos * sizeof(int), cudaMemcpyHostToDevice);
	// EJECUCIÓN EN EL DEVICE
	// llamada a la funcion "impares_GPU()"
	suma_GPU <<< m, bloques >>> (dev_salida1, dev_salida2, dev_resultado, elementos);

	// recogida de datos desde el device

	cudaMemcpy(hst_salida2, dev_salida2, elementos * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(hst_resultado, dev_resultado, elementos * sizeof(int), cudaMemcpyDeviceToHost);
	// impresion de resultados GPU
	printf("> SALIDA DE LA GPU:\n");
	printf("VECTOR 1:\n");
	for (int i = 0; i < elementos; i++)
		printf(" %2d", hst_salida1[i]);
	printf("\nVECTOR 2:\n");
	for (int i = 0; i < elementos; i++)
		printf(" %2d", hst_salida2[i]);
	printf("\nSUMA:\n");
	for (int i = 0; i < elementos; i++)
		printf(" %2d", hst_resultado[i]);
	printf("\n\n");

	cudaFree(dev_salida1);
	cudaFree(dev_salida2);
	cudaFree(dev_resultado);
	free(hst_salida1);
	free(hst_salida2);
	free(hst_resultado);


	time_t fecha;
	time(&fecha);
	printf("***************************************************\n");
	printf("Programa ejecutado el: %s\n", ctime(&fecha));
	printf("<pulsa [INTRO] para finalizar>");
	getchar();
	return 0;
}