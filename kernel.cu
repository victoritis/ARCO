// includes 
#include <stdio.h> 
#include <stdlib.h> 
#include <time.h> 
#include <cuda_runtime.h> 
#define N 16


// declaracion de funciones
// HOST: funcion llamada desde el host y ejecutada en el host 
__host__ void propiedades_Device(int deviceID);

// MAIN: rutina principal ejecutada en el host 
int main(int argc, char** argv)
{

	// declaracion 
	float *hst_A,*hst_B;
	float *dev_A,*dev_B;

	// buscando dispositivos
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount |= 0)
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

	// reserva en el host 
	hst_A = (float*)malloc(N * sizeof(float));
	hst_B = (float*)malloc(N * sizeof(float));

	// reserva en el device 
	cudaMalloc((void**)&dev_A, N * sizeof(float));
	cudaMalloc((void**)&dev_B, N * sizeof(float));

	// inicializacion de datos en el host 
	for (int i = 0; i < N; i++)
	{
		hst_A[i] = (float)rand() / RAND_MAX;
		hst_B[i] = 0;
	}

	// copia de datos CPU -> GPU 
	cudaMemcpy(dev_A, hst_A, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, dev_A, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(hst_B, dev_B, N * sizeof(float), cudaMemcpyDeviceToHost);

	// visualizacion de datos en el host 
	printf("DATOS (hst_A):\n");
	for (int i = 0; i < N; i++)
		printf("%.2f,",hst_A[i]);
	printf("\n");
	printf("SALIDA hst_B:\n");
	for (int i = 0; i < N; i++)
		printf("%.2f,",hst_B[i]);
	printf("\n");

	cudaFree(dev_A);
	cudaFree(dev_B);

	// salida 
	time_t fecha;
	time(&fecha);
	printf("***************************************************\n");
	printf("Programa ejecutado el: %s\n", ctime(&fecha));
	printf("<pulsa [INTRO] para finalizar>");
	getchar();
	return 0;
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
	printf("> CUDA Toolkit \t\t\t: %d.%d\n", rtV / 1000, (rtV % 1000) / 10);
	printf("> Arquitectura CUDA \t\t: %s\n", archName);
	printf("> Capacidad de Computo \t\t: %d.%d\n", major, minor);
	printf("> No. MultiProcesadores \t: %d\n", SM);
	printf("> No. Nucleos CUDA (%dx%d) \t: %d\n", cudaCores, SM, cudaCores* SM);
	printf("> Memoria Global (total) \t: %u MiB\n",	deviceProp.totalGlobalMem / (1024 * 1024));
	printf("***************************************************\n");
}