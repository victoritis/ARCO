// includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h> 
// declaracion de funciones
// 
// GLOBAL: funcion llamada desde el host y ejecutada en el device(kernel)
__global__ void pi_gpu(float* datos, float* pi)
// Funcion que suma los primeros N numeros naturales
{
	// KERNEL con 1 bloque de N hilos 
	int N = blockDim.x;
	// indice local de cada hilo 
	int myID = threadIdx.x;
	// rellenamos el vector de datos
	datos[myID] = 1.0/((myID+1)*(myID + 1));
	// sincronizamos para evitar riesgos de tipo RAW
	__syncthreads();
	// ******************
	// REDUCCION PARALELA
	// ******************
	int salto = N / 2;
	// realizamos log2(N) iteraciones
	while (salto > 0)
	{
		// en cada paso solo trabajan la mitad de los hilos
		if (myID < salto)
		{
			datos[myID] = datos[myID] + datos[myID + salto];
		}
		// sincronizamos los hilos evitar riesgos de tipo RAW
		__syncthreads();
		salto = salto / 2;
	}
	// ****************** 
	// Solo el hilo no.'0' escribe el resultado final:
	// evitamos los riesgos estructurales por el acceso a la memoria 
	if (myID == 0)
	{
		*pi = sqrt(datos[0]*6);
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

	printf("***************************************************\n");
}
__host__ int maximo_Hilos(int deviceID)
{

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, deviceID);
	int maximoHilos = deviceProp.maxThreadsPerBlock;
	return maximoHilos;
}
// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv)
{
	// declaraciones
	float* hst_A, * hst_B;
	float* dev_A, * dev_B;
	double errorRelativo;
	int elementos=0;
	int maximodeHilos = maximo_Hilos(0);
	do{
		printf("Introduzca el numero de elementos: ");
		scanf("%d", &elementos);
		if (elementos > maximodeHilos) {
			printf("\nERROR: numero maximo de hilos superado! [%d hilos]\n", maximodeHilos);
		}
	}while(elementos > maximodeHilos);

	// reserva en el host
	hst_A = (float*)malloc(elementos * sizeof(float));
	hst_B = (float*)malloc(elementos * sizeof(float));

	// reserva en el device
	cudaMalloc((void**)&dev_A, elementos * sizeof(float));
	cudaMalloc((void**)&dev_B, elementos * sizeof(float));

	// llamada al kernel bidimensional de NxN hilos
	pi_gpu <<< 1, elementos >>> (dev_A, dev_B);

	// recogida de datos
	cudaMemcpy(hst_B, dev_B, 1 * sizeof(float), cudaMemcpyDeviceToHost);

	errorRelativo = hst_B/3.14159265 * 100;
	// impresion de resultados
	printf("\nValor de pi =  3.14159265\n");
	printf("\nValor calculado =%f\n",hst_B[0]);
	


	// salida
	time_t fecha;
	time(&fecha);
	printf("***************************************************\n");
	printf("Programa ejecutado el: %s\n", ctime(&fecha));
	printf("<pulsa [INTRO] para finalizar>");
	getchar();
	return 0;
}



