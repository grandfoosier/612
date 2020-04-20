#include <stdio.h>
#include <stdlib.h>
#include "support.h"

__global__ void kernelP(int m, int n, int k, 
                        const float *A, const float *B, float* C) 
{
	const unsigned int BLOCK_SIZE = 32;
	
	int bx =  blockIdx.x; int by =  blockIdx.y;  
	int tx = threadIdx.x; int ty = threadIdx.y; 
	
	int Row = by * BLOCK_SIZE + ty;
	int Col = bx * BLOCK_SIZE + tx;
	 
	if (Row < m && Col < n)
		for(unsigned int i = 0; i < k; ++i)
			C[row*n + col] += A[row*k + i]*B[i*n + col];
	__syncthreads();
}

void midP(char transa, char transb, \
		  int m, int n, int k, \
		  float alpha, \
		  const float *A, int lda, \
		  const float *B, int ldb, \
		  float beta, \
		 float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
		printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'N') && (transb != 'n')) {
		printf("unsupported value of 'transb'\n");
		return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
		printf("unsupported value of alpha\n");
		return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
		printf("unsupported value of beta\n");
		return;
    }

    const unsigned int BLOCK_SIZE = 32;

	unsigned int grid_y = (unsigned int) ceil((double)m / (double)BLOCK_SIZE); 
	unsigned int grid_x = (unsigned int) ceil((double)n / (double)BLOCK_SIZE); 
	dim3 gridDim(grid_x, grid_y); 
	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
	
	kernelP<<<gridDim, blockDim>>>(m, n, k, A, B, C);
}

int main (int argc, char *argv[])
{

    Timer timer;
    printf("\nRunning Non-Tiled..."); fflush(stdout);
    startTime(&timer);

    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;
    size_t A_sz, B_sz, C_sz;
    unsigned matArow, matAcol;
    unsigned matBrow, matBcol;
    dim3 dim_grid, dim_block;

    if (argc == 1) {
        matArow = 1000;
        matAcol = matBrow = 1000;
        matBcol = 1000;
    } else if (argc == 2) {
        matArow = atoi(argv[1]);
        matAcol = matBrow = atoi(argv[1]);
        matBcol = atoi(argv[1]);
    } else if (argc == 4) {
        matArow = atoi(argv[1]);
        matAcol = matBrow = atoi(argv[2]);
        matBcol = atoi(argv[3]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./sgemm                # All matrices are 1000 x 1000"
           "\n    Usage: ./sgemm <m>            # All matrices are m x m"
           "\n    Usage: ./sgemm <m> <k> <n>    # A: m x k, B: k x n, C: m x n"
           "\n");
        exit(0);
    }

    A_sz = matArow*matAcol;
    B_sz = matBrow*matBcol;
    C_sz = matArow*matBcol;

    A_h = (float*) malloc( sizeof(float)*A_sz );
    for (unsigned int i=0; i < A_sz; i++) A_h[i] = (rand()%100)/100.00;

    B_h = (float*) malloc( sizeof(float)*B_sz );
    for (unsigned int i=0; i < B_sz; i++) B_h[i] = (rand()%100)/100.00;

    C_h = (float*) malloc( sizeof(float)*C_sz );

	printf("    A: %u x %u\n    B: %u x %u\n    C: %u x %u\n", 
		   matArow, matAcol, matBrow, matBcol, matArow, matBcol);
	
	cudaMalloc((void **) &A_d, sizeof(float)*A_sz);
	cudaMalloc((void **) &B_d, sizeof(float)*B_sz);
	cudaMalloc((void **) &C_d, sizeof(float)*C_sz);
	cudaDeviceSynchronize();

	cudaMemcpy(A_d, A_h, sizeof(float)*A_sz, cudaMemcpyHostToDevice);
	cudaMemcpy(B_d, B_h, sizeof(float)*B_sz, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	midP('N', 'N', matArow, matBcol, matBrow, 1.0f,
		 A_d, matArow, B_d, matBrow, 0.0f, C_d, matBrow);
    cudaDeviceSynchronize();

    cudaMemcpy(C_h, C_d, sizeof(float)*C_sz, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
	
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    printf("Verifying results..."); fflush(stdout);

    verify(A_h, B_h, C_h, matArow, matAcol, matBcol);

    free(A_h);
    free(B_h);
    free(C_h);

	cudaFree(A_d);
	cudaFree(B_d);
	cudaFree(C_d);

	return 0;
}