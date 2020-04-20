#include <stdio.h>
#include <stdlib.h>
#include "support.h"

int main (int argc, char *argv[])
{

    Timer timer;
    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    float *A_h, *B_h, *C_h;
    size_t A_sz, B_sz, C_sz;
    unsigned matArow, matAcol;
    unsigned matBrow, matBcol;

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

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
	printf("    A: %u x %u\n    B: %u x %u\n    C: %u x %u\n", 
		matArow, matAcol, matBrow, matBcol, matArow, matBcol);



    printf("Performing multiplication..."); fflush(stdout);
    startTime(&timer);
	
	for(int row = 0; row < matArow; ++row) {
		for(int col = 0; col < matBcol; ++col) {
			float sum = 0;
			for(unsigned int i = 0; i < matAcol; ++i)
				sum += A[row*matAcol + i]*B[i*matBcol + col];
			C[row*matBcol + col] = sum;
		}
	  
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));



    printf("Verifying results..."); fflush(stdout);

    verify(A_h, B_h, C_h, matArow, matAcol, matBcol);

    free(A_h);
    free(B_h);
    free(C_h);

	return 0;
}