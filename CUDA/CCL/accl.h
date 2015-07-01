#ifndef SCCL_CU_H
#define SCCL_CU_H
void acclCuda(int *out, int *components, const int *in,
                 uint nFrames, uint nFramsPerStream, const int rows,
                 const int cols);

#endif
