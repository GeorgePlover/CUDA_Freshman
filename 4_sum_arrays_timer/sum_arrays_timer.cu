#include <cuda_runtime.h>
#include <stdio.h>
#include "freshman.h"



void sumArrays(float * a,float * b,float * res,const int size)
{
  for(int i=0;i<size;i+=4)
  {
    res[i]=a[i]+b[i];
    res[i+1]=a[i+1]+b[i+1];
    res[i+2]=a[i+2]+b[i+2];
    res[i+3]=a[i+3]+b[i+3];
  }
}
__global__ void sumArraysGPU(float*a,float*b,float*res,int N)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i < N)
    res[i]=a[i]+b[i];
}

void sumArrayTimeTest(int grid_dim, int block_dim, int N,
  float* a_d, float* b_d, float* res_d, float* res_from_gpu_h, float* res_cpu) {
    dim3 block(block_dim);
    dim3 grid(grid_dim);

    //timer
    double iStart,iElaps;
    iStart=cpuSecond();
    sumArraysGPU<<<grid,block>>>(a_d,b_d,res_d,N); // 调用核函数
    size_t nByte = sizeof(float) * N;

    CHECK(cudaMemcpy(res_from_gpu_h,res_d,nByte,cudaMemcpyDeviceToHost)); // 拷贝回，隐式同步
    iElaps=cpuSecond()-iStart;
    printf("Execution configuration<<<%d,%d>>> Time elapsed %f sec\n",grid.x,block.x,iElaps);

    checkResult(res_cpu,res_from_gpu_h,N);
}
int main(int argc,char **argv)
{
  // set up device
  initDevice(0);

  int nElem=(1<<24) + 1;
  printf("Vector size:%d\n",nElem);
  int nByte=sizeof(float)*nElem;
  float *a_h=(float*)malloc(nByte);
  float *b_h=(float*)malloc(nByte);
  float *res_h=(float*)malloc(nByte);
  float *res_from_gpu_h=(float*)malloc(nByte);
  memset(res_h,0,nByte);
  memset(res_from_gpu_h,0,nByte);

  float *a_d,*b_d,*res_d;
  CHECK(cudaMalloc((float**)&a_d,nByte));
  CHECK(cudaMalloc((float**)&b_d,nByte));
  CHECK(cudaMalloc((float**)&res_d,nByte));

  initialData(a_h,nElem);
  initialData(b_h,nElem);

  CHECK(cudaMemcpy(a_d,a_h,nByte,cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(b_d,b_h,nByte,cudaMemcpyHostToDevice));

  sumArrays(a_h,b_h,res_h,nElem);

  int block_dims[] = {64, 64, 128, 256, 512, 1024};
  for (int i = 0;i < 6;i++){
    int block_dim = block_dims[i];
    int grid_dim = (nElem - 1) / block_dim + 1;
    sumArrayTimeTest(grid_dim, block_dim, nElem,
      a_d, b_d, res_d, res_from_gpu_h, res_h);
  }
  
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(res_d);

  free(a_h);
  free(b_h);
  free(res_h);
  free(res_from_gpu_h);

  return 0;
}
