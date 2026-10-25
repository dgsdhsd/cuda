// cutlass_gemm_f16.cu - CUTLASS fp16 HMMA GEMM, A/B/C 行主序, M=1024,N=4096,K=4096
#include <cutlass/gemm/device/gemm.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define CK(c) do{cudaError_t e=(c);if(e){fprintf(stderr,"CUDA err %s %s\n",__LINE__,cudaGetErrorString(e));exit(1);}}while(0)

using Element  = cutlass::half_t;
using ElementA = Element;
using ElementB = Element;
using ElementC = Element;
using ElementAccum = float;
using RowMajor = cutlass::layout::RowMajor;

using ThreadblockShape = cutlass::gemm::GemmShape<128,128,32>;
using WarpShape        = cutlass::gemm::GemmShape<64,64,32>;
using InstructionShape = cutlass::gemm::GemmShape<16,8,16>;   // m16n8k16 HMMA

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementC, 4, ElementAccum, ElementAccum>;

using Gemm = cutlass::gemm::device::Gemm<
    ElementA, RowMajor, ElementB, RowMajor, ElementC, RowMajor,
    ElementAccum,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadblockShape, WarpShape, InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
    4>;


int main(){
  int M=1024, N=4096, K=4096;
  size_t sA=(size_t)M*K, sB=(size_t)K*N, sC=(size_t)M*N;
  std::vector<half> hA(sA), hB(sB);
  std::srand(2024);
  for(auto&x:hA)x=__float2half((float)(std::rand()%20)-10.0f);
  for(auto&x:hB)x=__float2half((float)(std::rand()%20)-10.0f);
  Element *dA,*dB,*dC;
  CK(cudaMalloc(&dA,sA*2)); CK(cudaMalloc(&dB,sB*2)); CK(cudaMalloc(&dC,sC*2));
  CK(cudaMemcpy(dA,hA.data(),sA*2,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB,hB.data(),sB*2,cudaMemcpyHostToDevice));

  float alpha=1.0f, beta=0.0f;
  Gemm gemm;
  Gemm::Arguments args({M,N,K}, {dA,K}, {dB,N}, {dC,N}, {dC,N}, {alpha,beta});

  // warmup
  for(int i=0;i<5;i++){ cutlass::Status st=gemm(args); if(st!=cutlass::Status::kSuccess){ fprintf(stderr,"warmup fail\n");return 1;} }
  CK(cudaDeviceSynchronize());

  // timed
  double best=1e9;
  for(int t=0;t<3;t++){
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); cudaEventRecord(s);
    for(int i=0;i<10;i++){ cutlass::Status st=gemm(args); if(st != cutlass::Status::kSuccess) return 1; }
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms=0; cudaEventElapsedTime(&ms,s,e); ms/=10; if(ms<best)best=ms;
    cudaEventDestroy(s);cudaEventDestroy(e);
  }
  double flops=2.0*(double)M*N*K;
  printf("[CUTLASS fp16 HMMA, A/B/C 行主序] M=%d N=%d K=%d\n",M,N,K);
  printf("  time=%.4f ms  compute=%.2f TFLOPS (best of 3)\n", best, flops/(best*1e9));

  // CPU 参考校验(左上角 128x128)
  std::vector<half> hC(sC); CK(cudaMemcpy(hC.data(),dC,sC*2,cudaMemcpyDeviceToHost));
  double md=0,mean=0; int tot=0;
  for(int m=0;m<128;++m)for(int n=0;n<128;++n){ float acc=0;
    for(int k=0;k<K;++k) acc+=(float)__half2float(hA[(size_t)m*K+k])*(float)__half2float(hB[(size_t)k*N+n]);
    double d=std::fabs((double)__half2float(hC[(size_t)m*N+n])-acc); if(d>md)md=d; mean+=d; tot++; }
  double tol=0.01*(33.0*K)+8;
  printf("  CPU check: max=%.3f tol=%.0f %s\n",md,tol,md<tol?"[PASS]":"[FAIL]");
  return (md<tol)?0:1;
}
