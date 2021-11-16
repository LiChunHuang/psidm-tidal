
#ifdef __CUDACC__
#include "CUAPI.h"



extern int    Eridanus_Prof_NBin;
       
       real *d_ExtPotGrep_Eri;
extern void **d_ExtPotGenePtr;

//--------------------------------------------------------------------------
//Function:
//Description:
//Note:
//-------------------------------------------------------------------------
__host__
void SetGPUPtr(const real *h_table)
{
//   for ( int b = 0; b<3*Eridanus_Prof_NBin; b++)
//   {
//       printf("h table = %.4f\n", h_table[b]);
//   }
     
   const long MemSize = sizeof(real)*Eridanus_Prof_NBin*3;
 
   CUDA_CHECK_ERROR( cudaMalloc((void**) &d_ExtPotGrep_Eri, MemSize));
//   real *h_00 = (real*)malloc(MemSize);
//   memset(h_00, 0, MemSize);
//   cudaMemcpy(d_ExtPotGrep_Eri,h_00,MemSize,cudaMemcpyHostToDevice);
  
//   printf("1\n");
   CUDA_CHECK_ERROR( cudaMemcpy(d_ExtPotGrep_Eri, h_table,MemSize,cudaMemcpyHostToDevice));
//   printf("2\n");
   real *d_ExtPotGrep_Ptr[3] = {d_ExtPotGrep_Eri,d_ExtPotGrep_Eri+Eridanus_Prof_NBin,d_ExtPotGrep_Eri+2*Eridanus_Prof_NBin};
//   printf("3\n");

   CUDA_CHECK_ERROR( cudaMemcpy(d_ExtPotGenePtr, d_ExtPotGrep_Ptr, sizeof(real*)*3,cudaMemcpyHostToDevice));
//   printf("4\n");
}

#endif




