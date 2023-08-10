#ifndef BLISLAB_CONFIG_H
#define BLISLAB_CONFIG_H

// Allow C++ users to include this header file in their source code. However,
// we make the extern "C" conditional on whether we're using a C++ compiler,
// since regular C compilers don't understand the extern "C" construct.
#ifdef __cplusplus
extern "C" {
#endif

#define GEMM_SIMD_ALIGN_SIZE 32

#if 1

// Kc * Mc must fit in L3 cache
// Kc * Nr must fit in L2 cache
// Kc * Mr must fit in L1 cache

#define DGEMM_KC 256
#define DGEMM_MC 256
#define DGEMM_NC 512
#define DGEMM_MR 16
#define DGEMM_NR 4
#endif

//#define BL_MICRO_KERNEL bl_dgemm_ukr
#define BL_MICRO_KERNEL bl_dgemm_ukr_sve

// End extern "C" construct block.
#ifdef __cplusplus
}
#endif

#endif
