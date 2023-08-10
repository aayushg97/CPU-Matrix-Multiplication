#ifndef BLISLAB_DGEMM_KERNEL_H
#define BLISLAB_DGEMM_KERNEL_H

#include "bl_config.h"

#include <stdio.h>
#include <arm_sve.h>


// Allow C++ users to include this header file in their source code. However,
// we make the extern "C" conditional on whether we're using a C++ compiler,
// since regular C compilers don't understand the extern "C" construct.
#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned long long dim_t;

struct aux_s {
    double *b_next;
    float  *b_next_s;
    char   *flag;
    int    pc;
    int    m;
    int    n;
};
typedef struct aux_s aux_t;

void bl_dgemm_ukr(
    int k,
		int m,
		int n,
    double *a,
    double *b,
    double *c,
		unsigned long long ldc,
    aux_t *data );

void bl_dgemm_ukr_sve(
    int k,
    int m,
    int n,
    double *a,
    double *b,
    double *c,
    unsigned long long ldc,
    aux_t *data );

static void (*bl_micro_kernel) (
    int k,
	  int m,
	  int n,
    const double *restrict a,
    const double *restrict b,
    const double *restrict c,
		unsigned long long ldc,
    aux_t *aux
    ) = {
        BL_MICRO_KERNEL
};



// End extern "C" construct block.
#ifdef __cplusplus
}
#endif

#endif
