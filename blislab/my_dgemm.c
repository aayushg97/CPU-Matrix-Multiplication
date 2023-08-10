/*
 * --------------------------------------------------------------------------
 * BLISLAB
 * --------------------------------------------------------------------------
 * Copyright (C) 2016, The University of Texas at Austin
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name of The University of Texas nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * bl_dgemm.c
 *
 *
 * Purpose:
 * this is the main file of blislab dgemm.
 *
 * Todo:
 *
 *
 * Modification:
 *      bryan chin - ucsd
 *      changed to row-major order
 *      handle arbitrary  size C
 * */

#include <stdio.h>

#include "bl_dgemm_kernel.h"

#include "bl_dgemm.h"

const char *dgemm_desc = "my blislab ";

static inline
void packA_mcxkc_d(
    int m,
    int k,
    double *XA,
    int ldXA,
    double *packA) {

    int i, j;
    for (j = 0; j < k; j++) {

        // unrolling loop by a factor of 4
        for (i = 0; i < m; i += 4) {
            if (i + 3 < m) {
                *packA = XA[i * ldXA + j];
                *(packA + 1) = XA[(i + 1) * ldXA + j];
                *(packA + 2) = XA[(i + 2) * ldXA + j];
                *(packA + 3) = XA[(i + 3) * ldXA + j];
                packA += 4;
            } else
                break;
        }

        // fringe case when there are left over elements after unrolling
        while (i < m) {
            *packA++ = XA[i * ldXA + j];
            i++;
        }

        // add padding for matrices that are not multiple of block sizes
        for (; i < DGEMM_MR; i++)
            *packA++ = 0.0;
    }
}

static inline
void packB_kcxnc_d(
    int n,
    int k,
    double *XB,
    int ldXB,
    double *packB) {

    int i, j;

    for (i = 0; i < k; i++) {

        // unrolling loop by a factor of 4
        for (j = 0; j < n; j += 4) {
            if (j + 3 < n) {
                *packB = XB[i * ldXB + j];
                *(packB + 1) = XB[i * ldXB + j + 1];
                *(packB + 2) = XB[i * ldXB + j + 2];
                *(packB + 3) = XB[i * ldXB + j + 3];
                packB += 4;
            } else
                break;
        }

        // fringe case when there are left over elements after unrolling
        while (j < n) {
            *packB++ = XB[i * ldXB + j];
            j++;
        }

        // add padding for matrices that are not multiple of block sizes
        for (; j < DGEMM_NR; j++)
            *packB++ = 0.0;
    }
}

static
inline
void bl_macro_kernel(
    int m,
    int n,
    int k,
    const double *packA,
    const double *packB,
    double *C,
    int ldc) {

    int i, j;
    aux_t aux;

    for (i = 0; i < m; i += DGEMM_MR) {
        for (j = 0; j < n; j += DGEMM_NR) {
            (*bl_micro_kernel)(
                k,
                min(m - i, DGEMM_MR),
                min(n - j, DGEMM_NR),
                &packA[i * k],
                &packB[j * k],
                &C[i * ldc + j],
                (unsigned long long) ldc,
                &aux
            );
        }
    }
}

void bl_dgemm(
    int m,
    int n,
    int k,
    double *XA,
    int lda,
    double *XB,
    int ldb,
    double *C,
    int ldc
) {
    int ic, ib, jc, jb, pc, pb;
    double *packA, *packB;

    packA = bl_malloc_aligned(DGEMM_KC, (DGEMM_MC / DGEMM_MR + 1) * DGEMM_MR, sizeof(double));
    packB = bl_malloc_aligned(DGEMM_KC, (DGEMM_NC / DGEMM_NR + 1) * DGEMM_NR, sizeof(double));

    // commenting below lines because it seems to have no impact on results

    // int packIndex;
    // for(packIndex = 0; packIndex < DGEMM_KC * (DGEMM_MC/DGEMM_MR + 1) * DGEMM_MR; packIndex++)
    //   packA[packIndex] = 0.0;
    //
    // for(packIndex = 0; packIndex < DGEMM_KC * (DGEMM_NC/DGEMM_NR + 1) * DGEMM_NR; packIndex++)
    //   packB[packIndex] = 0.0;

    for (ic = 0; ic < m; ic += DGEMM_MC) { // 5-th loop around micro-kernel
        ib = min(m - ic, DGEMM_MC);
        for (pc = 0; pc < k; pc += DGEMM_KC) { // 4-th loop around micro-kernel
            pb = min(k - pc, DGEMM_KC);

            int i, j;
            for (i = 0; i < ib; i += DGEMM_MR) {

                packA_mcxkc_d(
                    min(ib - i, DGEMM_MR),
                    pb,
                    &XA[pc + lda * (ic + i)],
                    k,
                    &packA[0 * DGEMM_MC * pb + i * pb]);
            }

            for (jc = 0; jc < n; jc += DGEMM_NC) { // 3-rd loop around micro-kernel
                jb = min(n - jc, DGEMM_NC);

                for (j = 0; j < jb; j += DGEMM_NR) {
                    packB_kcxnc_d(
                        min(jb - j, DGEMM_NR),
                        pb,
                        &XB[ldb * pc + jc + j],
                        n,
                        &packB[j * pb]
                    );
                }

                bl_macro_kernel(
                    ib,
                    jb,
                    pb,
                    packA,
                    packB,
                    &C[ic * ldc + jc],
                    ldc
                );
            } // End 3.rd loop around micro-kernel
        } // End 4.th loop around micro-kernel
    } // End 5.th loop around micro-kernel

    free(packA);
    free(packB);

}

void square_dgemm(int lda, double *A, double *B, double *C) {
    bl_dgemm(lda, lda, lda, A, lda, B, lda, C, lda);
}
