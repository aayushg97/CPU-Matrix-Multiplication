#include "bl_config.h"

#include "bl_dgemm_kernel.h"

#define a(i, j, ld) a[(i) * (ld) + (j)]
#define b(i, j, ld) b[(i) * (ld) + (j)]
#define c(i, j, ld) c[(i) * (ld) + (j)]

void bl_dgemm_ukr(int k,
    int m,
    int n,
    double * a,
    double * b,
    double * c,
    unsigned long long ldc,
    aux_t * data) {

    int l;

    // if the ukernel recieves the fringe part of a matrix
    if (m < DGEMM_MR || n < DGEMM_NR) {
        int i, j;
        for (l = 0; l < k; ++l) {
            for (j = 0; j < n; ++j) {
                for (i = 0; i < m; ++i) {
                    c(i, j, ldc) += a(l, i, DGEMM_MR) * b(l, j, DGEMM_NR);
                }
            }
        }
        return;
    }

    register double c_00_reg, c_01_reg, c_02_reg, c_03_reg;
    register double c_10_reg, c_11_reg, c_12_reg, c_13_reg;
    register double c_20_reg, c_21_reg, c_22_reg, c_23_reg;
    register double c_30_reg, c_31_reg, c_32_reg, c_33_reg;

    c_00_reg = 0.0;
    c_01_reg = 0.0;
    c_02_reg = 0.0;
    c_03_reg = 0.0;
    c_10_reg = 0.0;
    c_11_reg = 0.0;
    c_12_reg = 0.0;
    c_13_reg = 0.0;
    c_20_reg = 0.0;
    c_21_reg = 0.0;
    c_22_reg = 0.0;
    c_23_reg = 0.0;
    c_30_reg = 0.0;
    c_31_reg = 0.0;
    c_32_reg = 0.0;
    c_33_reg = 0.0;

    // fix Mr = 4 and Nr = 4 and fully unroll the inner two loops
    for (l = 0; l < k; ++l) {

        register double b_l_reg;
        double * a0_pntr, * a1_pntr, * a2_pntr, * a3_pntr;

        a0_pntr = & a(l, 0, DGEMM_MR);
        a1_pntr = & a(l, 1, DGEMM_MR);
        a2_pntr = & a(l, 2, DGEMM_MR);
        a3_pntr = & a(l, 3, DGEMM_MR);

        b_l_reg = b(l, 0, DGEMM_NR);

        c_00_reg += * a0_pntr * b_l_reg;
        c_10_reg += * a1_pntr * b_l_reg;
        c_20_reg += * a2_pntr * b_l_reg;
        c_30_reg += * a3_pntr * b_l_reg;

        b_l_reg = b(l, 1, DGEMM_NR);

        c_01_reg += * (a0_pntr) * b_l_reg;
        c_11_reg += * (a1_pntr) * b_l_reg;
        c_21_reg += * (a2_pntr) * b_l_reg;
        c_31_reg += * (a3_pntr) * b_l_reg;

        b_l_reg = b(l, 2, DGEMM_NR);

        c_02_reg += * (a0_pntr) * b_l_reg;
        c_12_reg += * (a1_pntr) * b_l_reg;
        c_22_reg += * (a2_pntr) * b_l_reg;
        c_32_reg += * (a3_pntr) * b_l_reg;

        b_l_reg = b(l, 3, DGEMM_NR);

        c_03_reg += * (a0_pntr) * b_l_reg;
        c_13_reg += * (a1_pntr) * b_l_reg;
        c_23_reg += * (a2_pntr) * b_l_reg;
        c_33_reg += * (a3_pntr) * b_l_reg;
    }

    // store the values of C computed from registers to memory
    c(0, 0, ldc) += c_00_reg;
    c(0, 1, ldc) += c_01_reg;
    c(1, 0, ldc) += c_10_reg;
    c(1, 1, ldc) += c_11_reg;

    c(2, 0, ldc) += c_20_reg;
    c(2, 1, ldc) += c_21_reg;
    c(3, 0, ldc) += c_30_reg;
    c(3, 1, ldc) += c_31_reg;

    c(0, 2, ldc) += c_02_reg;
    c(0, 3, ldc) += c_03_reg;
    c(1, 2, ldc) += c_12_reg;
    c(1, 3, ldc) += c_13_reg;

    c(2, 2, ldc) += c_22_reg;
    c(2, 3, ldc) += c_23_reg;
    c(3, 2, ldc) += c_32_reg;
    c(3, 3, ldc) += c_33_reg;

}

void bl_dgemm_ukr_sve(int k, int m, int n, double * a, double * b, double * c, unsigned long long ldc, aux_t * data) {

    int l, j, i, iter;
    register svfloat64_t aVec, bVec,
    cVec0, cVec1, cVec2, cVec3,
    cVec4, cVec5, cVec6, cVec7,
    cVec8, cVec9, cVec10, cVec11,
    cVec12, cVec13, cVec14, cVec15;

    svbool_t mmPredicate = svwhilelt_b64_u64(0, n);

    // Handling remaining cases when m is not a multiple of numRows
    if (m < DGEMM_MR) {
        int iter = 0;
        for (; iter < m; iter++) {
            cVec0 = svld1_f64(mmPredicate, c + iter * ldc);

            for (l = 0; l < k; ++l) {
                bVec = svld1_f64(svptrue_b64(), b + l * DGEMM_NR);

                register float64_t aElement = a(l, iter, DGEMM_MR);
                aVec = svdup_f64(aElement);
                cVec0 = svmla_f64_m(mmPredicate, cVec0, bVec, aVec);
            }
            svst1_f64(mmPredicate, c + iter * ldc, cVec0);
        }
        return;
    }

    // load the corresponding rows of C into register vectors
    cVec0 = svld1_f64(mmPredicate, c + (0) * ldc);
    cVec1 = svld1_f64(mmPredicate, c + (1) * ldc);
    cVec2 = svld1_f64(mmPredicate, c + (2) * ldc);
    cVec3 = svld1_f64(mmPredicate, c + (3) * ldc);

    cVec4 = svld1_f64(mmPredicate, c + (4) * ldc);
    cVec5 = svld1_f64(mmPredicate, c + (5) * ldc);
    cVec6 = svld1_f64(mmPredicate, c + (6) * ldc);
    cVec7 = svld1_f64(mmPredicate, c + (7) * ldc);

    cVec8 = svld1_f64(mmPredicate, c + (8) * ldc);
    cVec9 = svld1_f64(mmPredicate, c + (9) * ldc);
    cVec10 = svld1_f64(mmPredicate, c + (10) * ldc);
    cVec11 = svld1_f64(mmPredicate, c + (11) * ldc);

    cVec12 = svld1_f64(mmPredicate, c + (12) * ldc);
    cVec13 = svld1_f64(mmPredicate, c + (13) * ldc);
    cVec14 = svld1_f64(mmPredicate, c + (14) * ldc);
    cVec15 = svld1_f64(mmPredicate, c + (15) * ldc);

    for (l = 0; l < k; ++l) {
        bVec = svld1_f64(mmPredicate, b + l * DGEMM_NR);

        // compute the new C vector for each row by multiplying a vector from A and a vector from B
        register float64_t aElement = a(l, 0, DGEMM_MR);
        aVec = svdup_f64(aElement);
        cVec0 = svmla_f64_m(mmPredicate, cVec0, bVec, aVec);

        aElement = a(l, 1, DGEMM_MR);
        aVec = svdup_f64(aElement);
        cVec1 = svmla_f64_m(mmPredicate, cVec1, bVec, aVec);

        aElement = a(l, 2, DGEMM_MR);
        aVec = svdup_f64(aElement);
        cVec2 = svmla_f64_m(mmPredicate, cVec2, bVec, aVec);

        aElement = a(l, 3, DGEMM_MR);
        aVec = svdup_f64(aElement);
        cVec3 = svmla_f64_m(mmPredicate, cVec3, bVec, aVec);

        aElement = a(l, 4, DGEMM_MR);
        aVec = svdup_f64(aElement);
        cVec4 = svmla_f64_m(mmPredicate, cVec4, bVec, aVec);

        aElement = a(l, 5, DGEMM_MR);
        aVec = svdup_f64(aElement);
        cVec5 = svmla_f64_m(mmPredicate, cVec5, bVec, aVec);

        aElement = a(l, 6, DGEMM_MR);
        aVec = svdup_f64(aElement);
        cVec6 = svmla_f64_m(mmPredicate, cVec6, bVec, aVec);

        aElement = a(l, 7, DGEMM_MR);
        aVec = svdup_f64(aElement);
        cVec7 = svmla_f64_m(mmPredicate, cVec7, bVec, aVec);

        aElement = a(l, 8, DGEMM_MR);
        aVec = svdup_f64(aElement);
        cVec8 = svmla_f64_m(mmPredicate, cVec8, bVec, aVec);

        aElement = a(l, 9, DGEMM_MR);
        aVec = svdup_f64(aElement);
        cVec9 = svmla_f64_m(mmPredicate, cVec9, bVec, aVec);

        aElement = a(l, 10, DGEMM_MR);
        aVec = svdup_f64(aElement);
        cVec10 = svmla_f64_m(mmPredicate, cVec10, bVec, aVec);

        aElement = a(l, 11, DGEMM_MR);
        aVec = svdup_f64(aElement);
        cVec11 = svmla_f64_m(mmPredicate, cVec11, bVec, aVec);

        aElement = a(l, 12, DGEMM_MR);
        aVec = svdup_f64(aElement);
        cVec12 = svmla_f64_m(mmPredicate, cVec12, bVec, aVec);

        aElement = a(l, 13, DGEMM_MR);
        aVec = svdup_f64(aElement);
        cVec13 = svmla_f64_m(mmPredicate, cVec13, bVec, aVec);

        aElement = a(l, 14, DGEMM_MR);
        aVec = svdup_f64(aElement);
        cVec14 = svmla_f64_m(mmPredicate, cVec14, bVec, aVec);

        aElement = a(l, 15, DGEMM_MR);
        aVec = svdup_f64(aElement);
        cVec15 = svmla_f64_m(mmPredicate, cVec15, bVec, aVec);
    }

    // store each computed C vector row from register to memory
    svst1_f64(mmPredicate, c + (0) * ldc, cVec0);
    svst1_f64(mmPredicate, c + (1) * ldc, cVec1);
    svst1_f64(mmPredicate, c + (2) * ldc, cVec2);
    svst1_f64(mmPredicate, c + (3) * ldc, cVec3);

    svst1_f64(mmPredicate, c + (4) * ldc, cVec4);
    svst1_f64(mmPredicate, c + (5) * ldc, cVec5);
    svst1_f64(mmPredicate, c + (6) * ldc, cVec6);
    svst1_f64(mmPredicate, c + (7) * ldc, cVec7);

    svst1_f64(mmPredicate, c + (8) * ldc, cVec8);
    svst1_f64(mmPredicate, c + (9) * ldc, cVec9);
    svst1_f64(mmPredicate, c + (10) * ldc, cVec10);
    svst1_f64(mmPredicate, c + (11) * ldc, cVec11);

    svst1_f64(mmPredicate, c + (12) * ldc, cVec12);
    svst1_f64(mmPredicate, c + (13) * ldc, cVec13);
    svst1_f64(mmPredicate, c + (14) * ldc, cVec14);
    svst1_f64(mmPredicate, c + (15) * ldc, cVec15);
}
