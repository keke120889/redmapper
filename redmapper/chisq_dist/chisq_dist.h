#ifndef _CALCLAMBDA_CHISQ_DIST
#define _CALCLAMBDA_CHISQ_DIST

#include <gsl/gsl_blas.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#define MIN_EIGENVAL 1E-6
#define SIGINT_DEFAULT 0.001

struct chisq_dist {
    int mode;
    double *covmat;
    double *c;
    double *slope;
    double *pivotmag;
    double *refmag;
    double *refmagerr;
    double *magerr;
    double *color;
    double *lupcorr;
    int ncalc;
    int ngal;
    int nz;
    int ncol;

    double sigint;
    //    long do_chisq;
    //long nophotoerr;
};


int chisq_dist(int mode, int do_chisq, int nophotoerr, int ncalc, int ncol, double *covmat, double *c, double *slope, double *pivotmag, double *refmag, double *refmagerr, double *magerr, double *color, double *lupcorr, double *dist, double sigint);


int check_and_fix_covmat(gsl_matrix *covmat);

#endif

