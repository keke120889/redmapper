#ifndef _SOLVER_NFW
#define _SOLVER_NFW

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#define TOL_DEFAULT 1e-2
#define SCALEVAL_DEFAULT 1.0
#define SCALEVAL_DEFAULT_NOTUSE -1.0
#define CVAL_DEFAULT 0.0
#define RSIG_DEFAULT 0.0
#define CPAR_NTERMS  4

struct solver {
    double r0;
    double beta;
    long ngal;
    double *ucounts;
    double *bcounts;
    double *r;
    double *w;
    double *p;
    double *wt;
    double *cpars;
    double rsig;
    
    double *lambda;
    double *rlambda;
};

int nfw_weights(double inlambda, double r0, double beta, long ngal,
		double *ucounts, double *bcounts, double *r, double *w,
		double *p, double *wt, double *rc, double rsig);

int solver_nfw(double r0, double beta, long ngal,
	       double *ucounts, double *bcounts, double *r, double *w,
	       double *lambda, double *p, double *wt, double *rlambda, double tol,
	       double *cpars, double rsig);


#endif
