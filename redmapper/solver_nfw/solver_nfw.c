#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "solver_nfw.h"

int solver_nfw(double r0, double beta, long ngal,
	       double *ucounts, double *bcounts, double *r, double *w,
	       double *lambda, double *p, double *wt, double *rlambda, double *theta_r,
               double tol, double *cpars, double rsig)
{
  double lamlo,lamhi,mid,outlo,outmid;
  double cval;
  int i;

  lamlo=0.5;
  lamhi=2000.0;
  outlo=-1.0;

  while (fabs(lamhi-lamlo) > 2*tol) {
    mid=(lamhi+lamlo)/2.0;
    if (outlo < 0.0) {
        nfw_weights(lamlo,r0,beta,ngal,ucounts,bcounts,r,w,p,wt,rlambda,theta_r,rsig,1);
      outlo=0.0;
      for (i=0;i<ngal;i++) {
	outlo+=wt[i];
      }
      cval = cpars[3] + cpars[2]*(*rlambda) + cpars[1]*(*rlambda)*(*rlambda) + cpars[0]*(*rlambda)*(*rlambda)*(*rlambda);
      if (cval < 0.0) { cval = 0.0; }

      outlo += lamlo*cval;
    }
    nfw_weights(mid,r0,beta,ngal,ucounts,bcounts,r,w,p,wt,rlambda,theta_r,rsig,1);
    outmid=0.0;
    for (i=0;i<ngal;i++) {
      outmid+=wt[i];
    }
    cval = cpars[3] + cpars[2]*(*rlambda) + cpars[1]*(*rlambda)*(*rlambda) + cpars[0]*(*rlambda)*(*rlambda)*(*rlambda);
    if (cval < 0.0) { cval = 0.0; }

    outmid += mid*cval;

    if (outlo < 1.0) { outlo = 0.9;} // stability at low end
    if ((outlo-lamlo)*(outmid-mid) > 0.0) {
      lamlo=mid;
      outlo=-1.0;
    } else {
      lamhi=mid;
    }
  }

  // lambda is the midpoint of the two
  *lambda = (lamlo+lamhi)/2.0;

  // and final computation of nfw_weights to update values with final lambda
  // at the moment, this will not update p, wt because that's not what happens
  // in the IDL code.  But the IDL code could/should be updated because there
  // is a discrepancy at the nth decimal place between the probabilities computed
  // at "mid" and the final lambda.
  nfw_weights(*lambda,r0,beta,ngal,ucounts,bcounts,r,w,p,wt,rlambda,theta_r,rsig,0);

  if (*lambda < 1.0) {
    *lambda = -1.0;
    *rlambda = -1.0;
  }

  return 0;
}
