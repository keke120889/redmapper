#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

#include "solver_nfw.h"

int nfw_weights(double inlambda, double r0, double beta, long ngal,
		double *ucounts, double *bcounts, double *r, double *w,
		double *p, double *wt, double *rc, double rsig)
{
  double nfwnorm,logrc;
  int i;
  double rsig_const;

  // pre-compute rsig_const
  rsig_const = 1./(sqrt(2)*rsig);

  // calculate rc
  *rc=r0*pow((inlambda/100.),beta);
  
  // and find the norm  -- hard coded
  logrc = log(*rc);
  nfwnorm = exp(1.65169 - 0.547850*logrc + 0.138202*pow(logrc,2)
		-0.0719021*pow(logrc,3) - 0.0158241*pow(logrc,4)
		-0.000854985*pow(logrc,5));
    
  for (i=0;i<ngal;i++) {
    // calculate p[i]
    p[i] = (inlambda*ucounts[i]*nfwnorm)/(inlambda*ucounts[i]*nfwnorm+bcounts[i]);

    // check for infinity/NaN
    if (!finitef(p[i])) p[i]=0.0;
    
    // and check radius and do weights
    if (rsig <= 0.0) {
      // old skool... hard cut.
      if (r[i] < *rc) {
	wt[i]=p[i]*w[i];
      } else {
        wt[i]=0.0;
      }
    } else {
      // Now we have a soft cut.
      if (r[i] < (*rc - 5.0*rsig)) {
        // r correction value is 1.0
	wt[i] = p[i]*w[i]*1.0;
      } else if (r[i] > (*rc + 5.0*rsig)) {
	// r correction value is 0.0
	wt[i] = 0.0;
      } else {
	// need to calculate r correction...
	wt[i] = p[i]*w[i]*(0.5+0.5*erf((*rc - r[i])*rsig_const));
      }
    }
  }
  
  return 0;
}
