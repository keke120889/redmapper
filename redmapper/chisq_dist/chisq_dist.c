#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_cblas.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_eigen.h>


#include "chisq_dist.h"

int chisq_dist(int mode, int do_chisq, int nophotoerr, int ncalc, int ncol, 
	       double *covmat, double *c, double *slope,
	       double *pivotmag, double *refmag, double *refmagerr, double *magerr, 
	       double *color, double *lupcorr, double *dist, double sigint) {

  //int i,j,k;
  int i,j,k;
  int nmag = ncol+1;

  //gsl_matrix *mcovmat;
  gsl_matrix_view mvcovmat;
  gsl_matrix *cobs;
  gsl_matrix *cimat;
  gsl_matrix *cobsmat;
  gsl_matrix *cobstemp;
  gsl_matrix *mmetric;
  gsl_matrix *mrotmat;
  gsl_vector *vdcm;
  gsl_permutation *pp;
  gsl_vector *vdc;
  int s;
  double *covmat_temp = NULL;

  int covmat_stride;
  int test;
  double val;

  double chisq,norm;

  int use_refmagerr=0;

  norm = 1.0;
  
  covmat_stride = ncol*ncol; //*sizeof(double);

  mrotmat = gsl_matrix_alloc(ncol,nmag);
  gsl_matrix_set_zero(mrotmat);  // important!
  for (i=0;i<ncol;i++) {
    gsl_matrix_set(mrotmat, i, i, 1.0);
    gsl_matrix_set(mrotmat, i, i+1, -1.0);
  }
  cobs = gsl_matrix_alloc(nmag, nmag);
  cobsmat = gsl_matrix_alloc(ncol,ncol);
  cobstemp = gsl_matrix_alloc(ncol,nmag);
  pp=gsl_permutation_alloc(ncol);
  mmetric=gsl_matrix_alloc(ncol,ncol);
  vdc = gsl_vector_alloc(ncol);
  vdcm=gsl_vector_alloc(ncol);

  cimat = NULL;
  if (refmagerr != NULL) {
      use_refmagerr = 1;
      cimat = gsl_matrix_alloc(ncol,ncol);
  }

  if (mode == 0) {
    // mode = 0
    //   Many galaxies, one redshift
    //     - refmag/refmagerr is an array with ncalc (==ngal)
    //     - color is a matrix with ncol x ncalc (==ngal)
    //     - magerr is a matrix with nmag x ncalc (==ngal)
    //     - c is an array with ncol
    //     - slope is an array with ncol
    //     - pivotmag is a single value
    //     - lupcorr is a matrix with ncol x ncalc (==ngal)
    //     //- covmat is a matrix with ncol x ncol x ncalc (==ngal)
    //     - covmat is a matrix with ncol x ncol
    //
    // // covmat[ncol,ncol,ncalc]: (k*ncol+j)*ncol + i
    // covmat[ncol,ncol]: k*ncol + j
    // c[ncol]: j  
    // slope[ncol]: j
    // pivotmag[0]
    // refmag[ncalc]: i
    // refmagerr[ncalc]: i
    // magerr[nmag,ncalc]: i*nmag + j
    // color[ncol,ncalc]: i*ncol + j
    // lupcorr[ncol,ncalc]: i*ncol + j

    // first make a buffer for the local copy of the covmat (which gets overwritten)
    if ((covmat_temp = (double *)calloc(ncol*ncol, sizeof(double))) == NULL) {
	return -1;
    }
    
    for (i=0;i<ncalc;i++) {

      memcpy(covmat_temp,covmat,sizeof(double)*ncol*ncol);
	
      gsl_matrix_set_identity(cobs);
      for (j=0;j<nmag;j++) {
	gsl_matrix_set(cobs, j, j, magerr[i*nmag+j]*magerr[i*nmag+j]);
      }

      gsl_matrix_set_zero(cobstemp);
      
      gsl_blas_dgemm(CblasNoTrans, CblasTrans,
		     1.0, mrotmat, cobs,
		     0.0, cobstemp);
      gsl_blas_dgemm(CblasNoTrans, CblasTrans,
		     1.0, mrotmat, cobstemp,
		     0.0, cobsmat);


      //mvcovmat = gsl_matrix_view_array(&covmat[covmat_stride*i], ncol, ncol);
      mvcovmat = gsl_matrix_view_array(covmat_temp, ncol, ncol);

      // and the ci matrix
      if (use_refmagerr) {
	  // make the C_i matrix for the refmag err
	  //  this is a matrix with ncol x ncol size
	  gsl_matrix_set_zero(cimat);
	  for (j=0;j<ncol;j++) {
	      for (k=j;k<ncol;k++) {
		  val = slope[j] * slope[k] * refmagerr[i] * refmagerr[i];
		  gsl_matrix_set(cimat, j, k, val);
		  if (k != j) {
		      gsl_matrix_set(cimat, k, j, val);
		  }
	      }
	  }
      }

      // check sigint
      test = 1;
      for (j=0;j<ncol;j++) {
	if (gsl_matrix_get(&mvcovmat.matrix, j, j) < sigint*sigint) {
	  test = 0;
	}
      }

      if (test == 0) {
	if (do_chisq) {
	  dist[i] = 1e11;
	} else {
	  dist[i] = -1e11;
	}
      } else {
	if (!nophotoerr) {
	  gsl_matrix_add(&mvcovmat.matrix, cobsmat);
	}

	if (use_refmagerr) {
	    gsl_matrix_add(&mvcovmat.matrix, cimat);
	}


	// check and fix the matrix if necessary
	check_and_fix_covmat(&mvcovmat.matrix);


	gsl_linalg_LU_decomp(&mvcovmat.matrix, pp, &s);
	gsl_linalg_LU_invert(&mvcovmat.matrix, pp, mmetric);
	
	if (!do_chisq) {
	  // need the determinant
	  norm = gsl_linalg_LU_det(&mvcovmat.matrix, s);
	}
	
	// now need the slope, etc.
	for (j=0;j<ncol;j++) {
	  gsl_vector_set(vdc,j,
			 (c[j]+slope[j]*(refmag[i]-pivotmag[0])) + lupcorr[i*ncol+j] -
			 color[i*ncol+j]);
	}
	
	gsl_blas_dgemv(CblasNoTrans, 1.0, mmetric, vdc, 0.0, vdcm);
	gsl_blas_ddot(vdcm, vdc, &chisq);
	
	if (do_chisq) {
	  dist[i] = chisq;
	} else {
	  dist[i]=-0.5*chisq-0.5*log(norm);
	}
      }
    
    }
    free(covmat_temp);

  } else if (mode==1) {
    // mode = 1
    //   One galaxy, many redshifts
    //     - refmag/refmagerr is a single value
    //     - color is an array with ncol values
    //     - magerr is an array with nmag=ncol+1 values
    //     - c is a matrix with ncol x ncalc (==nz)
    //     - slope is a matrix with ncol x ncalc (==nz)
    //     - pivotmag is an array with ncalc (==nz)
    //     - lupcorr is a matrix with ncol x ncalc (==nz)
    //     - covmat is a matrix with ncol x ncol x ncalc (==nz)
    //
    // covmat[ncol,ncol,ncalc] : (k*ncol+j)*ncol+i
    // c[ncol,ncalc]: i*ncol+j
    // slope[ncol,ncalc]: i*ncol+j
    // pivotmag[ncalc]: i
    // magerr[nmag]: j
    // color[ncol]: j
    // lupcorr[ncol,ncalc]: i*ncol + j
    // refmag[0]
    // refmagerr[0]

    // first make a buffer for the local copy of the covmat (which gets overwritten)
    if ((covmat_temp = (double *)calloc(ncol*ncol*ncalc, sizeof(double))) == NULL) {
	return -1;
    }
    memcpy(covmat_temp,covmat,sizeof(double)*ncol*ncol*ncalc);
      
    // We can generate the C_obs matrix (cobsmat) just once
    gsl_matrix_set_identity(cobs);
    for (j=0;j<nmag;j++) {
      gsl_matrix_set(cobs,j,j,magerr[j]*magerr[j]);
    }

    gsl_matrix_set_zero(cobstemp);

    gsl_blas_dgemm(CblasNoTrans, CblasTrans,
		   1.0, mrotmat, cobs,
		   0.0, cobstemp);
    gsl_blas_dgemm(CblasNoTrans, CblasTrans,
		   1.0, mrotmat, cobstemp,
		   0.0, cobsmat);

    for (i=0;i<ncalc;i++) {
      mvcovmat = gsl_matrix_view_array(&covmat_temp[covmat_stride*i], ncol, ncol);

      if (use_refmagerr) {
	  // make the C_i matrix for the refmag err
	  //  this is a matrix with ncol x ncol size...
	  gsl_matrix_set_zero(cimat);
	  for (j=0;j<ncol;j++) {
	      for (k=j;k<ncol;k++) {
		  val = slope[i*ncol+j] * slope[i*ncol+k] * refmagerr[0] * refmagerr[0];
		  gsl_matrix_set(cimat, j, k, val);
		  if (k != j) {
		      gsl_matrix_set(cimat, k, j, val);
		  }
	      }
	  }
      }
      
      // check sigint
      test = 1;
      for (j=0;j<ncol;j++) {
	if (gsl_matrix_get(&mvcovmat.matrix, j, j) < sigint*sigint) {
	  test = 0;
	}
      }

      if (test == 0) {
	if (do_chisq) {
	  dist[i] = 1e11;
	} else {
	  dist[i] = -1e11;
	}
      } else {
	if (!nophotoerr) {
	  gsl_matrix_add(&mvcovmat.matrix, cobsmat);
	}

	if (use_refmagerr) {
	    gsl_matrix_add(&mvcovmat.matrix, cimat);
	}

	// check and fix the matrix if necessary
	check_and_fix_covmat(&mvcovmat.matrix);
	
	gsl_linalg_LU_decomp(&mvcovmat.matrix, pp, &s);
	gsl_linalg_LU_invert(&mvcovmat.matrix, pp, mmetric);

	if (!do_chisq) {
	  norm = gsl_linalg_LU_det(&mvcovmat.matrix, s);
	}

	for (j=0;j<ncol;j++) {
	  gsl_vector_set(vdc,j,
			 (c[i*ncol+j] + slope[i*ncol+j]*(refmag[0] - pivotmag[i])) + lupcorr[i*ncol+j] -
			 color[j]);
	}

	gsl_blas_dgemv(CblasNoTrans, 1.0, mmetric, vdc, 0.0, vdcm);
	gsl_blas_ddot(vdcm,vdc, &chisq);
	
	if (do_chisq) {
	  dist[i] = chisq;
	} else {
	  dist[i]=-0.5*chisq-0.5*log(norm);
	}

      }
    }
    free(covmat_temp);
  } else {
      // mode = 2
      //  Many galaxies, many redshifts
      //     - refmag/refmagerr is an array with ncalc (==ngal)
      //     - color is a matrix with ncol x ncalc (==ngal)
      //     - magerr is a matrix with nmag x ncalc (==ngal)
      //     - c is a matrix with ncol x ncalc (==ngal)
      //     - slope is a matrix with ncol x ncalc (==ngal)
      //     - pivotmag is an array with ncalc (==ngal)
      //     - lupcorr is a matrix with ncol x ncalc (==ngal)
      //     - covmat is a matrix with ncol x ncol x ncalc (==ngal)
      //
      // covmat[ncol,ncol,ncalc] : (k*ncol+j)*ncol+i   -> mode 0    
      // c[ncol,ncalc]: i*ncol+j                       -> mode 1
      // slope[ncol,ncalc]: i*ncol+j                   -> mode 1
      // pivotmag[ncalc]: i                              -> mode 1
      // refmag[ncalc]: i                                -> mode 0
      // refmagerr[ncalc]: i                             -> mode 0
      // magerr[nmag,ncalc]: i*nmag + j                -> mode 0
      // color[ncol,ncalc]: i*ncol + j                 -> mode 0
      // lupcorr[ncol,ncalc]: i*ncol + j               -> mode 0, 1

      if ((covmat_temp = (double *)calloc(ncol*ncol*ncalc, sizeof(double))) == NULL) {
	  return -1;
      }
      memcpy(covmat_temp,covmat,sizeof(double)*ncol*ncol*ncalc);

      
      for (i=0;i<ncalc;i++) {
	  // copy from mode 0
	  gsl_matrix_set_identity(cobs);
	  for (j=0;j<nmag;j++) {
	      // okay
	      gsl_matrix_set(cobs, j, j, magerr[i*nmag+j]*magerr[i*nmag+j]);
	  }

	  gsl_matrix_set_zero(cobstemp);
      
	  gsl_blas_dgemm(CblasNoTrans, CblasTrans,
			 1.0, mrotmat, cobs,
			 0.0, cobstemp);
	  gsl_blas_dgemm(CblasNoTrans, CblasTrans,
			 1.0, mrotmat, cobstemp,
			 0.0, cobsmat);

	  mvcovmat = gsl_matrix_view_array(&covmat_temp[covmat_stride*i], ncol, ncol);

	  // and the ci matrix
	  // copy from mode 0
	  if (use_refmagerr) {
	      // make the C_i matrix for the refmag err
	      //  this is a matrix with ncol x ncol size
	      gsl_matrix_set_zero(cimat);
	      for (j=0;j<ncol;j++) {
		  for (k=j;k<ncol;k++) {
		      val = slope[j] * slope[k] * refmagerr[i] * refmagerr[i];
		      gsl_matrix_set(cimat, j, k, val);
		      if (k != j) {
			  gsl_matrix_set(cimat, k, j, val);
		      }
		  }
	      }
	  }

	  // check sigint
	  // copy from mode 0
	  test = 1;
	  for (j=0;j<ncol;j++) {
	      if (gsl_matrix_get(&mvcovmat.matrix, j, j) < sigint*sigint) {
		  test = 0;
	      }
	  }

	  if (test == 0) {
	      if (do_chisq) {
		  dist[i] = 1e11;
	      } else {
		  dist[i] = -1e11;
	      }
	  } else {
	      // copy from mode 0
	      if (!nophotoerr) {
		  gsl_matrix_add(&mvcovmat.matrix, cobsmat);
	      }

	      // copy from mode 0
	      if (use_refmagerr) {
		  gsl_matrix_add(&mvcovmat.matrix, cimat);
	      }

	      // check and fix the matrix if necessary
	      // copy from mode 0
	      check_and_fix_covmat(&mvcovmat.matrix);

	      gsl_linalg_LU_decomp(&mvcovmat.matrix, pp, &s);
	      gsl_linalg_LU_invert(&mvcovmat.matrix, pp, mmetric);
	      
	      if (!do_chisq) {
		  // need the determinant
		  norm = gsl_linalg_LU_det(&mvcovmat.matrix, s);
	      }

	      // vdc is a vector with ncol...
	      // copy from mode 1
	      for (j=0;j<ncol;j++) {
		  gsl_vector_set(vdc,j,
				 (c[i*ncol+j] + slope[i*ncol+j]*(refmag[i] - pivotmag[i])) + lupcorr[i*ncol+j] -
				 color[i*ncol+j]);
	      }

	      gsl_blas_dgemv(CblasNoTrans, 1.0, mmetric, vdc, 0.0, vdcm);
	      gsl_blas_ddot(vdcm,vdc, &chisq);

	      if (do_chisq) {
		  dist[i] = chisq;
	      } else {
		  dist[i]=-0.5*chisq-0.5*log(norm);
	      }
	  }
	  
	  
      }
      free(covmat_temp);
  }

  gsl_matrix_free(mrotmat);
  gsl_matrix_free(cobs);
  gsl_matrix_free(cobsmat);
  gsl_matrix_free(cobstemp);
  if (use_refmagerr) gsl_matrix_free(cimat);
  gsl_permutation_free(pp);
  gsl_matrix_free(mmetric);
  gsl_vector_free(vdcm);
  gsl_vector_free(vdc);


  return 0;
}

int check_and_fix_covmat(gsl_matrix *covmat) {
  int i, s, test;
  double eigenval_i;
  gsl_vector_view diag;

  int nelt;
  gsl_matrix *mat;
  gsl_vector *eigenval;
  gsl_vector *eigenval_temp;
  gsl_eigen_symm_workspace *wval;
  gsl_eigen_symmv_workspace *wvec;
  gsl_matrix *Q;
  gsl_matrix *Qinv;
  gsl_matrix *Lambda;
  gsl_matrix *temp;
  gsl_permutation *pp;

  nelt = covmat->size1;
  
  mat=gsl_matrix_alloc(nelt,nelt);
  wval = gsl_eigen_symm_alloc(nelt);
  wvec = gsl_eigen_symmv_alloc(nelt);
  eigenval = gsl_vector_alloc(nelt);
  
  

  // don't destroy the input matrix!
  gsl_matrix_memcpy(mat, covmat);

  // calculate eigenvalues...
  gsl_eigen_symm(mat, eigenval, wval);
  

  // test if the eigenvalues are negative...
  test = 0;
  for (i=0;i<nelt;i++) {
    eigenval_i = gsl_vector_get(eigenval,i);
    if (eigenval_i < MIN_EIGENVAL) {
      test = 1;
      gsl_vector_set(eigenval,i,MIN_EIGENVAL);
    }
  }
  if (test) {
    // initialize
    Q=gsl_matrix_alloc(nelt,nelt);
    Qinv=gsl_matrix_alloc(nelt,nelt);
    Lambda=gsl_matrix_alloc(nelt,nelt);
    temp=gsl_matrix_alloc(nelt,nelt);
    eigenval_temp=gsl_vector_alloc(nelt);
    pp=gsl_permutation_alloc(nelt);


    // reset the matrix
    gsl_matrix_memcpy(mat, covmat);

    // calculate eigenvalues and eigenvector matrix Q
    gsl_eigen_symmv(mat, eigenval_temp, Q, wvec);
   
    // invert eigenvector matrix Q-> Qinv (leaving Q in place)
    gsl_matrix_memcpy(temp,Q);
    gsl_linalg_LU_decomp(temp, pp, &s);
    gsl_linalg_LU_invert(temp, pp, Qinv);
    
    // create a diagonal matrix Lambda
    diag = gsl_matrix_diagonal(Lambda);
    gsl_matrix_set_zero(Lambda);
    gsl_vector_memcpy(&diag.vector, eigenval);

    // do the multiplication
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, Q, Lambda, 0.0, temp);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, temp, Qinv, 0.0, covmat);

    gsl_matrix_free(Q);
    gsl_matrix_free(Qinv);
    gsl_matrix_free(Lambda);
    gsl_matrix_free(temp);
    gsl_vector_free(eigenval_temp);
    gsl_permutation_free(pp);

  }

  gsl_matrix_free(mat);
  gsl_vector_free(eigenval);
  gsl_eigen_symm_free(wval);
  gsl_eigen_symmv_free(wvec);

  return 0;

}

