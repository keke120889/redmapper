#include <Python.h>
#include "chisq_dist.h"
#include <numpy/arrayobject.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

struct ChisqDistObject {
    PyObject_HEAD
    struct chisq_dist* chisq_dist;
    int allocated;
};
static void
ChisqDistObject_dealloc(struct ChisqDistObject* self)
{
    if (self->allocated) {
	free(self->chisq_dist);
	self->allocated = 0;
    }

    self->ob_type->tp_free((PyObject*)self);
}

static int
ChisqDistObject_init(struct ChisqDistObject* self, PyObject* args)
{
    long mode;
    long ngal;
    long nz;
    long ncol;
    PyArrayObject *covmat_obj = NULL;
    PyArrayObject *c_obj = NULL;
    PyArrayObject *slope_obj = NULL;
    PyArrayObject *pivotmag_obj = NULL;
    PyArrayObject *refmag_obj = NULL;
    PyArrayObject *refmagerr_obj = NULL;
    PyArrayObject *magerr_obj = NULL;
    PyArrayObject *color_obj = NULL;
    PyArrayObject *lupcorr_obj = NULL;

    // debug
    int i,j,k,stride;

    self->allocated = 0;

    if (!PyArg_ParseTuple(args,
			  (char*)"iiiiOOOOOOOOO",
			  &mode,
			  &ngal,
			  &nz,
			  &ncol,
			  &covmat_obj,
			  &c_obj,
			  &slope_obj,
			  &pivotmag_obj,
			  &refmag_obj,
			  &refmagerr_obj,
			  &magerr_obj,
			  &color_obj,
			  &lupcorr_obj)){
	PyErr_SetString(PyExc_RuntimeError,"Failed to parse init");
	return -1;
    }

    if ((self->chisq_dist = calloc(1,sizeof(struct chisq_dist))) == NULL) {
	PyErr_SetString(PyExc_RuntimeError,"Failed to allocate struct chisq_dist");
	return -1;
    }
    self->allocated = 1;

    self->chisq_dist->mode = mode;
    self->chisq_dist->ngal = ngal;
    self->chisq_dist->nz = nz;
    self->chisq_dist->ncol = ncol;
    self->chisq_dist->covmat = (double *) PyArray_DATA(covmat_obj);
    self->chisq_dist->c = (double *) PyArray_DATA(c_obj);
    self->chisq_dist->slope = (double *) PyArray_DATA(slope_obj);
    self->chisq_dist->pivotmag = (double *) PyArray_DATA(pivotmag_obj);
    self->chisq_dist->refmag = (double *) PyArray_DATA(refmag_obj);
    self->chisq_dist->refmagerr = (double *) PyArray_DATA(refmagerr_obj);
    self->chisq_dist->magerr = (double *) PyArray_DATA(magerr_obj);
    self->chisq_dist->color = (double *) PyArray_DATA(color_obj);
    self->chisq_dist->lupcorr = (double *) PyArray_DATA(lupcorr_obj);

    if (self->chisq_dist->mode == 0) {
	self->chisq_dist->ncalc = ngal;
    } else {
	self->chisq_dist->ncalc = nz;
    }

    //    fprintf(stdout,"ncalc: %d\n", self->chisq_dist->ncalc);

    self->chisq_dist->sigint = SIGINT_DEFAULT;

    // and some debugging tests...
    /*
    if (self->chisq_dist->mode == 0) {
	fprintf(stdout,"mode 0\n");
	
	fprintf(stdout,"c: ");
	for (i=0;i<self->chisq_dist->ncol;i++) {
	    fprintf(stdout,"%.4f ",self->chisq_dist->c[i]);
	}
	fprintf(stdout,"\n");
	fprintf(stdout,"slope: ");
	for (i=0;i<self->chisq_dist->ncol;i++) {
	    fprintf(stdout,"%.4f ",self->chisq_dist->slope[i]);
	}
	fprintf(stdout,"\n");
	fprintf(stdout,"pivotmag: %.4f\n",self->chisq_dist->pivotmag[0]);
	fprintf(stdout,"refmag: ");
	for (i=0;i<self->chisq_dist->ncalc;i++) {
	    fprintf(stdout,"%.4f ",self->chisq_dist->refmag[i]);
	}
	fprintf(stdout,"\n");
	fprintf(stdout,"magerr: ");
	for (i=0;i<self->chisq_dist->ncalc;i++) {
	    for (j=0;j<(self->chisq_dist->ncol+1);j++) {
		fprintf(stdout,"%.4f ",self->chisq_dist->magerr[i*(self->chisq_dist->ncol+1) + j]);
	    }
	    fprintf(stdout,"\n");
	}
	fprintf(stdout,"lupcorr: ");
	for (i=0;i<self->chisq_dist->ngal;i++) {
	    for (j=0;j<self->chisq_dist->ncol;j++) {
		fprintf(stdout,"%.5f ",self->chisq_dist->lupcorr[i*(self->chisq_dist->ncol) + j]);
	    }
	    fprintf(stdout,"\n");
	}
	

    } else if (self->chisq_dist->mode == 1) {
	fprintf(stdout,"mode 1\n");

	stride = self->chisq_dist->ncol * self->chisq_dist->ncol;
	
	for (i=0;i<2;i++) {
	    for (j=0;j<self->chisq_dist->ncol;j++) {
		for (k=0;k<self->chisq_dist->ncol;k++) {
		    fprintf(stdout,"%.7f ", self->chisq_dist->covmat[i*stride + k*self->chisq_dist->ncol + j]);
		}
		fprintf(stdout,"\n");
	    }
	    fprintf(stdout,"------------------------\n");
	}

    } else {
	fprintf(stdout,"mode 2\n");


    }
    */
    return 0;
}

static PyObject*
ChisqDistObject_repr(struct ChisqDistObject* self) {
    return PyString_FromString("");
}

PyObject* ChisqDistObject_compute(const struct ChisqDistObject* self, PyObject *args)
{
    npy_intp dims[1];
    PyObject* chisq_obj = NULL;
    double *chisq;
    int do_chisq, nophotoerr;

    dims[0] = self->chisq_dist->ncalc;
    chisq_obj = PyArray_ZEROS(1, dims, NPY_FLOAT64, 0);

    chisq = (double *) PyArray_DATA((PyArrayObject*)chisq_obj);

    
    // parse the args
    if (!PyArg_ParseTuple(args,
			  (char*)"ii",
			  &do_chisq,
			  &nophotoerr)) {
	PyErr_SetString(PyExc_RuntimeError,"Failed to parse args");
	return chisq_obj;
    }
	    
    
    // and do the work.
    chisq_dist(self->chisq_dist->mode, do_chisq, nophotoerr, self->chisq_dist->ncalc,
	       self->chisq_dist->ncol, self->chisq_dist->covmat, self->chisq_dist->c,
	       self->chisq_dist->slope, self->chisq_dist->pivotmag, self->chisq_dist->refmag,
	       self->chisq_dist->refmagerr, self->chisq_dist->magerr, self->chisq_dist->color,
	       self->chisq_dist->lupcorr, chisq, self->chisq_dist->sigint);
    
    return chisq_obj;
}

static PyMethodDef ChisqDistObject_methods[] = {
    {"compute", (PyCFunction)ChisqDistObject_compute, METH_VARARGS, "compute()"},
    {NULL} /* Sentinel */
};

static PyTypeObject PyChisqDistType = {
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
    "_chisq_dist_pywrap.ChisqDist",             /*tp_name*/
    sizeof(struct ChisqDistObject), /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)ChisqDistObject_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    (reprfunc)ChisqDistObject_repr,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /*tp_flags*/
    "ChisqDist Class",           /* tp_doc */
    0,                     /* tp_traverse */
    0,                     /* tp_clear */
    0,                     /* tp_richcompare */
    0,                     /* tp_weaklistoffset */
    0,                     /* tp_iter */
    0,                     /* tp_iternext */
    ChisqDistObject_methods,             /* tp_methods */
    0,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    //0,     /* tp_init */
    (initproc)ChisqDistObject_init,      /* tp_init */
    0,                         /* tp_alloc */
    //PyPSFExObject_new,                 /* tp_new */
    PyType_GenericNew,                 /* tp_new */
};

static PyMethodDef ChisqDist_type_methods[] = {
    {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC
init_chisq_dist_pywrap(void)
{
    PyObject* m;

    PyChisqDistType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&PyChisqDistType) < 0)
	return;

    m = Py_InitModule3("_chisq_dist_pywrap", ChisqDist_type_methods, "ChisqDist (with color)");

    Py_INCREF(&PyChisqDistType);
    PyModule_AddObject(m, "ChisqDist", (PyObject *) &PyChisqDistType);

    import_array();
}


	
