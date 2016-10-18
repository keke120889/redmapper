redmapper
=========
This will be the public, python version of redMaPPer (see (here)
[http://adsabs.harvard.edu/abs/2014ApJ...785..104R]).

Installation
------------
You can install this module with
```
cd redmapper
make
python setup.py install
```

If you want to keep the root directory clean then run
```
cd redmapper
python setup.py clean
```

Tests
-----
Once installed, feel free to run the nosetests with
```
cd redmapper/tests
```
where the tests can be run individually.

Dependencies
------------
Note: this list is ongoing. A list of dependencies includes
the following:
* numpy
* nose
* pyyaml
* sphinx
* fitsio
* esutil
* healpy

These modules will be installed when _make_ runs.