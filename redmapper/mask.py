import utilities
import healpy as hp
import numpy as np
from catalog import Entry

class Mask(object):
	"""Docstring."""

	def __init__(self): pass
	def calc_radmask(self): pass


class HPMask(Mask):
	"""Docstring."""

	def __init__(self, filename):
		maskinfo, hdr = (Entry(arr) for arr in 
							fitsio.read(filename, ext=1, header=True))
		hpix_area = utilities.TOTAL_SQDEG/(12 * hdr.nside**2)
		


	def calc_radmask(self): pass