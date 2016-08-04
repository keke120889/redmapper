import healpy as hp
import numpy as np
from catalog import Entry
from utilities import TOTAL_SQDEG, SEC_PER_DEG, astro_to_sphere

_PIXRES = 8

class Mask(object):
	"""Docstring."""

	def __init__(self): pass
	def calc_radmask(self, maskgals, clusters, mpcscale_temp, **kwargs): pass


class HPMask(Mask):
	"""Docstring."""

	def __init__(self, confstr):
		maskinfo, hdr = fitsio.read(confstr.maskfile, ext=1, header=True)
		maskinfo = Entry(maskinfo)
		nlim, nside, nest = maskinfo.hpix.size, hdr['NSIDE'], hdr['NEST']
		hpix_ring = maskinfo.hpix if nest != 1 else hp.nest2ring(nside,maskinfo.hpix) # replace none w/ fn call
		if confstr.hpix > 0:
			hpix_area = TOTAL_SQDEG/(12 * hdr.nside**2)
			border = confstr.border + 2*np.sqrt(hpix_area)

			# which hpix_ring pixels are within confstr.hpix (at confstr.nside)?
			# with some border allowance
			
			#theta,phi=hp.pix2ang(nside,hpix_ring)
			#vec=hp.ang2vec(theta,phi)

			# if we ignore the border...
			theta,phi=hp.pix2ang(nside,hpix_ring)
			hpix_coarse=hp.ang2pix(confstr.nside,theta,phi)

			muse,=np.where(hpix_coarse == confstr.hpix)

			## or...

			theta,phi=hp.pix2ang(nside,hpix_ring)
			ipring_big=hp.ang2pix(confstr.nside,theta,phi)
			indices,=np.where(ipring_big == hpix)
			if border > 0.0:
				# find the extra boundary around the big pixel,
				# at steps equal to the size of the small (mask) pixels
				boundaries=hp.boundaries(confstr.nside,confstr.hpix,step=nside/confstr.nside)
				# see galaxy.py
			# to do query_disc
				# theta,phi is of the center of the big pixel
			vec=hp.ang2vec(theta,phi)

			# set radius to diagonal radius of pixel + border + cushion
			pixint=hp.query_disc(nside,vec[:],radius*np.pi/180.,inclusive=False)
				
			muse,=esutil.numpy_util.match(hpix_ring,pixint)
			
			

			
			# more to come
			#ra, dec = sphere_to_astro(*hp.pix2ang(nside, hpix_ring))
			# more to come
			#muse = None
		else:
			muse = np.arange(nlim)
		offset, ntot = min(hpix_ring)-1, max(hpix_ring)-min(hpix_ring)+3
		self.nside, self.offset, self.npix = nside, offset, npix
		try:
			self.fracgood_float = 1
			self.fracgood[hpix_ring-offset] = instr[muse].fracgood
		except ValueError:
			self.fracgood_float = 0
			self.fracgood[hpix_ring-offset] = 1

	def calc_radmask(self, maskgals, clusters, mpcscale, 
							bsmaskind = None, bfmaskind = None):
		ras = clusters.ra + maskgals.x/(mpcscale*SEC_PER_DEG)/np.cos(np.radians(clusters.dec))
		decs = clusters.dec + maskgals.y/(mpcscale*SEC_PER_DEG)
		theta, phi = astro_to_sphere(ras, dec)
		ipring = hp.ang2pix(self.nside, theta, phi)
		ipring_off = np.clip(ipring - maskstr.offset, 0, maskstr.npix-1)
		if self.fracgood_float == 0: 
			gd, = np.where(self.fracgood[ipring_off] > 0)	
		else:
			gd, = np.where(self.fracgood[ipring_off] > np.random.rand(ras.size))
		radmask[gd] = 1
		return radmask

