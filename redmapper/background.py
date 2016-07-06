import fitsio


class Background:

	""" docstring """

	def __init__(self, filename, confdict=None, scale=1.0):
		self._filename = filename
		self._read_background(confdict)

	def _read_background(self, confdict):
		if confdict is None: pass # populate dictionary
		# then set attributes
		self.bkgdict = fitsio.read(self._filename)


	def 

