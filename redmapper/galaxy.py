

class Galaxy:

	def __init__(self, ra, dec):
		self.ra, self.dec = ra, dec


class Cluster:

	def __init__(self, center):
		self.center = center

	def ra(self): return self.center.ra

	def dec(self): return self.center.dec
	
	