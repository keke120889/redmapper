

class Galaxy:

	def __init__(self, ra, dec):
		self.ra, self.dec = ra, dec
		self.neighbors = None

	def get_neighbors(self, galaxy_list):
		if self.neighbors: return self.neighbors
		# else, use htm_find_neighbors


class Cluster:

	def __init__(self, center):
		self.center = center
		self.members = None

	def ra(self): return self.center.ra
	def dec(self): return self.center.dec
		
	def get_potential_members(self):
		self.members = self.center.get_neighbors()
		return self.members
	
	def get_redshift(self): pass