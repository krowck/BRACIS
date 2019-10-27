###############################################################################
# Version: 1.1
# Last modified on: 3 April, 2016 
# Developers: Michael G. Epitropakis
#      email: m_(DOT)_epitropakis_(AT)_lancaster_(DOT)_ac_(DOT)_uk 
###############################################################################
from scipy.spatial.distance import pdist, squareform
import numpy as np
import math
from functions import *
from cfunction import *
from CF1 import *
from CF2 import *
from CF3 import *
from CF4 import *

class CEC2013(object):
	__nfunc_ = -1
	__functions_ = {1:five_uneven_peak_trap, 2:equal_maxima, 3:uneven_decreasing_maxima, 
			4:himmelblau, 5:six_hump_camel_back, 6:shubert, 7:vincent, 8:shubert, 9:vincent,
			10:modified_rastrigin_all, 11:CF1, 12:CF2, 13:CF3, 14:CF3, 15:CF4, 16:CF3, 
			17:CF4, 18:CF3, 19:CF4, 20:CF4}
	__f_ = None
	__fopt_ = [200.0, 1.0, 1.0, 200.0, 1.031628453489877, 186.7309088310239, 1.0, 2709.093505572820, 1.0, -2.0, 
				0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
	__rho_ = [0.01, 0.01, 0.01, 0.01, 0.5, 0.5, 0.2, 0.5, 0.2, 0.01, 
			0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01 ]
	__nopt_ = [2, 5, 1, 4, 2, 18, 36, 81, 216, 12, 6, 8, 6, 6, 8, 6, 8, 6, 8, 8 ]
	__maxfes_ = [50000, 50000, 50000, 50000, 50000, 200000, 200000, 400000, 400000, 200000, 
			200000, 200000, 200000, 400000, 400000, 400000, 400000, 400000, 400000, 400000 ]
	__dimensions_ = [1, 1, 1, 2, 2, 2, 2, 3, 3, 2, 2, 2, 2, 3, 3, 5, 5, 10, 10, 20]

	def __init__(self, nofunc):
		assert (nofunc > 0 and nofunc <= 20)
		self.__nfunc_ = nofunc
		if (self.__nfunc_ > 0 and self.__nfunc_ < 11): 
			self.__f_ = self.__functions_[self.__nfunc_]
		else:
			self.__f_ = self.__functions_[self.__nfunc_]( self.get_dimension() )

	def evaluate(self, x):
		assert (len(x) == self.get_dimension())
		if (self.__nfunc_ > 0 and self.__nfunc_ < 11): 

			return self.__f_(x)
		else:
			#print(x)
			return self.__f_.evaluate(x)

	def get_lbound(self, n):
		assert (n >= 0 and n <self.__dimensions_[self.__nfunc_-1])
		result = 0
		if (self.__nfunc_ == 1 or self.__nfunc_ == 2 or self.__nfunc_ == 3):
			result = 0
		elif (self.__nfunc_ == 4):
			result = -6
		elif (self.__nfunc_ == 5):
			tmp = [-1.9, -1.1]
			result = tmp[n]
		elif (self.__nfunc_ == 6 or self.__nfunc_ == 8):
			result = -10
		elif (self.__nfunc_ == 7 or self.__nfunc_ == 9):
			result = 0.25
		elif (self.__nfunc_ == 10):
			result = 0
		elif (self.__nfunc_ > 10):
			result = self.__f_.get_lbound(n)
		return result

	def get_ubound(self, n):
		assert (n >= 0 and n <self.__dimensions_[self.__nfunc_-1])
		result = 0
		if (self.__nfunc_ == 1):
			result = 30
		elif (self.__nfunc_ == 2 or self.__nfunc_ == 3):
			result = 1
		elif (self.__nfunc_ == 4):
			result = 6
		elif (self.__nfunc_ == 5):
			tmp = [1.9, 1.1]
			result = tmp[n]
		elif (self.__nfunc_ == 6 or self.__nfunc_ == 8):
			result = 10
		elif (self.__nfunc_ == 7 or self.__nfunc_ == 9):
			result = 10
		elif (self.__nfunc_ == 10):
			result = 1
		elif (self.__nfunc_ > 10):
			result = self.__f_.get_ubound(n)
		return result

	def get_fitness_goptima(self):
		return self.__fopt_[self.__nfunc_-1]
	def get_dimension(self):
		return self.__dimensions_[self.__nfunc_-1]
	def get_no_goptima(self):
		return self.__nopt_[self.__nfunc_-1]
	def get_rho(self):
		return self.__rho_[self.__nfunc_-1]
	def get_maxfes(self):
		return self.__maxfes_[self.__nfunc_-1]
	def get_info(self):
		return {'fbest': get_fitness_goptima(), 
				'dimension': get_dimension(),
				'nogoptima': get_no_goptima(),
				'maxfes': get_maxfes(),
				'rho': get_rho()}

def how_many_goptima(pop, f, accuracy, popsize, pop_aux):
	# pop: NP, D

	# pop_aux[0] = [-3.395113021, -3.317307]
	# pop_aux[1] = [4.14125693, 2.4770118101]
	# pop_aux[2] = [-0.499695065, -4.012597084]
	# pop_aux[3] = [-2.184984351, 1.687053938]
	# pop_aux[4] = [1.7577435, 1.5957372547]
	# pop_aux[5] = [-1.561544695, 4.400020671757]
	# pop_aux[6] = [0.1628760722, 3.78150997]
	# pop_aux[7] = [-1.481471, 3.674022685]

	# print(pop)
	# print(pop_aux)

	pop_aux = np.array(pop_aux)

	# print(pop_aux)
	NP = pop_aux.shape[0]

	D = pop_aux.shape[1]

	# pop[0] = [-3.395113021, -3.317307]
	# pop[1] = [4.14125693, 2.4770118101]
	# pop[2] = [-0.499695065, -4.012597084]
	# pop[3] = [-2.184984351, 1.687053938]
	# pop[4] = [1.7577435, 1.5957372547]
	# pop[5] = [-1.561544695, 4.400020671757]
	# pop[6] = [0.1628760722, 3.78150997]
	# pop[7] = [-1.481471, 3.674022685]

	#print("SIZE: ", NP, D)
	#print(pop)
	# Evaluate population
	fits = np.zeros( popsize )
	for i in range( popsize ):
		fits[i] = f.evaluate(pop[i])
		#print(fits[i])

	# Descenting sorting
	order = np.argsort(fits)[::-1]

	# Sort population based on its fitness values
	sorted_pop = pop_aux[order]
	spopfits = fits[order]
	
	# find seeds in the temp population (indices!)
	seeds_idx = find_seeds_indices(sorted_pop, f.get_rho() )

	
	count = 0
	goidx = []
	fitness = []
	#print(sorted_pop)
	for idx in seeds_idx:
		# evaluate seed
		seed_fitness = spopfits[idx] #f.evaluate(sorted_pop[idx])

		# |F_seed - F_goptimum| <= accuracy
		if math.fabs( seed_fitness - f.get_fitness_goptima() ) <= accuracy:
			count = count + 1
			goidx.append(idx)

		# save time
		if count == f.get_no_goptima():
			break

	# gather seeds
	seeds = sorted_pop[goidx]
	#print(seeds)
	return count, seeds

def find_seeds_indices(sorted_pop, radius):
	#print(sorted_pop)
	seeds = []
	seeds_idx = []
	# Determine the species seeds: iterate through sorted population 
	for i, x in enumerate(sorted_pop):
		found = False 
		# Iterate seeds
		for j, sx in enumerate(seeds):
			# Calculate distance from seeds
			dist = math.sqrt( sum( (x - sx)**2 ) )

			# If the Euclidean distance is less than the radius
			if dist <= radius:
				found = True
				break
		if not found:
			seeds.append(x)
			seeds_idx.append(i)
	#print(seeds)
	return seeds_idx


# def how_many_goptima(pop, f, accuracy, popsize, pop_aux):
# 	# pop: NP, D
# 	#NP, D = pop.shape[0], pop.shape[1]
# 	NP = popsize

# 	# Evaluate population
# 	fits = np.zeros( NP )
# 	for i in range( NP ):
# 		fits[i] = f.evaluate(pop[i])

# 	# Descenting sorting
# 	order = np.argsort(fits)[::-1]
# 	print np.array(pop_aux)
# 	# # Sort population based on its fitness values
# 	# sorted_pop.sort()
# 	# print sorted_pop
# 	# sorted_pop = pop_aux[order,:]
# 	# print sorted_pop
# 	pop_aux.sort()
# 	spopfits = fits[order]
# 	seeds = []

# 	# find seeds in the temp population (indices!)
# 	seeds_idx = find_seeds_indices(pop_aux, f.get_rho() )
	
# 	print seeds_idx
# 	count = 0
# 	goidx = []
# 	for idx in seeds_idx:
# 		# evaluate seed
# 		seed_fitness = spopfits[idx] #f.evaluate(sorted_pop[idx])

# 		# |F_seed - F_goptimum| <= accuracy
# 		if math.fabs( seed_fitness - f.get_fitness_goptima() ) <= accuracy:
# 			count = count + 1
# 			goidx.append(idx)

# 		# save time
# 		if count == f.get_no_goptima():
# 			break

# 	# gather seeds
# 	for i in range(0,len(goidx)):		
# 		seeds.append(goidx)
# 		#seeds = pop_aux[goidx]

# 	return count, seeds

# def find_seeds_indices(sorted_pop, radius):
# 	seeds = []
# 	seeds_idx = []
	
# 	# Determine the species seeds: iterate through sorted population 
# 	for i, x in enumerate(sorted_pop):
# 		a = 0.0
# 		found = False 
# 		# Iterate seeds
# 		for j, sx in enumerate(seeds):
# 			# Calculate distance from seeds
# 			a += (x-sx)**2
# 			dist = math.sqrt(a)
# 		#math.sqrt( sum( (x - sx)**2 ) )

# 		# If the Euclidean distance is less than the radius
# 			if dist <= radius:
# 				found = True
# 				break
# 		if not found:
# 			seeds.append(x)
# 			seeds_idx.append(i)

# 	return seeds_idx

