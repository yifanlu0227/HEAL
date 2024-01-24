"""
Pytorch implementation of the logarithm of the modified Bessel function 
of the 1st kind I(nu, z).
Based mainly on scipy and extends the definition domain compared to 
scipy methods (which can easily provide infinite values). The extension
is done using numerical approximations involving the ratio of Bessel 
functions:   
-	https://arxiv.org/pdf/1606.02008.pdf (Theorems 5,6)
-	https://arxiv.org/pdf/1902.02603.pdf (Appendix)
Jean-Remy Conti
2021
"""

import torch
from scipy import special, pi


def logbessel_I_scipy(nu, z, check = True):
	'''
	Pytorch version of scipy computation of modified Bessel functions 
	of the 1st kind I(nu,z).
	Parameters
	----------
	nu: positive int, float
		Order of modified Bessel function of 1st kind.
	z: int/float or tensor, shape (N,) 
		Argument of Bessel function.
	check: bool
		If True, check if argument of log is non zero.
	
	Return
	------
	result: tensor, shape (N,)
	'''
	if not isinstance(z, torch.Tensor):
		z = torch.tensor(z)
	z = z.reshape(-1)

	result = special.ive(nu, z)
	if check:
		assert len(result[ result == 0]) == 0, ('Bessel functions take ' +
												'value 0 for z = {}'.format(
												z[ result == 0]))
	result = torch.log(result) + z
	return result



def logbessel_I_asymptotic(nu, z):
	'''
	Asymptotic branches of the modified Bessel function of the 1st kind.
	https://arxiv.org/pdf/1902.02603.pdf
	Parameters
	----------
	nu: positive int, float
		Order of modified Bessel function of 1st kind.
	z: tensor, shape (N,) 
		Argument of Bessel function.
	
	Return
	------
	result: tensor, shape (N,)
	'''
	z = z.double()
	eta = (nu + 0.5)/(2* (nu+1) )
	result = torch.zeros(z.shape[0]).double()

	result[ z <= nu ] = (
						nu*torch.log(z[ z <= nu ]) + eta*z[ z <= nu ] 
						- (nu+eta)*torch.log(torch.tensor(2.))
						- torch.log( torch.tensor(special.gamma(nu+1)) )
						)
	
	result[ z > nu ] = (
						z[ z > nu ] - 0.5*torch.log(z[ z > nu ]) 
						- 0.5*torch.log(torch.tensor(2*pi)) 
						)
	return result

def B(alpha, nu, z):
	'''
	https://arxiv.org/pdf/1606.02008.pdf
	'''
	nu = nu.reshape(1,-1)
	z = z.reshape(-1,1)
	lamda = nu + float(alpha-1)/2. 
	delta = nu-0.5 + lamda / (2*torch.sqrt(lamda**2 + z**2))
	return z / (delta + torch.sqrt(delta**2 + z**2) )
def B_tilde(alpha, nu, z):
	'''
	https://arxiv.org/pdf/1606.02008.pdf
	'''
	nu = nu.reshape(1,-1)
	z = z.reshape(-1,1)
	sigma = nu + float(alpha+1)/2.
	delta_p = nu + 0.5 + sigma/(2*torch.sqrt(sigma**2 + z**2))
	delta_m = nu - 0.5 - sigma/(2*torch.sqrt(sigma**2 + z**2))
	return z/( delta_m + torch.sqrt(delta_p**2 + z**2))
def lb_Ak(nu, z):
	'''
	Lower-bound for the ratio of modified Bessel functions of 1st kind. 
	https://arxiv.org/pdf/1606.02008.pdf (Theorems 5 and 6).
	'''
	assert torch.all(nu >= 0)
	nu = nu.reshape(1,-1)
	z = z.reshape(-1,1)
	return B_tilde(0, nu, z)
def ub_Ak(nu, z):
	'''
	Upper-bound for the ratio of modified Bessel functions of 1st kind.
	https://arxiv.org/pdf/1606.02008.pdf (Theorems 5 and 6).
	Return
	------
	ub: tensor, shape (z.shape[0], nu.shape[0])
		Upper-bound for Ak(nu, z).
	'''
	assert torch.all(nu >= 0)
	nu = nu.reshape(1,-1)
	z = z.reshape(-1,1)

	ub = torch.zeros(z.shape[0], nu.shape[1])
	ub[:, nu.reshape(-1) >= 0.5] = torch.min(B(0, nu[ nu >= 0.5 ], z), 
											B_tilde(2, nu[ nu >= 0.5 ], z))
	ub[:, nu.reshape(-1) < 0.5] = B_tilde(2, nu[ nu < 0.5 ], z) 
	return ub

def Ak_approx(nu, z):
	'''
	Approximation of ratio of modified Bessel functions of 1st kind.
	https://arxiv.org/pdf/1902.02603.pdf
	Parameters
	----------
	nu: tensor, shape (N0,)
		Order of modified Bessel functions of 1st kind.
	z: tensor, shape (N1,) 
		Argument of Bessel function. Positive values only.
	
	Return
	------
	tensor, shape (N1, N0)
	'''
	return 0.5*(lb_Ak(nu, z) + ub_Ak(nu, z)) 


def logbessel_I_approx(nu, z):
	'''
	Approximation of the logarithm of the modified Bessel function of 
	1st kind I(nu, z) using their ratio.
	https://arxiv.org/pdf/1902.02603.pdf
	Parameters
	----------
	nu: positive int, float
		Order of modified Bessel function of 1st kind.
	z: tensor, shape (N,) 
		Argument of Bessel function. Positive values only.
	Return
	------
	approx: tensor, shape (N,)
	'''
	assert nu >= 0
	approx = logbessel_I_scipy(nu-int(nu), z, check= True) 
	nu_v = nu - torch.arange(0, int(nu))
	A = Ak_approx(nu_v, z) 
	approx += torch.log(A).sum(axis= 1)
	return approx


def logbessel_I(nu, z, fast = False, check = True):
	'''
	Pytorch implementation of the logarithm of the modified Bessel 
	function of the 1st kind I(nu, z).
	Based mainly on scipy and extends the definition domain compared to 
	scipy methods (which can easily provide infinite values). The 
	extension is done using numerical approximations involving the ratio
	of Bessel functions:   
	-	https://arxiv.org/pdf/1606.02008.pdf (Theorems 5,6)
	-	https://arxiv.org/pdf/1902.02603.pdf (Appendix)
	Parameters
	----------
	nu: positive int, float
		Order of modified Bessel function of 1st kind.
	z: int/float or tensor, shape (N,) 
		Argument of Bessel function.
	fast: bool
		If True, use asymptotic behavior as approximation when main 
		scipy method is not tractable. If False, use tight bounds for 
		the ratio of Bessel functions:
		https://arxiv.org/pdf/1902.02603.pdf
	check: bool
		If True, check if argument of log is non zero and not NaN.
	
	Return
	------
	result: tensor, shape (N,)
	'''
	if not isinstance(z, torch.Tensor):
		z = torch.tensor(z)
	z = z.reshape(-1)

	result = special.ive(nu, z)
	# Indices for which scipy.special.ive is wrong
	bad_idx = torch.arange(result.shape[0])[ result == 0]
	result = torch.log(result) + z
	if fast:
		result[ bad_idx ] = logbessel_I_asymptotic(nu, z[ bad_idx ])
	else: 	 
		result[ bad_idx ] = logbessel_I_approx(nu, z[ bad_idx ])

	if check:
		# If problem with assertion, use a better defined init in 
		# logbessel_approx
		assert len(result[ torch.isnan(result) ]) == 0, ('Bessel functions take ' +
												'NaN value for z = {}'.format(
		 										z[ torch.isnan(result) ]))
		assert len(result[ torch.isinf(result) ]) == 0, ('Bessel functions take ' +
		 										'inf value for z = {}'.format(
		 										z[ torch.isinf(result) ]))
	return result



if __name__ == "__main__":

	import time
	import matplotlib.pyplot as plt

	# ------- Single test ------- #
	print(logbessel_I(nu= 10000, z= 10000))


	# ------- Vectorized tests ------- #
	nu = 1000
	z = torch.arange(1, 200001)

	# Scipy
	print('------\nScipy adaptation:')
	start = time.time()
	scipy_adapt = logbessel_I_scipy(nu, z, check= False)
	end = time.time()
	print('Computation time: ', "%.4f"%(end-start), 's')

	# Asymptotic
	print('------\nAsymptotic computation:')
	start = time.time()
	asympt = logbessel_I_asymptotic(nu, z)
	end = time.time()
	print('Computation time: ', "%.4f"%(end-start), 's')

	# Approximation using ratios of Bessel functions
	print('------\nApproximation via ratios of Bessel functions:')
	start = time.time()
	approx = logbessel_I_approx(nu, z)
	end = time.time()
	print('Computation time: ', "%.4f"%(end-start), 's')

	# Fast extension of scipy
	print('------\nFast method (proposed):')
	start = time.time()
	logbessels_fast = logbessel_I(nu, z, fast= True, check= False)
	end = time.time()
	print('Computation time: ', "%.4f"%(end-start), 's')

	# Precise extension of scipy
	print('------\nPrecise method (proposed):')
	start = time.time()
	logbessels = logbessel_I(nu, z, fast= False, check= False)
	end = time.time()
	print('Computation time: ', "%.4f"%(end-start), 's')


	# ------- Plots ------- #
	linewidth = 4

	# Plot different methods
	plt.plot(z, scipy_adapt, '-', label='scipy', linewidth = linewidth)
	plt.plot(z, asympt, '--', label='asymptotic', linewidth = linewidth)
	plt.plot(z, logbessels, '--', label='ours', linewidth = linewidth)
	plt.xlabel(r'$z$')	
	plt.title(r'$\log[I(\nu = {}, z)]$'.format(nu))
	plt.legend()

	# Plot relative error of approximation
	plt.figure()
	plt.plot(z, torch.abs((approx - scipy_adapt)/scipy_adapt), '-', label=None, linewidth = linewidth)
	plt.xlabel(r'$z$')
	plt.title(r'Relative error of approximation for $\log[I(\nu = {}, z)]$'.format(nu))

	plt.show()