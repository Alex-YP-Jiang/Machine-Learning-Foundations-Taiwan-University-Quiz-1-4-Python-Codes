>>> # Quiz 2, Q17-20
>>> import math, random
>>> from pylab import *
>>>
>>> # Function that generates training examples regarding the given P(x) of input space and the noisy target function(for y_n).
>>> def training_set(N, x_l, x_r, noise):
	x = []
	y = []
	for i in range(N):   # Inputs of training examples are generated under P(x), has to be sorted for running dicotomy algo(for theta,s of each 'h')
		x_i = uniform(x_l,x_r)
		x.append(x_i)
	x.sort()
	for i in range(N):
		s = sign(x[i])  # 'sign()' is from numpy
		y_i = choice([s,-s], p=[1-noise,noise])  # random.choice(list, p=prob-list)
		#x.append(x_i)
		y.append(y_i)
	return [x,y]


>>> def dichotomy(points, index):  # The dichotomy() function is the key! It defines one specific dichotomy with 'index' argument and returns its E_in and hypothesis parameters.
	x = points[0]
	y = points[1]
	N = len(x)
	if index>2*N-1 or index<0 or type(index)!=int:
		raise ValueError('Inappropriate dichotomy index!')
	E_in, theta, s = 0, 0, 0  # Initialize the 3 key values of the dicotomy. Beware of the 'a,b = 0' mistake!
	## In terms of symmetry, N dicotomies are for positive ray and other N for negative ray. First dicotomies of both rays have a special theta, due to "no range".
	if index<N: s = 1
	else: s = -1
	if index == 0 or index == N:
		theta = x[0]-1
	else:
		if index<N:
			theta = 0.5*(x[index]-x[index-1]) + x[index-1]
		if index>N:
			theta = 0.5*(x[index-N]-x[index-N-1]) + x[index-N-1]  # Locating theta("median of range") with sorted x-list.
	y_dico = []  # Setting up the dicotomy outputs!!!
	if index<N:
		for i in range(N):
			y_dico.append(1)
		if index>0:
			for i in range(0,index):
				y_dico[i] = -1
	elif index>=N:
		for i in range(N):
			y_dico.append(-1)
		if index>N:
			for i in range(0,index-N):
				y_dico[i] = 1    # How cumbersome!!!
	error = 0  # Calculating E_D(dico)
	for i in range(N):
		if y_dico[i]!=y[i]:
			error+=1
	E_in = error/N
	return [s, theta, E_in]

>>> def DSA(n_exp, N, x_l, x_r, noise):
	E_D_mean = 0
	E_X_mean = 0
	for i in range(n_exp):
		t_data = training_set(N, x_l, x_r, noise)
		E_D_min = 1
		dico_min = 0
		for j in range(2*N):
			d = dichotomy(t_data, j)
			if d[2]<E_D_min:
				E_D_min = d[2]
				dico_min = j
		E_D_mean += E_D_min
		d_min = dichotomy(t_data, dico_min)
		s = d_min[0]
		theta = d_min[1]
		E_X_g = 0.5+0.3*s*(abs(theta)-1)
		E_X_mean += E_X_g
	E_D_mean = E_D_mean/n_exp
	E_X_mean = E_X_mean/n_exp
	return E_D_mean, E_X_mean

>>> DSA(5000,20, -1, 1, 0.2)
(0.1681100000000005, 0.25715611716437825)
>>> DSA(5000,20, -1, 1, 0.2)
(0.16749000000000003, 0.25488045851543445)
>>> 


>>> ## Multidimensional Decision Stump: Q19-20

>>> #  Run DSA with positive/negative rays(H) on each dimension of input x, choose the h(s,theta) from the 9 dimensions that has the lowest E_D on 1 of all 9 dimensions.
>>> def multi_d_examples(fname):  # returns the sorted x_d--y lists for running DSA/dicotomy()
	F = open(fname)
	L_strings = F.readlines()   # Re-use the file-reading codes from PLA.
	L_float_lists = []
	for l in L_strings:
		t1 = l.strip()
		t2 = t1.split()
		for i in range(len(t2)):
			t2[i] = float(t2[i])
		L_float_lists.append(t2)
  	all_d_data = []             # each element is a list of [x,y] with sorted x, like output of function 'training_set()'
	dimension = len(L_float_lists[0]) - 1
	N = len(L_float_lists)
	def takeFirst(L):    # Similar to lambda function, for the "key" in 'sorted()'.
		return L[0]
	for i in range(dimension):   # Collecting x-y pairs of every dimension and convert the data to the format of sorted [x,y].
		x = []
		y = []
		unsorted = []
		for j in range(N):
			x_y = [L_float_lists[j][i], L_float_lists[j][-1]]
			unsorted.append(x_y)
		ascend = sorted(unsorted, key = takeFirst) # The 'sorted()' function!
		for l in ascend:
			x.append(l[0])
			y.append(l[1])
		d_data = [x,y]
		all_d_data.append(d_data)
	return all_d_data  # A list of 'dimension'# of [x,y] lists, i.e [[x_0,y], [x_1,y].....]



>>> def DSA_multi_d(train, test):  # One function that covers Q19 and Q20.
	training_data = multi_d_examples(train)
	optimal_dim = 0  # Recording the dimension with lowest E_D, 'classification dimension'.
	E_D_d_min = 1    # Recording the lowest E_D of all dimensions, return for Q19.
	dico_best = []    # Recording the best h of all dimensions.
	dim = len(training_data)   # Calculating the # of dimension.
	N = len(training_data[0][0])  # Getting the size of training set of any dimension, ie # of inputs/labels.
	for i in range(dim):
		t_d = training_data[i]   # Running DSA on each dimension, recording the dimension with the lowest E_D and its parameters: dimension #, h-theta/s.
		E_D_min = 1
		dico_min = 0
		for j in range(2*N):
			d = dichotomy(t_d, j)
			if d[2]<E_D_min:
				E_D_min = d[2]
				dico_min = j
		if E_D_min < E_D_d_min:        # Final step!
			E_D_d_min = E_D_min
			optimal_dim = i
			dico_best = dichotomy(t_d, dico_min)  # Key elements(E_D_d_min, optimal_dim, dico_best) all attained!
	test_data = multi_d_examples(test)
	error_test = 0
	test_d_x = test_data[optimal_dim][0]   # Getting the x and its outputs y of the classification dimension.
	test_d_y = test_data[optimal_dim][1]
	s = dico_best[0]
	theta = dico_best[1]
	for i in range(len(test_d_y)):
		h = s*sign(test_d_x[i]-theta)
		if h!=test_d_y[i]:
			error_test+=1
	E_test = error_test/len(test_d_y)
	print('Minimal E_in is: ',E_D_d_min,'; E_test is: ',E_test)

	
>>> DSA_multi_d('C:/Users/logic/Desktop/train.txt', 'C:/Users/logic/Desktop/test.txt')
Minimal E_in is:  0.25 ; E_test is:  0.355
>>> 
