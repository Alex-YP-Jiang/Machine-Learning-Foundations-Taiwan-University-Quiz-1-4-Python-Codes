>>> # Quiz 3
>>> # Q13-15  1st order and feature-transformed(2nd order) Linear Regression
>>> import random, math
>>> from pylab import*


>>> def training_set(N,noise,x_l,x_r):  # Generates noisy quadratic-separable binary class. data points.
	x_1 = []
	x_2 = []
	y = []
	for i in range(N):
		x1 = uniform(x_l,x_r)  # The bug with 'uniform([,])' induced interesting stuff: 1).random.choice() vs. numpy.random.choice(); 2).'sign(array)' gives an array of
		x2 = uniform(x_l,x_r)  # signs! 3).'a has to be 1-dimensional' for numpy's choice()!!!
		s = sign(x1**2 + x2**2 - 0.6)
		y_noise = choice([s,-s], p = [1-noise, noise])
		x_1.append(x1)
		x_2.append(x2)
		y.append(y_noise)
	return [x_1,x_2,y]



>>> def lin_reg(D):  # D is the output of the function above.
	x_1 = D[0]
	x_2 = D[1]
	y = array(D[2])
	x0 = 1
	N = len(y)
	input_matrix = []
	for i in range(N):
		x_n = [x0,x_1[i],x_2[i]]
		input_matrix.append(x_n)
	X = matrix(input_matrix)  # 'numpy.matrix([row1,row2...])'
	X_psu_inv = linalg.pinv(X)  # 'numpy.linalg.pinv(M)' for getting the pseudo inverse of M
	w_lin = matmul(X_psu_inv, y)  # 'numpy.matmul()' can have arguments as matrix,array or lists, it returns arrays if arguments are lists/arrays.
	a = array(w_lin)  # To reduce the output matrix to an array that contains only 1 array as its element.
	w_lin = a[0]
	return w_lin


>>> def E_in(w, D):
	x_1 = D[0]
	x_2 = D[1]
	y = D[2]
	N = len(y)
	err = 0
	for i in range(N):
		x_n = array([1, x_1[i], x_2[i]])
		scalar = x_n*w
		if sign(scalar.sum())!= y[i]:
			err+=1
	return err/N

>>> def experiment(N, noise, x_l, x_r, num_exp):
	err_mean = 0
	for i in range(num_exp):
		D = training_set(N,noise,x_l,x_r)
		w_lin = lin_reg(D)
		err = E_in(w_lin,D)
		err_mean += err
	return err_mean/num_exp

>>> experiment(1000,0.1,-1,1,1000)
0.5087710000000005
>>> experiment(1000,0.1,-1,1,1000)
0.5081229999999999
>>> experiment(1000,0.1,-1,1,1000)
0.5100310000000006
>>> # The w_lin acquired on linear/original inputs has poor performance(50% error), because g(x) isn't close to target function(f(x) = sign(x1**2 + x2**2 - 0.6)).


>>> # Q14,15: Run LinReg on the quadratic transformed (Q = 2) training data, find its w_lin_telta.
>>> def z_n(D):  # Outputs a list of all z_n ([1,x_1,...,x_2**2])/"Transformed inputs".
	zn = []
	x_1 = D[0]
	x_2 = D[1]
	N = len(x_1)
	for i in range(N):
		z_i = [1,x_1[i],x_2[i],x_1[i]*x_2[i],x_1[i]**2,x_2[i]**2]
		zn.append(z_i)
	return zn

>>> def lin_reg_trans(D):
	y = array(D[2])
	inputs = z_n(D)
	input_matrix = matrix(inputs)
	X_p_i = linalg.pinv(input_matrix)  # d_z*N matrix
	w = matmul(X_p_i,y)
	w_array = array(w)
	w_lin_trans = w_array[0]
	return w_lin_trans

>>> D = training_set(2000,0.1,-1,1)
>>> lin_reg_trans(D)
array([-1.23052778,  0.0115436 ,  0.00437408, -0.05369831,  1.96227535,
        1.96963185])
>>> D = training_set(5000,0.1,-1,1)
>>> lin_reg_trans(D)
array([-1.25003646,  0.00471739, -0.00247692,  0.0223822 ,  1.98585132,
        1.93721312])   # The w_lin_trans of d_z dimensions creates a g(x) that's similar to the target function, ie. g(x)=sign(w*z)=sign(phi(x)), where phi(x) is a 2nd order
                       # polynomial that has similar coefficients for x1**2 and x2**2, which forms a circle as target f(x) does.

>>> def E_out(w, D):
	x_1 = D[0]
	x_2 = D[1]
	y = D[2]
	N = len(y)
	err = 0
	for i in range(N):
		x_n = array([1, x_1[i], x_2[i],x_1[i]*x_2[i],x_1[i]**2,x_2[i]**2])
		scalar = x_n*w
		if sign(scalar.sum())!= y[i]:  # Binary classification done on the Z-space with w_lin_trans and z_n of d_z dimensions. The real input or point to be classified
			err+=1                 # is still x_n = (x1, x2). Since g(x) here is close to f(x), the misclass. rate is much lower (under 3%).
	return err/N

>>> def trans_linreg_performance(num_runs):
	mean_E_out = 0
	for i in range(num_runs):
		D_train = training_set(1000,0.1,-1,1)
		w_lin_trans = lin_reg_trans(D_train)
		D_test = training_set(1000,0.1,-1,1)
		err = E_out(w_lin_trans, D_test)
		mean_E_out += err
	return mean_E_out/num_runs

>>> trans_linreg_performance(1000)
0.02957899999999997
>>> trans_linreg_performance(1000)
0.02954499999999995
>>> 

