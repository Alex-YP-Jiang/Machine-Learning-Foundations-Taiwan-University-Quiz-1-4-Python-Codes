>>> import math,random
>>> from pylab import*
>>> # Quiz 3, Q18-20 Logistic Regression
>>> def getList(fname):  # Read the .txt file that contains the training examples(x_n, y_n), processing it to a list of lists in float.
	F = open(fname)
	L_strings = F.readlines() # returns a list of strings, each line in file is a string needs to be processed
	L_float_lists = []
	for l in L_strings:
		t1 = l.strip()
		t2 = t1.split()
		for i in range(len(t2)):
			t2[i] = float(t2[i])
		L_float_lists.append(t2)
	return L_float_lists

>>> def input_array(List):  # Converts the raw list of training examples to a list of input arrays with -1 as an extra attribute value for w's threshold.
	array_list = []
	for l in List:
		L = array(l)
		L[-1] = -1
		array_list.append(L)
	return array_list

>>> def labels(List):  # saves the y_n of sample into a list
        label = []
        for l in List:
                label.append(l[-1])
        return label

>>> def theta(x_n,y_n,w):     # Calculating 'theta(-y_n*w*x_n)'
	product = w*x_n
	scalar = product.sum()
	s = -y_n*scalar
	return 1/(1 + exp(-s))

>>> def Grad_CE(w,x_n,y_n):  # Calculating the gradient of cross entropy error.
	N = len(y_n)
	d = len(x_n[0])     # d is actually 'd+1'
	grad = [0.0000]*d   # '[0]' yields a TypeError at 'grad+=vec', since initial 'grad' is type of 'int(32)' and 'vec' is of 'float(64)'.
	grad = array(grad) 
	for i in range(N):  # Calculating the sum of vectors over entire 'D'(n from 1 to N).
		sig = theta(x_n[i],y_n[i],w)
		vec = -y_n[i]*sig*x_n[i]
		grad += vec
	return (1/N)*grad

>>> def log_reg(ita,T,file):
	F = getList(file)
	x_n = input_array(F)
	y_n = labels(F)
	d = len(x_n[0])    # 'd' is actually d+1, same as in Grad_CE()
	w = [0]*d
	w = array(w)
	for i in range(T):
		grad = Grad_CE(w,x_n,y_n)
		w = w - ita*grad
	return w

>>> def E_out(w,file):
	F = getList(file)
	x_n = input_array(F)
	y_n = labels(F)
	N = len(y_n)
	err = 0
	for i in range(N):
		p = w*x_n[i]
		s = p.sum()  # check the sign of 'score', no need for calculating theta function/h(x) (monotonic increasing, theta(0) = 0.5)
		if s*y_n[i]<0:
			err += 1
	return err/N

>>> w = log_reg(0.001,2000,'C:/Users/logic/Desktop/train.txt')
>>> E_out(w,'C:/Users/logic/Desktop/test.txt')
0.475
>>> w = log_reg(0.01,2000,'C:/Users/logic/Desktop/train.txt')
>>> E_out(w,'C:/Users/logic/Desktop/test.txt')   # Ita closer to '0.1', yields better performance.
0.22


>>> def log_reg_stoch(ita, T, file):  # Log. reg. with stochastic grad. descent, x_n not random but in cyclic order regarding T.
	F = getList(file)
	x_n = input_array(F)
	y_n = labels(F)
	N = len(y_n)
	d = len(x_n[0])
	w = [0]*d
	w = array(w)
	flag = True
	index = 0
	t = 0
	while flag:
		sig = theta(x_n[index],y_n[index],w)
		w = w + ita*sig*y_n[index]*x_n[index]
		t+=1
		index+=1
		if t>=T:
			flag = False
		if index == N:
			index = 0    # same iteration process as in Pocket algorithm
	return w

>>> w = log_reg_stoch(0.001,2000,'C:/Users/logic/Desktop/train.txt')  # Interestingly, SGD didn't result in a 'looser' error.
>>> E_out(w,'C:/Users/logic/Desktop/test.txt')
0.473
>>> w = log_reg_stoch(0.001,20000,'C:/Users/logic/Desktop/train.txt')
>>> E_out(w,'C:/Users/logic/Desktop/test.txt')
0.22033333333333333
>>> ## Note: When T=20000>>2000, a smaller grad_ce and a better w(lower E_out/in) are resulted. The running of log_reg_stoch() is indeed much faster!! O(1) each iteration compared
>>> #        to O(N) per iteration for regular log. reg, ie. 'one point/(xn,yn)' vs. gradient calculation with N points!
>>> 
