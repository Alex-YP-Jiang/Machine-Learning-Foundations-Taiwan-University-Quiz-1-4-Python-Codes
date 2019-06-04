>>> ## Quiz 4, Q13-Q20
>>> import math,random
>>> from pylab import *
>>> # Regularized Lin. Reg.(Ridge Regression) and (cross) validation.

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


>>> def ridge_reg(file,lmda):
	F = getList(file)
	Z = input_array(F)
	Z = array(Z)
	Y = array(labels(F))
	Z_trans = Z.transpose() # Array/list of equally sized arrays/lists is by default considered as matrix in Python, no need converting to 'np.matrix()'! 
	d = 3
	invs = matmul(Z_trans,Z) + lmda*identity(d) # 'np.identity(n)' creates a unit matrix as an array of n arrays.
	invs = linalg.inv(invs)
	mat = matmul(invs,Z_trans)
	w_regu = matmul(mat,Y)
	return w_regu

>>> w = ridge_reg('C:/Users/logic/Desktop/train.txt',10)
>>> w
array([ 1.04618645,  1.046171  ,  0.93238149])

>>> def error(w,file):
	F = getList(file)
	x = input_array(F)
	y = labels(F)
	N = len(y)
	err = 0
	for i in range(N):
		prod = w*x[i]
		s = prod.sum()
		if sign(s)!=y[i]:
			err +=1
	return err/N



>>> def ridge_reg_vali(file,lmda,num_D_train):    # Validation with number of D_train examples as argument, returns w_regu on D_train, prints E_train and E_val.
	F = getList(file)
	Z = input_array(F)
	Z = array(Z)
	Z_train = Z[:num_D_train]
	x_val = Z[num_D_train:]  # Segmenting the inital X/Z list into two parts(D_train/val) using a[n:].
	Y = array(labels(F))
	Y_train = Y[:num_D_train]
	y_val = Y[num_D_train:]
	Z_trans = Z_train.transpose()
	d = 3
	invs = matmul(Z_trans,Z_train) + lmda*identity(d)
	invs = linalg.inv(invs)
	mat = matmul(invs,Z_trans)
	w_regu = matmul(mat,Y_train)
	E_train = 0
	E_val = 0
	for i in range(num_D_train):   # calculates E_train
		prod = Z_train[i]*w_regu
		s = prod.sum()
		if sign(s)!= Y_train[i]:
			E_train+=1
	E_train = E_train/num_D_train  # calculates E_val
	N_val = len(y_val)
	for j in range(N_val):
		prod = x_val[j]*w_regu
		s = prod.sum()
		if sign(s)!= y_val[j]:
			E_val += 1
	E_val = E_val/N_val
	print('E_train: ',E_train,'; E_val: ',E_val,' with val. set size of ', N_val)
	return w_regu





>>> def ridge_reg_cv(file,lmda,V):    # Cross validation with V folds, return the E_cv given lambda.
	F = getList(file)
	Z = input_array(F)
	Z = array(Z)
	Y = array(labels(F))
	E_cv = 0
	N = len(Y)
	chunk_size = N/V
	chunks = array_split(Z,V)  # (numpy.)splitting the array of 200 input(x1,x2,x0) arrays to a list of V chunks, each chunk is an array of N/V input arrays.
	chunks_y = array_split(Y,V)
	for i in range(V):
		err = 0
		seg_start = int(i*chunk_size)  # The slice index in np.delete() have to be type of 'int'!!
		seg_end = int(i*chunk_size+chunk_size)
		Z_train = delete(Z, slice(seg_start,seg_end),axis = 0)  # (numpy.)deleting matrix rows of the validation chunk, 'axis = 0/1' for rows/columns.
		Y_train = delete(Y, slice(seg_start,seg_end),axis = 0)  # 'axis=0' is mandatory for matrix-shaped arrays, ie arrays with equally sized lists/arrays as elements!
		x_val = chunks[i]                                       # For simple arrays like 'Y' it's omissible. 'np.s_[seg_start:seg_end]' can also be used in 'delete()'.
		y_val = chunks_y[i]  # D_val for this run obtained
		Z_trans = Z_train.transpose()
		d = 3
		invs = matmul(Z_trans,Z_train) + lmda*identity(d)
		invs = linalg.inv(invs)
		mat = matmul(invs,Z_trans)
		w_regu = matmul(mat,Y_train)  # w_regu on D_train obtained
		for j in range(len(y_val)):
			prod = w_regu*x_val[j]
			s = prod.sum()
			if sign(s)!=y_val[j]:
				err += 1   # E_val of this run obtained
		E_cv += err/chunk_size
		#print(len(Z_train),len(Y_train),err)
	E_cv = E_cv/V
	return E_cv

>>> ridge_reg_cv('C:/Users/logic/Desktop/train.txt',10**(-8),5)
160 160 0
160 160 3
160 160 0
160 160 0
160 160 3
0.03
>>> 
