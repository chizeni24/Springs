def obj_func(weights):
    return 0.5*np.dot(np.array(weights)**2
                      ,np.array([1,1]))

def obj_funct(w):
    return 0.5*(w[0]**2 + w[1]**2)


def grad_func(learning_rate,weights):
    return -learning_rate*weights

def cost_function_MSE():
    
    w = np.array([1,1])  #can be randomized
    feature_len = len(w)
    func_history = obj_func(w)
    w_history = w
    k = 0
    maxit = 10
    cost = func_history
    cost_history = cost
    tolerance = 0.1

    while k < maxit or cost > tolerance:

        w = w + grad_func(0.1, w)

        w_history = np.vstack((w_history,w))
        func_history = np.vstack((func_history,obj_func(w)))
        k+=1
        cost = np.abs(func_history[-1]-func_history[-2])
        cost_history = np.vstack((cost_history,cost))


    return w_history, func_history , cost_history


 
cost_function_MSE()