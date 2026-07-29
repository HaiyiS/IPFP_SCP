
# define a forward model/computer codes from parameter domain to data range.
# Here consider the simpliest forward model with two parameters: a0 * lam0 + a1* lam1,
# where a  are coefficients, which defines different forward models. 
import numpy as np



def main_model(lam: np.ndarray, a: np.ndarray) -> float:
    # lam is a 2d array of parameters
    # a are coefficients
    
    return a[0] * lam[:, 0]  + a[1] * (lam[:, 1])**3

# create different forward models by changing the parameter a

f_1 = lambda lam: main_model(lam, a = np.array([0,1]))
f_2 = lambda lam: main_model(lam, a = np.array([0.1,0.9]))
f_3 = lambda lam: main_model(lam, a = np.array([0.2,0.8]))
f_4 = lambda lam: main_model(lam, a = np.array([0.3,0.7]))
f_5 = lambda lam: main_model(lam, a = np.array([0.4,0.6]))
f_6 = lambda lam: main_model(lam, a = np.array([0.5,0.5]))
f_7 = lambda lam: main_model(lam, a = np.array([0.6,0.4]))
f_8 = lambda lam: main_model(lam, a = np.array([0.7,0.3]))
f_9 = lambda lam: main_model(lam, a = np.array([0.8,0.2]))
f_10 = lambda lam: main_model(lam, a = np.array([0.9,0.1]))     
f_11 = lambda lam: main_model(lam, a = np.array([1,0]))
f_12 = lambda lam: main_model(lam, a = np.array([1,-0.1]))
f_13 = lambda lam: main_model(lam, a = np.array([1,-0.2]))
f_14 = lambda lam: main_model(lam, a = np.array([1,-0.3]))
f_15 = lambda lam: main_model(lam, a = np.array([1,-0.4]))
f_16 = lambda lam: main_model(lam, a = np.array([1,-0.5]))
f_17 = lambda lam: main_model(lam, a = np.array([1,-0.6]))
f_18 = lambda lam: main_model(lam, a = np.array([1,-0.7]))




def main_model2(lam: np.ndarray, a:float ) -> float:
    # lam is a 2d array of parameters
    # a are coefficients
    
    return lam[:,0] * np.exp(-lam[:,1]*a)

g_1 = lambda lam: main_model2(lam, a = 0.01)
g_2 = lambda lam: main_model2(lam, a = 0.05)
g_3 = lambda lam: main_model2(lam, a = 0.1)
g_4 = lambda lam: main_model2(lam, a = 0.2)
g_5 = lambda lam: main_model2(lam, a = 0.3)
g_6 = lambda lam: main_model2(lam, a = 0.4)
g_7 = lambda lam: main_model2(lam, a = 0.5)
g_8 = lambda lam: main_model2(lam, a = 0.6)
g_9 = lambda lam: main_model2(lam, a = 0.7)
g_10 = lambda lam: main_model2(lam, a = 0.8)
g_11 = lambda lam: main_model2(lam, a = 0.9)
g_12 = lambda lam: main_model2(lam, a = 1.0)



#######make a list of forward models 
#forward_models =[g_1,g_2,g_3,g_4,g_5,g_6,g_7]
#forward_models = [g_6,g_7,g_8,g_9,g_10,g_11,g_12]
#forward_models = [g_1,g_2,g_3]
forward_models = [g_1,g_2,g_3,g_4,g_5,g_6,g_7,g_8,g_9,g_10,g_11,g_12]
#forward_models = [g_1]
#forward_models = [g_1,g_2,g_3,g_5,g_4]