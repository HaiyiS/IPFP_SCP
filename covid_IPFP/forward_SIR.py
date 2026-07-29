
# SIR forward model with two parameters: beta and gamma.

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
num_days = 100 
tspan = np.linspace(0, num_days, num_days)

S0 = 0.999
I0 = 0.001
R0 = 0.0
y0 = [S0, I0, R0]

def deriv1(t, y, beta, gamma):
    # beta is the transmission rate, gamma is the recovery rate
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


# SIR solution for plotting and testing

def SIR_solution(beta,gamma):
    sol = solve_ivp(deriv1, [0, num_days], y0, args=(beta, gamma), t_eval=tspan)
    return sol.t, sol.y[0], sol.y[1], sol.y[2]  # return time, S(t), I(t), R(t) 





def SIR_model(lam: np.ndarray, t) -> np.ndarray: 
    # lam is a 2d array of parameters each row is [beta,gamma]
    # return the solution of the SIR model at time t for each parameter set in lam
    # the return is 1-S(t), which represents cumulative infections
    beta = lam[:,0]
    gamma = lam[:,1]
    QoI = np.empty(lam.shape[0])
    
    for i in range(lam.shape[0]):
        # Run the differential equation solver
        sol = solve_ivp(deriv1, [0, t], y0, args=(beta[i], gamma[i]), t_eval=[t])
        
        # FIX 2: sol.y[0] returns an array like [0.85]. 
        # We need sol.y[0][0] to get the actual float value out of it.
        QoI[i] = 1 - sol.y[0][0]
        
    return QoI
SIR_model0 = lambda lam: SIR_model(lam, t= 1)
SIR_model0_1 = lambda lam: SIR_model(lam, t= 1.5)
SIR_model0_2 = lambda lam: SIR_model(lam, t= 2)
SIR_model0_3 = lambda lam: SIR_model(lam, t= 2.5)
SIR_model0_4 = lambda lam: SIR_model(lam, t= 3)
SIR_model0_5 = lambda lam: SIR_model(lam, t= 3.5)
SIR_model0_6 = lambda lam: SIR_model(lam, t= 4)
SIR_model0_7 = lambda lam: SIR_model(lam, t= 4.5)
SIR_model1 = lambda lam: SIR_model(lam, t= 5)
SIR_model2 = lambda lam: SIR_model(lam, t= 10)
SIR_model3 = lambda lam: SIR_model(lam, t=15)
SIR_model4 = lambda lam: SIR_model(lam, t=20)
SIR_model5 = lambda lam: SIR_model(lam, t=25)
SIR_model6 = lambda lam: SIR_model(lam, t=30)
SIR_model7 = lambda lam: SIR_model(lam, t=35)
SIR_model8 = lambda lam: SIR_model(lam, t=40)
SIR_model9 = lambda lam: SIR_model(lam, t=45)
SIR_model10 = lambda lam: SIR_model(lam, t=50)
SIR_model11 = lambda lam: SIR_model(lam, t=55)
SIR_model12 = lambda lam: SIR_model(lam, t=60)
SIR_model13 = lambda lam: SIR_model(lam, t=65)
SIR_model14 = lambda lam: SIR_model(lam, t=70)
SIR_model15 = lambda lam: SIR_model(lam, t=75)
SIR_model16 = lambda lam: SIR_model(lam, t=80)

#forward_models = [SIR_model1, SIR_model2, SIR_model3, SIR_model4, SIR_model5, SIR_model6, SIR_model7, SIR_model8, SIR_model9, SIR_model10, SIR_model11, SIR_model12, SIR_model13, SIR_model14, SIR_model15, SIR_model16]
#forward_models = [SIR_model1,  SIR_model3,  SIR_model5,  SIR_model7,  SIR_model9,SIR_model11,  SIR_model13,  SIR_model15]
#forward_models = [SIR_model1, SIR_model2, SIR_model10, SIR_model12]
forward_models = [SIR_model4, SIR_model6, SIR_model10]
#forward_models = [SIR_model1, SIR_model2]
#forward_models = [SIR_model2]
#forward_models =[SIR_model0,SIR_model0_1,SIR_model0_2,SIR_model0_3,SIR_model0_4,SIR_model0_5,SIR_model0_6,SIR_model0_7,SIR_model1, SIR_model2, SIR_model3, SIR_model4, SIR_model5, SIR_model6, SIR_model7, SIR_model8, SIR_model9, SIR_model10, SIR_model11, SIR_model12, SIR_model13, SIR_model14, SIR_model15, SIR_model16]
# --- QUICK TEST TO VERIFY IT WORKS ---
if __name__ == "__main__":
    # Test with two sets of [beta, gamma] parameters
    test_lam = np.array([
        [0.6, 0.2],  # R0 = 3.0 (Fast spread)
        [0.15, 0.1]  # R0 = 1.5 (Slow spread)
    ])
    
    print("Cumulative infections (1 - S) at Day 7:")
    print(forward_models[0](test_lam))
    
    print("\nCumulative infections (1 - S) at Day 14:")
    print(forward_models[1](test_lam))


# plot the SIR solution for a given set of parameters


plt.figure(figsize=(8, 6))
colors = ['blue', 'red'] # S, I, colors
beta,gamma = [[0.4,0.7,0.9],[0.2,0.5,0.7] ]
for i in range(len(beta)):
    t, S, I, R = SIR_solution(beta[i], gamma[i])
    plt.plot(t, S, label='S(t) (beta={:.3f},gamma = {:.3f})'.format(beta[i],gamma[i]), color=colors[0])
    plt.plot(t, I, label='I(t) (beta={:.3f},gamma = {:.3f})'.format(beta[i],gamma[i]), color=colors[1])

beta,gamma = [[1.25,1.5,1.75],[ 0.25,0.5,0.75] ]
for i in range(len(beta)):
    t, S, I, R = SIR_solution(beta[i], gamma[i])
    plt.plot(t, S, label='S(t) (beta={:.3f},gamma = {:.3f})'.format(beta[i],gamma[i]), color='green')   
    plt.plot(t, I, label='I(t) (beta={:.3f},gamma = {:.3f})'.format(beta[i],gamma[i]), color='orange')


plt.title(f'SIR Model Solution for Different Parameters)')
plt.xlabel('Time (days)')
plt.ylabel('Proportion of Population')
plt.legend()
plt.show()
