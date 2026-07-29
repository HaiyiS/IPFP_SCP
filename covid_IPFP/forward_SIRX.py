# SIR forward model with three parameters: beta, gamma, and initial susceptible proportion
# we model susceptible proportion as random to account for uncertainty in initial conditions. Also because we do not know the exact start date of the surge
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
num_days = 120
tspan = np.linspace(0, num_days, num_days)

def deriv1(t, y, beta, gamma):
    # beta is the transmission rate, gamma is the recovery rate
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt



# SIR solution for plotting and forecasting

def SIR_solution(beta,gamma, S0,I0):
    y0 = [S0, I0, 1-S0-I0]
    sol = solve_ivp(deriv1, [0, num_days], y0, args=(beta, gamma), t_eval=tspan)
    
    return sol.t, sol.y[0]  # return time and S(t)




def SIR_model(lam: np.ndarray, t) -> np.ndarray: 
    # lam is a 2d array of parameters each row is [beta,gamma]
    # return the solution of the SIR model at time t for each parameter set in lam
    # the return is 1-S(t), which represents cumulative infections
    beta = lam[:,0]
    gamma = lam[:,1]
    S0 = lam[:,2]
    I0 = lam[:,3]
    y0 = np.column_stack((S0,I0, 1-S0-I0))  # Initial conditions for S, I, R
    QoI = np.empty(lam.shape[0])
    
    for i in range(lam.shape[0]):
        # Run the differential equation solver
        sol = solve_ivp(deriv1, [0, t], y0[i], args=(beta[i], gamma[i]), t_eval=[t])
        
        # We need sol.y[0][0] to get the actual float value out of it.
        QoI[i] = S0[i] - sol.y[0][0]
        
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
SIR_model7d = lambda lam: SIR_model(lam, t=7)
SIR_model2 = lambda lam: SIR_model(lam, t= 10)
SIR_model14d = lambda lam: SIR_model(lam, t=14)

SIR_model3 = lambda lam: SIR_model(lam, t=15)
SIR_model4 = lambda lam: SIR_model(lam, t=20)
SIR_model21d =lambda lam: SIR_model(lam, t=21)
SIR_model5 = lambda lam: SIR_model(lam, t=25)
SIR_model28d = lambda lam: SIR_model(lam, t=28)
SIR_model6 = lambda lam: SIR_model(lam, t=30)
SIR_model7 = lambda lam: SIR_model(lam, t=35)
SIR_model8 = lambda lam: SIR_model(lam, t=40)
SIR_model42d = lambda lam: SIR_model(lam, t=42)
SIR_model49d = lambda lam: SIR_model(lam, t=49)
SIR_model56d = lambda lam: SIR_model(lam, t=56)
SIR_model9 = lambda lam: SIR_model(lam, t=45)
SIR_model10 = lambda lam: SIR_model(lam, t=50)
SIR_model11 = lambda lam: SIR_model(lam, t=55)
SIR_model12 = lambda lam: SIR_model(lam, t=60)
SIR_model13 = lambda lam: SIR_model(lam, t=65)
SIR_model14 = lambda lam: SIR_model(lam, t=70)
SIR_model15 = lambda lam: SIR_model(lam, t=75)
SIR_model16 = lambda lam: SIR_model(lam, t=80)
SIR_model17 = lambda lam: SIR_model(lam, t=85)
SIR_model18 = lambda lam: SIR_model(lam, t=90)

#forward_models = [SIR_model1, SIR_model2, SIR_model3, SIR_model4, SIR_model5, SIR_model6, SIR_model7, SIR_model8, SIR_model9, SIR_model10, SIR_model11, SIR_model12, SIR_model13, SIR_model14, SIR_model15, 
#                  SIR_model16,SIR_model17,SIR_model18]
#forward_models = [SIR_model1,  SIR_model3,  SIR_model5,  SIR_model7,  SIR_model9,SIR_model11,  SIR_model13,  SIR_model15]
#forward_models = [SIR_model1, SIR_model2, SIR_model10, SIR_model12]
#forward_models = [SIR_model2, SIR_model4, SIR_model6]
#forward_models = [SIR_model2, SIR_model4, SIR_model6, SIR_model8, SIR_model10]
#forward_models = [SIR_model2,SIR_model6,SIR_model10]
#forward_models = [SIR_model1, SIR_model2]
#forward_models = [SIR_model2]
#forward_models =[SIR_model0,SIR_model0_1,SIR_model0_2,SIR_model0_3,SIR_model0_4,SIR_model0_5,SIR_model0_6,SIR_model0_7,SIR_model1, SIR_model2, SIR_model3, SIR_model4, SIR_model5, SIR_model6, SIR_model7, SIR_model8, SIR_model9, SIR_model10, SIR_model11, SIR_model12, SIR_model13, SIR_model14, SIR_model15, SIR_model16]
forward_models = [SIR_model7d,SIR_model21d,SIR_model7,SIR_model49d,SIR_model56d]
#forward_models = [SIR_model2,SIR_model6,SIR_model12,SIR_model14]

forecast_models = [SIR_model2,SIR_model4,SIR_model6,SIR_model8,SIR_model10,SIR_model12,SIR_model14,SIR_model16,SIR_model18]
#forecast_models = [SIR_model1, SIR_model2, SIR_model3, SIR_model4, SIR_model5, SIR_model6, SIR_model7, SIR_model8, SIR_model9, SIR_model10, SIR_model11, SIR_model12, SIR_model13, SIR_model14, SIR_model15, 
#                  SIR_model16,SIR_model17,SIR_model18]
# --- QUICK TEST TO VERIFY IT WORKS ---
if __name__ == "__main__":
    # Test with two sets of [beta, gamma, S0] parameters
    test_lam = np.array([
        [0.6, 0.2,0.59],  # R0 = 3.0 (Fast spread)
        [0.15, 0.1,0.99]  # R0 = 1.5 (Slow spread)
    ])
    
    print("Cumulative infections (1 - S) at Day 7:")
    print(forward_models[0](test_lam))
    
    print("\nCumulative infections (1 - S) at Day 14:")
    print(forward_models[1](test_lam))

