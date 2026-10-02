# SIR forward model with four parameters: beta, gamma, and initial susceptible proportion, initial infected proportion. The initial recovered proportion is computed as 1 - S0 - I0. This allows for uncertainty in the initial conditions, which is important for modeling real-world scenarios where the exact start date of an outbreak may not be known. The model uses the `solve_ivp` function from SciPy to solve the system of differential equations that describe the SIR dynamics.
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


######### SIR master model for efficient running forward models ################
def SIR_master_model(lam: np.ndarray, t_eval: list) -> np.ndarray: 

    """
    Runs the SIR model ONCE per parameter set for all requested time points.
    Returns an array of shape (N_samples, N_times) containing cumulative infections (1 - S(t)).
    """
    t = np.atleast_1d(t_eval)
    
    beta = lam[:,0]
    gamma = lam[:,1]
    S0 = lam[:,2]
    I0 = lam[:,3]
    y0 = np.column_stack((S0, I0, 1 - S0 - I0))
    
    QoI = np.empty((lam.shape[0], len(t)))
    t_span = [0, np.max(t)]
    
    for i in range(lam.shape[0]):
        # The solver runs ONCE per sample, recording state at all t_eval points
        sol = solve_ivp(deriv1, t_span, y0[i], args=(beta[i], gamma[i]), t_eval=t)
        QoI[i, :] = S0[i] - sol.y[0]
        
    return QoI


