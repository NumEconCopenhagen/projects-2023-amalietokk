from scipy import optimize

def solve_ss(s, g, n, delta, alpha, theta):
    """
    Args:
        s, g, n, delta, alpha, theta - parameters

    Returns:
        result (RootResults): the solution represented as a RootResults object.

    """ 
    
    # a. Objective function, depends on k (endogenous).
    f = lambda k: k**alpha
    obj_k_star = lambda k_star: (s*(1-theta)*f(k_star) -(delta+g+n)*k_star)

    #. b. call root finder to find kss.
    result = optimize.root_scalar(obj_k_star,bracket=[0.1,100],method='brentq')
    s = result.root
    
    return print('the steady state for k is',s)



from types import SimpleNamespace
import time
import numpy as np
from scipy import optimize
from types import SimpleNamespace
import time
import numpy as np
from scipy import optimize
"""
class GreenSolowclass():

    def __init__(self,do_print=True):
        # create the model 

        if do_print: print('initializing the model:')

        self.par = SimpleNamespace()
        self.sim = SimpleNamespace()

        if do_print: print('calling .setup()')
        self.setup()

        if do_print: print('calling .allocate()')
        self.allocate()
    
    def setup(self):
        # baseline parameters 
        # a. household
        par.sigma = 2.0 # CRRA coefficient
        par.beta = 1/1.40 # discount factor

        # b. firms
        par.alpha = 0.30 # capital weight
        par.theta = 0.05 # fraction of output spent on pollution abatement
        par.delta = 0.50 # depreciation rate

        # d. environmental
        par.z = 0.03 # rate of decline in emission coefficient
        par.epsilon=1.2
        
        # e. misc
        par.K_lag_ini = 1.0 # initial capital stock
        par.B_lag_ini = 1.0 # initial index of labor efficiency
        par.L_lag_ini = 1.0 # initial 
        par.Omega_lag_ini = 1.0 # initial index of 
        par.simT = 50 # length of simulation

        # f. growth rates
        par.n = 0.01 # labor force growth rate
        par.g = 0.02 # growth rate of labor efficiency
        
        # g. savings rate
        par.s = 0.3 # savings rate

    def allocate(self):
        # allocate arrays for simulation 
        
        par = self.par
        sim = self.sim

        
        # a. list of variables
        economy = ['Q','R','Y','K','L','B','E','dotK', 'dotL', 'dotE', 'dotOmega']
        environment = ['Omega']
        
        # b. allocate
        allvarnames = economy + environment
        for varname in allvarnames:
            sim.__dict__[varname] = np.nan*np.ones(par.simT)
                  
    def simulate(self,do_print=True):
        # simulate model 

        t0 = time.time()

        par = self.par
        sim = self.sim
        
        # a. initial values
        sim.K_lag[0] = par.K_lag_ini
        sim.B_lag[0] = par.B_lag_ini
        sim.L_lag[0] = par.L_lag_ini
        sim.Omega_lag[0] = par.Omega_lag_ini

        # b. iterate
        for t in range(par.simT):
            
            # i. simulate before k
            simulate_before_k(par,sim,t)

            if t == par.simT-1: continue          


            # ii. find optimal s
            f = lambda k: k**alpha
            obj = lambda k_star: targetfunc(k,par,sim,t=t)
            result = optimize.root_scalar(obj,bracket=[0.1,100],method='bisect')
            s = result.root

            # iii. simulate after k
            simulate_after_k(par,sim,t,k)

        if do_print: print(f'simulation done in {time.time()-t0:.2f} secs')

def targetfunc(k,par,sim,t):
    # target function for finding s with bisection 

    # a. simulate forward
    simulate_after_k(par,sim,t,s)
    simulate_before_k(par,sim,t+1) # next period

    # c. target func
    target = (par.s*(1-par.theta)*(sim.k[t]**alpha)-(par.delta+par.g+par.n)*(sim.k[t]**alpha))

    return target

def simulate_before_k(par,sim,t):
    # simulate forward 

    if t > 0:
        sim.K_lag[t] = sim.K[t-1]
        sim.B_lag[t] = sim.B[t-1]
        sim.L_lag[t] = sim.L[t-1]
        sim.Omega_lag[t] = sim.Omega[t-1]
    
        # i. production
        sim.Y[t] = (1-par.theta) * (sim.K_lag[t]**(par.alpha)(sim.B_lag[t]*sim.L_lag[t])**(1-par.alpha)
        # ii. Capital
        sim.dotK[t] = par.s*sim.Y[t]-par.delta*sim.K[t]
        # iii. Labour  
        sim.dotL[t] = par.n*sim.L[t]
        sim.L[t]=sim.L[0]*exp(par.n*t)
        sim.dotB[t] = par.g*sim.B[t]
        sim.B[t]=sim.B[0]*exp(par.g*t)                            
        sim.E[t]=sim.Omega[t]*sim.Q[t](1-\theta)**par.epsilon
        sim.dotOmega[t]=-par.z*sim.Omega[t] 
        sim.Omega[t]=sim.Omega[0]*exp(par.z*t)                            
                

def simulate_after_k(par,sim,t,k):
    # simulate forward THIS IS FROM THE OLG MODEL

    # a. consumption of young
    sim.C1[t] = (1-par.tau_w)*sim.w[t]*(1.0-s)

    # b. end-of-period stocks
    I = sim.Y[t] - sim.C1[t] - sim.C2[t] - sim.G[t]
    sim.K[t] = (1-par.delta)*sim.K_lag[t] + I
    
    """
