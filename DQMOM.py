# -*- coding: utf-8 -*-
"""
Created on Fri May 27 11:55:06 2016

@author: harshavardhan.babu
@email: amail2harsha@gmail.com
"""
from __future__ import division
import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt
import seaborn as sns

def DQMOM(aw,t,Deff):
    
    '''
    Solves the $ A\alpha = d $
    A = [A1 A2]
    d = A3C + beta
    Ref: D.L. Marchisio, R.O. Fox / Aerosol Science 36 (2005)
    params:
    aw: flattened array of weights and abscissa
    Deff: Diffusion in phase space
    t : time 
    '''
    
    N = int(np.size(aw)/2); # Number of modes in the distribution

    aw = np.reshape(aw,(2,N)); # Retrieve the weights and abscissas
    
    weights  = np.squeeze(aw[0,:]); # Weights of the distribution
    abscissa = np.squeeze(aw[1,:]); # Abscissa of the distribution
    
    A1 = np.zeros((2*N,N));
    A2 = np.zeros((2*N,N));
    
    for row in range(2*N):
        if(row>=2):
            A1[row,:] = [-(row-1)*ab**(row) for ab in abscissa];
            A2[row,:] = [(row)*ab**(row-1) for ab in abscissa];
    A1[0,:] = np.ones(N);
    A2[1,:] = np.ones(N);
    
    A = np.concatenate((A1,A2),1)
    
    A3 = np.zeros((2*N,N));
    for row in range(2*N):
        if(row>=3):
            index = row-1;
            A3[row,:] = [2*(2*index-1)*(index-1)*ab**(index-1) for ab in abscissa];
    A3[2,:] = np.ones(N)*2;
    
    C = np.zeros(N); # Modify this array if there is spatial diffusion
    beta = np.zeros(2*N);
    
    D = np.ones(N)*Deff;
    Source = weights*D + C; # Add any other sources due to aggregation,breakage
                            # Nucleation

    d = np.dot(A3,Source);
    
    rcond = 1./np.linalg.cond(A);
    if(rcond < 1e-15):
        print("rcond of A is %e"%(rcond))
        
    
    alpha = np.linalg.solve(A,d);
    alpha = np.reshape(alpha,(2,N));

    # modifying the source terms for abscissas as the original equation gives
    # source terms for weighted abscissas
        
    alpha[1,:] = (alpha[1,:] - abscissa*alpha[0,:])/weights;
    alpha = alpha.flatten();
    return alpha;
  
def HomogenousDispersion():
    '''
    Solves homogenous dispersion in phase space as given in pg52 of reference
    '''
    N = 2;    
    Deff = 1.152;
    
    w  = np.ones(N)*0.5;
    ab = [1. + np.sqrt(2.*Deff*0.1),1. - np.sqrt(2.*Deff*0.1)];
    
    aw = np.concatenate((w,ab));
    
    t = np.linspace(0,2e3,2e3);
    faw = odeint(	DQMOM,aw,t,args=(Deff,));

    plt.plot(t,faw);
    plt.plot(t,1+np.sqrt(2.*Deff*t),'o',markevery=100)
    plt.plot(t,1-np.sqrt(2.*Deff*t),'o',markevery=100)
    plt.xlabel(r"Time $(\mathrm{secs})$")
    plt.ylabel(r"Abscissa $(\mathrm{-})$")
    plt.title(r"Comparison between the analytical solution (continuous line) and DQMOM (open symbols).")
    plt.show();
    
if __name__ == "__main__":
    HomogenousDispersion()