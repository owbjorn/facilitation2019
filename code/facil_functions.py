#### Functions to solve the ODE system described
#### in Piccardi, Vessman, Mitri (2019) PNAS
#### https://doi.org/10.1073/pnas.1906172116
#### ... and plot the solution

#### Import modules
import numpy as np
import scipy
import scipy.integrate as integr


#### Functions for models
def f_monod(s,K,r):
    """
    Monod function f = rs/(s+K), aka Holling type II
    INPUT
      s    substrate concentration
      K    half-saturation of growth effect
      r    max growth rate
    Output
      f    rs/(s+K)
    """
    ######

    f_num = np.multiply(s,r)
    f_den = np.add(s,K)
    ftemp = np.divide(f_num,f_den) # i.e. element-wise, f = rs/(s+K)
    return( ftemp )

def f_Hill(s,K,r,h):
    """
    Hill function f = rs^h/(s^h+K^h), aka Holling type III
    INPUT
      s    substrate concentration
      K    half-saturation of growth effect
      r    max growth rate
      h    Hill coefficient
    Output
      f    rs^h/(s^h+K^h)
    """

    spowh = np.power(s,h)
    Kpowh = np.power(K,h)
    f_num = np.multiply(spowh,r)   # Numerator:   r*s^h
    f_den = np.add(spowh,Kpowh)    # Denominator: s^h+K^h
    ftemp = np.divide(f_num,f_den)
    return( ftemp )

def f_mech(t,x, args):
    """
    Functional response of the ODE system described in Supplementary S3
    in Piccardi, Vessman, Mitri (2019) PNAS.
    Input:
      t      time scale, used by ODE solver scipy.integrate
      x      state vector with
               species abundance in x[:-2]
               nutrient concentration in x[-2]
               toxic concentration in x[-1]
      args   dictionary of model parameters, further described below.
    Output:
      dX     Change in state vector, used by ODE solver scipy.integrate
    """
    rmax = args["r"]    # Max growth effect of nutrient
    mmax = args["m"]    # Max mortality of toxin
    Kn   = args["Kn"]   # Half-max effect of nutrients
    Kt   = args["Kt"]   # Half-max effect of toxins
    Yinv = args["Y"]    # Nutrient Yield [say gram of biomass per mol of nutrient]
    finv = args["finv"] # Average population investment [0.0, 1.0] into growth
    kdeg = args["kd"]   # Passive uptake of toxins
    d    = args["d"]    # Toxin degradation rate [say mol of toxin per gram of biomass]
    hT   = args["hT"]   # Hill coefficient for toxin equation

    x_spec = x[:-2]
    c_nutr = x[-2]
    c_tox  = x[-1]

    Ns = len(x_spec)
    # Change in x due to growth and mortality
    dS_grow = np.multiply(f_monod(c_nutr,Kn,rmax),x_spec)   # Growth    rc/(c+Kn)*x per species
    # dS_mort = np.multiply(f_monod(c_tox,Kt,mmax),x_spec)  # Mortality mc/(c+Kt)*x
    dS_mort = np.multiply(f_Hill(c_tox,Kt,mmax,hT),x_spec)  # Mortality mc^h/(c^h+Kt^h)*x

    dN_nutr = -1.0*np.dot(Yinv, dS_grow)
    dT_invest = np.multiply(np.multiply(finv,d),f_monod(c_nutr,Kn,rmax))
    dT_percap = np.dot(np.add(dT_invest,kdeg),x_spec)
    dT_tox    = -1.0*np.multiply(dT_percap, c_tox)

    # Assemble state vector
    invest_gr = np.subtract([1.0]*Ns,finv)
    dS = np.subtract(np.multiply(invest_gr,dS_grow), dS_mort)
    dX = np.append(np.append(dS,dN_nutr),dT_tox)
    return dX

def jac_df(s,K,r):
    """
    Derivative df/ds of Monod function f(s) = rs/(s+K)
      df_Monod(s)/ds = rK/(s+K)^2
    INPUT
      s    substrate concentration
      K    half-saturation of growth effect
      r    max growth rate
    Output
      df(s)/ds = rK/(s+K)^2
    """
    df_numer = np.multiply(r,K)
    df_denom = np.power(np.add(s,K),2.0)
    return(np.divide(df_numer,df_denom))

def jac_df_Hill(s,K,r,h):
    """
    Derivative df/ds of Hill function f(s) = rs^h/(s^h+K^h)
      df_Hill(s)/ds = rK^h*hs^(h-1)/(s^h+K^h)^2
    INPUT
      s    substrate concentration
      K    half-saturation of growth effect
      r    max growth rate
      h    Hill coefficient
    Output
      df(s)/ds = rK^h*hs^(h-1)/(s^h+K^h)^2
    """
    spowh = np.power(s,h)
    Kpowh = np.power(K,h)

    df_numer = h*np.multiply(r,Kpowh)*np.power(s,h-1.0)
    df_denom = np.power(np.add(spowh,Kpowh),2.0)
    return np.divide(df_numer,df_denom)

def jac_mech(t, x, args):
    """
    Jacobian of the ODE system described in f_mech()
    """
    rmax = args["r"]    # Max growth effect of nutrient
    mmax = args["m"]    # Max mortality of toxin
    Kn   = args["Kn"]   # Half-max effect of nutrients
    Kt   = args["Kt"]   # Half-max effect of toxins
    Yinv = args["Y"]    # Nutrient Yield [say gram of biomass per mol of nutrient]
    finv = args["finv"] # Average population investment [0.0, 1.0] into growth
    kdeg = args["kd"]   # Passive uptake of toxins
    d    = args["d"]    # Toxin degradation rate [say mol of toxin per gram of biomass]
    hT   = args["hT"]   # Hill coefficient for toxin equation
    x_spec = x[:-2]
    c_nutr = x[-2]
    c_tox  = x[-1]

    #### Prelims
    f_growth = f_monod(c_nutr,Kn,rmax)
    # Yinv     = np.divide(1.0,Y)
    Ns = len(x_spec)
    invest_gr = np.subtract([1.0]*Ns,finv)

    #### Compute partial derivatives wrt species abundance x and nutrient conc n
    # Effect on species dynamics: per-capita functional response
    # dfdS =  np.subtract(f_growth,f_monod(c_tox,Kt,mmax))
    invest_gr = np.subtract([1.0]*Ns,finv)
    dfdS =  np.subtract(np.multiply(invest_gr,f_growth),f_Hill(c_tox,Kt,mmax,hT))
    dfdN =  np.multiply(np.multiply(invest_gr,jac_df(c_nutr,Kn,rmax)),x_spec)        # Wrt nutrients: rKS/(c+K)^2
    dfdT = -np.multiply(jac_df_Hill(c_tox,Kt,mmax,hT),x_spec) # Wrt toxins: mhc^(h-1)K^hS/(c^h+K^h)^2

    # Effect on nutrient dynamics
    dNdS = -1.0*np.multiply(Yinv,f_growth) # Nutrient conc wrt species growth
    dNdN = -1.0*np.dot(Yinv,dfdN) # Nutrient conc wrt nutrients
    dNdT = 0.0

    # Effect on toxin
    dT_invest = np.multiply(np.multiply(finv,d),f_monod(c_nutr,Kn,rmax))
    dT_percap = np.dot(np.add(dT_invest,kdeg),x_spec)
    dT_tox    = -1.0*np.multiply(dT_percap, c_tox)

    invest_de = np.multiply(finv,d)
    dTdS = -c_tox*np.add(np.multiply(invest_de,f_growth),kdeg)
    dTdN = -c_tox*np.dot(np.multiply(invest_de,jac_df(c_nutr,Kn,rmax)),x_spec)
    dTdT = -1.0*np.dot(np.add(np.multiply(invest_de,f_growth),kdeg),x_spec)

    # Assemble Jacobian from partial derivatives
    df = np.transpose(np.vstack((np.diag(dfdS),dfdN,dfdT)))
    dN = np.append(np.append(dNdS,dNdN),dNdT)
    dT = np.append(np.append(dTdS,dTdN),dTdT)
    J_out = np.vstack((df,dN,dT))
    return( J_out )

def computeODE(N, t0, T, y0, pars_dict):
    """
    Function to solve the ODE defined by f_mech()
    with the corresponding Jacobian jac_mech().
    """
    Nspec = len(y0)
    x     = integr.ode(f_mech,jac_mech).set_integrator('dopri5')
    xout  = scipy.zeros([N+1,Nspec])
    xout[0,:] = y0

    x.set_initial_value(y0,t0).set_f_params(pars_dict).set_jac_params(pars_dict)
    dt = (T-t0)/N
    t  = scipy.linspace(t0,T,N+1)
    idx_temp = 0

    while x.successful() and idx_temp<N:
        idx_temp += 1
        x_temp = x.integrate(x.t+dt)
        xout[idx_temp] = x_temp

    return( t, xout )

def f_dAUC_minimize(Nspec,pars):
    """
    Function to optimize the number of species to simulate
    the benefit of co-culturing species. We assume that the benefit of
    co-culturing should not be due to just increasing the biomass, so
    we scale the biomass yield and degradation coefficients by the
    number of species. The yield and degradation in a co-culture with scaled
    parameters is equivalent to that of the corresponding mono-culture.
    # INPUT
    - Nspec   Number of cocultured species
    - p       [r, m, Y, f, d, k, K_N, K_T] array of parameters, where
    -  -r     maximum growth rate with respect to nutrient conc
    -  -m     maximum death rate with respect to toxin conc
    -  -Y     inverse biomass yield, i.e. how many units of nutrient to produce one cell
    -  -d     toxin degradation rate with respect to species growth
    -  -cN    nutrient conc modifier, how nutritious are AAs compared to MWF?
    - conc    [C_T, C_N] data array of time points in first col, CFUs in second
    # OUTPUT
    - errout  Combined total fitting error to data
    """

    pars_dict = pars['pars_dict']

    ### Time points for integration
    N     = pars['N']
    t0    = pars['t0']
    t_end = pars['t_end']

    #### Initial conditions
    x0 = pars['x0']
    AUC_mono = pars['AUC_mono']

    #### Simulate coculture
    pars_co = pars_dict.copy()
    pars_co['Y'] = np.multiply(pars_dict['Y'],Nspec)
    pars_co['d'] = np.multiply(pars_dict['d'],Nspec)
    tsimco, xsimco = computeODE(N, t0, t_end, x0, pars_co)
    dAUC = -np.sum(xsimco[:,0])/AUC_mono

    return( dAUC )
