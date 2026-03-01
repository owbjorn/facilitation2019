###### Function to generate panel C of Figure 2

#### Import standard modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#### Import own functions
import facil_functions as ffun

#### Plot or not?
plotflag = False

### Experimental/computational parameters
N     = 250
t0    = 0.0
t_end = 80.0
x0    = 1.0

#### Model parameters
Nspec = 2
hT = [1.0,]
r  = [0.1]*Nspec
m  = [0.15]*Nspec
finv = [0.6]*Nspec
Y  = [0.2]*Nspec
d  = [10.0]*Nspec
Kn = [1.0]*Nspec
Kt = [1.0]*Nspec
kd = [1.0e-2]*Nspec

par_mono = {'r':r[0],'m':m[0],'Kn':Kn[0],'Kt':Kt[0],'Y':Y[0], 'finv':finv[0], 'kd':kd[0],'d':d[0],'hT':hT[0]}
par_co   = {'r':r,'m':m,'Kn':Kn,'Kt':Kt,'Y':Y, 'finv':finv, 'kd':kd,'d':d,'hT':hT}

### Parameter grid of interest
Nsteps = 15
cn_vec = np.linspace(0.0,1.0,Nsteps)
ct_vec = np.linspace(0.0,1.0,Nsteps)
AUC_mono  = np.zeros([Nsteps,Nsteps])
AUC_co    = np.zeros([Nsteps,Nsteps])

#### Compute where survival is enhanced/decreased with respect to chemical concentrations
for idxn, cn in enumerate(cn_vec):
    for idxt, ct in enumerate(ct_vec):
        tsim, xmono = ffun.computeODE(N, t0, t_end, [x0,cn,ct], par_mono)
        tsim, xco   = ffun.computeODE(N, t0, t_end, [x0,x0,cn,ct], par_co)

        ### Biomass is measured as area-under-the-curve (AUC) of the species
        AUC_mono[idxn,idxt] = np.sum(xmono[:,0])*t_end/(N+1)
        AUC_co[idxn,idxt]   = np.sum(xco[:,0])*t_end/(N+1)

#### Save output in R-friendly data frame
dAUC_out = pd.DataFrame(columns=('CN','CT','dAUC'))
dAUC_out.loc[:,'CN']  = np.repeat(cn_vec,Nsteps)
dAUC_out.loc[:,'CT']  = np.tile(cn_vec,Nsteps)
dAUC_out.loc[:,'dAUC'] = np.transpose(Z).flatten()
dAUC_out.to_csv(path_or_buf='./Fig2c_model_NT_common.csv',sep='\t')

#### Plot landscape of species growth in different combinations of
##   nutrients and toxic compounds
if plotflag:
    X, Y = np.meshgrid(cn_vec, ct_vec)
    Z = np.transpose(AUC_co-AUC_mono)
    vmaxp = np.max([abs(np.min(Z)),np.max(Z)])

    fig = plt.figure(dpi=480, figsize = (6,7))
    ax = fig.gca()
    CS = plt.contourf(X, Y, Z, 36, vmin=-vmaxp, vmax=vmaxp, cmap='coolwarm_r') # BrBG

    cbar = fig.colorbar(CS, ticks=[-0.8*vmaxp, 0, 0.8*vmaxp], orientation='horizontal')
    cbar.ax.set_xticklabels(['Negative','Neutral','Positive'])  # horizontal colorbar

    plt.xlabel('Nutrient conc [A.U.]')
    plt.ylabel('Toxin conc [A.U.]')

    plt.show()
    outfile = './Figure2C_NTlandscape.pdf'
    fig.savefig(outfile,format='pdf',frameon='false',bbox_inches='tight')
