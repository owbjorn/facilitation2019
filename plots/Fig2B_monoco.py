###### Function to generate panel B of Figure 2

#### Import standard modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#### Import own functions
import facil_functions as ffun

#### Plot or not?
plotflag = False

# Left graph: Monoculture
N     = 500
t0    = 0.0
t_end = 100.0  # Days

Nspec = 2
x0s = [1.0]*Nspec # Initial biomass
c0 = [1.0,1.0]  # Conc of nutrient, toxin
x0 = np.hstack((x0s,c0))
r  = [0.1]*Nspec
m  = [0.2]*Nspec
finv = [0.1]*Nspec
Y  = [0.2]*Nspec
d  = [15.0]*Nspec
Kn = [1.5]*Nspec
Kt = [1.0]*Nspec
kd = [0.0e-3]*Nspec 

hT = [1.0,]
## Monoculture
par_dict = {'r':r[0],'m':m[0],'Kn':Kn[0],'Kt':Kt[0],'Y':Y[0],'finv':finv[0],'kd':kd[0],'d':d[0],'hT':hT}
tplot, xplotm = ffun.computeODE(N, t0, t_end, np.hstack((x0[0],c0)), par_dict)

## Co-oculture
par_dict = {'r':r,'m':m,'Kn':Kn,'Kt':Kt,'Y':Y, 'finv':finv, 'kd':kd, 'd':d,'hT':hT}
tplot, xplotcn = ffun.computeODE(N, t0, t_end, x0, par_dict)

## For paper: print to CSV and process figure in CRAN R
xmono_struct = {'time': tplot, 'Xm': xplotm[:,0], 'CNm': xplotm[:,1], 'CTm': xplotm[:,2]}
xco_struct   = {'time': tplot, 'Xc1': xplotcn[:,0], 'Xc2': xplotcn[:,1],
                'CNc': xplotcn[:,2], 'CTc': xplotcn[:,3]}

xmono_df = pd.DataFrame(xmono_struct)
xco_df   = pd.DataFrame(xco_struct)

## Print to CSV
xmono_df.to_csv(path_or_buf='./Fig2b_model_mono.csv',sep='\t')
xco_df.to_csv(path_or_buf='./Fig2b_model_co.csv',sep='\t')


#### Plot quartet
if plotflag:
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey='row')
    legendtup = (r'$S_1$',r'$N$',r'$T$')

    axtmp = ax[0]
    x_spec = xplotm[:,:-2]
    x_conc = xplotm[:,-2:]
    axtmp.plot(tplot,x_spec,lw=2,color='k')
    axtmp.plot(tplot,x_conc[:,0],ls='dashed',lw=2,color='k')
    axtmp.plot(tplot,x_conc[:,1],ls='dotted',lw=2,color='k')
    axtmp.legend((r'$S_1$',r'$N$',r'$T$'),loc='upper right')
    axtmp.set(xlabel="Time [a.u.]",ylabel="Abundance/conc. [a.u.]",title=r"$S_1$ mono-culture")

    axtmp = ax[1]
    x_spec = xplotcn[:,:-2]
    x_conc = xplotcn[:,-2:]
    axtmp.plot(tplot,x_spec[:,0],lw=2,color='k')
    axtmp.plot(tplot,x_conc[:,0],ls='dashed',lw=2,color='k')
    axtmp.plot(tplot,x_conc[:,1],ls='dotted',lw=2,color='k')
    axtmp.legend((r'$S_1$',r'$N$',r'$T$'),loc='upper right')
    axtmp.set(xlabel="Time [a.u.]",title=r"$S_1$ co-culture w/ $S_2$")

    plt.show()

    outfile = '../figs/181120_2s1n1t_Co-half-popsize.pdf'
    fig.savefig(outfile,format='pdf',frameon='false',bbox_inches='tight')
