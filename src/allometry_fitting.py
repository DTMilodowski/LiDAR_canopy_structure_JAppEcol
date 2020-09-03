import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib import rcParams

# Set up some basiic parameters for the plots
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['arial']
rcParams['font.size'] = 8
rcParams['legend.numpoints'] = 1
axis_size = rcParams['font.size']+2


"""
# linear regression with confidence intervals and prediction intervals
"""
def linear_regression(x_,y_,conf=0.95):
    mask = np.all((np.isfinite(x_),np.isfinite(y_)),axis=0)
    x = x_[mask]
    y = y_[mask]

    # regression to find power law exponents D = a.H^b
    m, c, r, p, serr = stats.linregress(x,y)
    x_i = np.arange(x.min(),x.max(),(x.max()-x.min())/1000.)
    y_i = m*x_i + c
    PI,PI_u,PI_l = calculate_prediction_interval(x_i,x,y,m,c,conf)
    CI,CI_u,CI_l = calculate_confidence_interval(x_i,x,y,m,c,conf)

    model_y = m*x + c
    error = y-model_y
    MSE = np.mean(error**2)

    return m, c, r**2, p, x_i, y_i, CI_u, CI_l, PI_u, PI_l
"""
# log-log regression with confidence intervals
"""
def log_log_linear_regression(x,y,conf=0.95):
    mask = np.all((np.isfinite(x),np.isfinite(y)),axis=0)
    logx = np.log(x[mask])
    logy = np.log(y[mask])

    # regression to find power law exponents D = a.H^b
    b, loga, r, p, serr = stats.linregress(logx,logy)
    logx_i = np.arange(logx.min(),logx.max(),(logx.max()-logx.min())/1000.)
    PI,PI_upper,PI_lower = calculate_prediction_interval(logx_i,logx,logy,b,loga,conf)

    x_i = np.exp(logx_i)
    PI_u = np.exp(PI_upper)
    PI_l = np.exp(PI_lower)

    model_logy = b*logx + loga
    error = logy-model_logy
    MSE = np.mean(error**2)
    CF = np.exp(MSE/2) # Correction factor due to fitting regression in log-space (Baskerville, 1972)
    a = np.exp(loga)
    return a, b, CF, r**2, p, x_i, PI_u, PI_l

"""
# ANALYTICAL CONFIDENCE AND PREDICTION INTERVALS

# Calculate confidence intervals analytically (assumes normal distribution)
# x_i = x location at which to calculate the confidence interval
# x_obs = observed x values used to fit model
# y_obs = corresponding y values
# m = gradient
# c = constant
# conf = confidence interval
# returns dy - the confidence interval
"""
def calculate_confidence_interval(x_i,x_obs,y_obs,m,c,conf):

    alpha = 1.-conf

    n = x_obs.size
    y_mod = m*x_obs+c
    se =  np.sqrt(np.sum((y_mod-y_obs)**2/(n-2)))
    x_mean = x_obs.mean()

    # Quantile of Student's t distribution for p=1-alpha/2
    q=stats.t.ppf(1.-alpha/2.,n-2)
    dy = q*se*np.sqrt(1/float(n)+((x_i-x_mean)**2)/np.sum((x_obs-x_mean)**2))
    y_exp=m*x_i+c
    upper = y_exp+abs(dy)
    lower = y_exp-abs(dy)
    return dy,upper,lower
"""
# Calculate prediction intervals analytically (assumes normal distribution)
# x_i = x location at which to calculate the prediction interval
# x_obs = observed x values used to fit model
# y_obs = corresponding y values
# m = gradient
# c = constant
# conf = confidence interval
# returns dy - the prediction interval
"""
def calculate_prediction_interval(x_i,x_obs,y_obs,m,c,conf):

    alpha = 1.-conf

    n = x_obs.size
    y_mod = m*x_obs+c
    se =  np.sqrt(np.sum((y_mod-y_obs)**2/(n-2)))
    x_mean = x_obs.mean()

    # Quantile of Student's t distribution for p=1-alpha/2
    q=stats.t.ppf(1.-alpha/2.,n-2)
    dy = q*se*np.sqrt(1+1/float(n)+((x_i-x_mean)**2)/np.sum((x_obs-x_mean)**2))
    y_exp=m*x_i+c
    upper = y_exp+abs(dy)
    lower = y_exp-abs(dy)
    return dy,upper,lower

"""
# Calculate a prediction based on a linear regression model
# As above, but this time randomly sampling from prediction interval
# m = regression slope
# c = regression interval
"""
def random_sample_from_regression_model_prediction_interval(x_i,x_obs,y_obs,m,c,array=False):
    mask = np.all((np.isfinite(x_obs),np.isfinite(y_obs)),axis=0)
    x_obs=x_obs[mask]
    y_obs=y_obs[mask]
    n = x_obs.size
    y_mod = m*x_obs+c
    se =  np.sqrt(np.sum((y_mod-y_obs)**2/(n-2)))
    y_exp = x_i*m+c # expected value of y from model
    x_mean = x_obs.mean()
    # randomly draw quantile from t distribution (n-2 degrees of freedom for linear regression)
    if array:
        q = np.random.standard_t(n-2,size=x_i.size)
    else:
        q = np.random.standard_t(n-2)
    dy = q*se*np.sqrt(1+1/float(n)+((x_i-x_mean)**2)/np.sum((x_obs-x_mean)**2))
    y_i = y_exp+dy

    return y_i

"""
# as above, but using log-log space (i.e. power law functions)
# a = scalar
# b = exponent
"""
def random_sample_from_powerlaw_prediction_interval(x_i,x_obs,y_obs,a,b,array=False):
    if array:
        logy_i = random_sample_from_regression_model_prediction_interval(np.log(x_i),np.log(x_obs),np.log(y_obs),b,np.log(a),array=True)
    else:
        logy_i = random_sample_from_regression_model_prediction_interval(np.log(x_i),np.log(x_obs),np.log(y_obs),b,np.log(a))
    y_i = np.exp(logy_i)
    return y_i

"""
#=================================
# BOOTSTRAP TOOLS

# Calculate prediction intervals through bootstrapping and resampling from residuals.
# The bootstrap model accounts for parameter uncertainty
# The residual resampling accounts for uncertainty in the residual - i.e. effects not
# accounted for by the regression model
# Inputs:
# - x_i = x location(s) at which to calculate the prediction interval
#   This should be either a numpy array or single value. nD arrays will be
#   converted to 1D arrays
# - x_obs = the observed x values used to fit model
# - y_obs = corresponding y values
# - conf = confidence interval, as fraction
# - niter = number of iterations over which to bootstrap
# - n_i = number of locations x_i (default is
# Returns:
# - ll and ul = the upper and lower bounds of the confidence interval
"""
def calculate_prediction_interval_bootstrap_resampling_residuals(x_i,x_obs,y_obs,conf,niter):

    from matplotlib import pyplot as plt
    # some fiddles to account for likely possible data types for x_i
    n=0
    if np.isscalar(x_i):
        n=1
    else:
        try:
            n=x_i.size # deal with numpy arrays
            if x_i.ndim > 1: # linearize multidimensional arrays
                x_i=x_i.reshape(n)
        except TypeError:
            print("Sorry, not a valid type for this function")

    y_i = np.zeros((n,niter))*np.nan
    # Bootstrapping
    for ii in range(0,niter):
        # resample observations (with replacement)
        ix = np.random.choice(x_obs.size, size=n,replace=True)
        x_boot = np.take(x_obs,ix)
        y_boot = np.take(y_obs,ix)

        # regression model
        m, c, r, p, serr = stats.linregress(x_boot,y_boot)

        # randomly sample from residuals with replacement
        res = np.random.choice((y_boot-(m*x_boot + c)),size = n,replace=True)

        # estimate y based on model and randomly sampled residuals
        y_i[:,ii] = m*x_i + c + res

    # confidence intervals simply derived from the distribution of y
    ll=np.percentile(y_i,100*(1-conf)/2.,axis=1)
    ul=np.percentile(y_i,100*(conf+(1-conf)/2.),axis=1)

    return ll,ul

# equivalent to above but for log-log space prediction
def calculate_powerlaw_prediction_interval_bootstrap_resampling_residuals(x_i,x_obs,y_obs,
                                                                    conf=.9,niter=1000):
    log_ll,log_ul = calculate_prediction_interval_bootstrap_resampling_residuals(np.log(x_i),np.log(x_obs),np.log(y_obs),conf,niter)
    return np.exp(log_ll),np.exp(log_ul)


"""
plot_allometric_relationship
Fit and plot an allometric relationship, labelling plot with r-squared value and
best fit equation.
Currently uses power law fit by default, other options include 'linear' for
simple linear regression.
Prediction uncertainty calculated using a bootstrap method. Analytical method
could also be used based on the functions above, although not coded as an option for this particular function
"""
def plot_allometric_relationship(x,y,figure_name,model='powerlaw',niter=10000,annotate_str='',xlabel_str='',ylabel_str=''):
    if model=='powerlaw':
        a, b, CF, r_sq, p, x_, PI_u, PI_l = log_log_linear_regression(x,y,conf=0.90)

        PI_25,PI_75 = calculate_powerlaw_prediction_interval_bootstrap_resampling_residuals(H_,H_BAAD,D_BAAD,conf=0.5,niter=niter)
        PI_05,PI_95 = calculate_powerlaw_prediction_interval_bootstrap_resampling_residuals(H_,H_BAAD,D_BAAD,conf=0.9,niter=niter)

        y_ = CF*a*x_**b

    elif model=='linear':
        a, b, CF, r_sq, p, x_, PI_u, PI_l = linear_regression(x,y,conf=0.90)

        PI_25,PI_75 = calculate_prediction_interval_bootstrap_resampling_residuals(H_,H_BAAD,D_BAAD,conf=0.5,niter=niter)
        PI_05,PI_95 = calculate_prediction_interval_bootstrap_resampling_residuals(H_,H_BAAD,D_BAAD,conf=0.9,niter=niter)

        y_ = CF*a*x_**b
    else:
        print('specified model not available at the moment')
        return 1

    fig,ax = plt.subplots(facecolor='White',figsize=[4,4])
    ax = plt.subplot2grid((1,3),(0,0))
    ax.set_ylabel(ylabel_str)
    ax.set_xlabel(xlabel_str)
    ax.annotate(annotate_str, xy=(0.08,0.95), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='top', fontsize=10)

    ax.fill_between(x_,PI_05,PI_95,color='0.95')
    ax.fill_between(x_,PI_25,PI_75,color='0.85')
    ax.plot(x,y,'.',color='#1A2BCE')#,alpha=0.2)
    ax.plot(x_,y_,'-',color='black')

    eq = '$y=%.2fx^{%.2f}$\n$r^2=%.2f$' % (CF*a, b,r_sq)
    ax.annotate(eq, xy=(0.08,0.85), xycoords='axes fraction',backgroundcolor='none',horizontalalignment='left', verticalalignment='top', fontsize=10)

    ax.set_xlim(xmin=0)
    ax.set_ylim(ymin=0)
    plt.tight_layout()
    plt.savefig(figure_name)
    plt.show()
    return 0
