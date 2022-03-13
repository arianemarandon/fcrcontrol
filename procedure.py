#####################################################################################################
# Plug-in procedure & Bootstrap procedure
#####################################################################################################

from mixture import sample_tmixture
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from studenttmixture import EMStudentMixture

from misc import get_permutation
from mixture import estimate_resp, estimate_tmixture_resp, sample_mixture

SMALL = 1e-10




# Baseline procedure 
def baseline_procedure(probability_matrix, level):
    misclust_probs = 1 - np.max(probability_matrix, 1)

    return np.nonzero(misclust_probs < level)[0]



# Plug-in procedure 
def plug_in_procedure(probability_matrix, level): 
    """
    In the plug-in procedure, we substitute the true class probabilities with estimates, and select the points for which the corresponding class probability estimate does not exceed some threshold.
    The threshold is chosen in a data-driven manner s.t. the error does not excede <level>, see <get_selection>. 
    
    probability_matrix: n x Q array of class probabilities. 
    level: nominal level. 

    Returns: the points for which we ACCEPT the cluster assignment. 
    """
    misclust_probs = 1 - np.max(probability_matrix, 1)
    
    #then we threshold the misclust_probs in a data-driven manner. 

    n = len(misclust_probs)
    indices = np.argsort(misclust_probs)
    misclust_probs_sort = np.sort(misclust_probs)

    Tsum = np.cumsum(misclust_probs_sort) / np.arange(1, n + 1)

    if np.nonzero(Tsum < level)[0].size:
        n_sel = np.nonzero(Tsum < level)[0][-1] +1 
        selection = indices[:n_sel]
    else: 
        selection = []
    return selection
    


# Bootstrap procedure based on adjustement of the level
def bootstrap_procedure(level, n_classes, x, parameters_estimates, probability_matrix, 
                        n_bootstrap_samples = 1000, parametric = False,
                        covariance_type = 'full',
                        tmixture=False): 
    """
    for gaussian mixtures or t mixtures. 
    """

    selection_levels = np.array([level/np.power(10, k) for k in range(1, 10)])

    estimated_fcr = estimate_fcr(selection_levels, n_classes, x, parameters_estimates, n_bootstrap_samples, parametric, covariance_type = covariance_type, tmixture=tmixture) 


    #get the maximum level for which estimated_fcr < level
    if np.sum(estimated_fcr < level): 
        lower_bound = np.max(selection_levels[estimated_fcr < level])

        #then look for the best level in [lower_bound, lower_bound*10) 

        selection_levels = np.array([lower_bound + k*lower_bound for k in range(10)])
        estimated_fcr = estimate_fcr(selection_levels, n_classes, x, parameters_estimates, n_bootstrap_samples, parametric, covariance_type = covariance_type, tmixture=tmixture) #the new estimation may not be under level the second time

        if np.sum(estimated_fcr < level):
            lower_bound = np.max(selection_levels[estimated_fcr < level])

        else:  
           
            pass  #just keep current lower bound 

    else: 
        lower_bound = 0
        
    
    selection = plug_in_procedure(probability_matrix, lower_bound)

    return selection


def estimate_fcr(selection_levels, n_classes, x, parameters_estimates, 
                n_bootstrap_samples = 1000, parametric = False, 
                covariance_type = 'full',
                tmixture=False):


    if tmixture: 
        mp, means, scales, scales_cholesky, scales_inv, dofs = parameters_estimates #t-mixtures model
    else: 
        mp, means, covs = parameters_estimates #GMM
        

    n = len(x)

    samples =  np.empty((n_bootstrap_samples, len(selection_levels))) 

    for b in range(n_bootstrap_samples):
        if parametric:
            #sample according to estimates 
            x_bs,_ = sample_tmixture(n, (mp, means, scales, dofs))  if tmixture else sample_mixture(n,  mp, means, covs) 
        else: 
            x_bs = x[np.random.randint(0, n, size=n)]

        #compute probabilities under theta_hat for the bootstrap sample

        if tmixture:
            probs = estimate_tmixture_resp(x_bs, mp, means, scales_cholesky, scales_inv, dofs) #t-mixtures model
        else: 
            probs = estimate_resp(x_bs, mp, means, covs) #= bayes probabilities in the case of the parametric bootstrap

        
        y = np.argmax(probs, 1) #partition of the bootstrap sample, based on theta_hat 

        #next, we must compute the partition of the bootstrap sample based on theta_bootstrap, taking into account LS 

        if tmixture:
            clf = EMStudentMixture(n_components= n_classes, n_init=10, max_iter=100, df = 4., fixed_df = True, tol=1e-3) 
        else: 
            clf = GaussianMixture(n_components = n_classes, max_iter = 100, n_init=10, tol=1e-3, covariance_type = covariance_type) 
            

        
        clf.fit(x_bs)
        probs_bs = clf.predict_proba(x_bs)

        
        #let us match the clustering of the bootstrap with the clustering. we do this by comparing the partition obtained with each 
        y_bs = np.argmax(probs_bs, 1) #partition for the bootstrap sample, based on theta_bs 
        
        permumax = get_permutation(y, y_bs)

        #then, update bootstraped probabilities and use argmax
        probs_bs = probs_bs[:, list(permumax)]
        y_bs = np.argmax(probs_bs, 1) #partition for the bootstrap sample, based on theta_bs

        misclass_probabilities = 1 - probs[np.arange(len(x_bs)),y_bs] #see paper 

        
        for k, selection_level in enumerate(selection_levels):

            selection_bs = plug_in_procedure(probs_bs, selection_level)

            emp_fcr = np.mean(misclass_probabilities[selection_bs]) if len(selection_bs) else 0
            samples[b, k] = emp_fcr


    return np.mean(samples, 0)






