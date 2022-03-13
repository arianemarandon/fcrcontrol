import numpy as np 
from scipy import linalg
from scipy.special import logsumexp, gammaln
from scipy.spatial.distance import mahalanobis
from scipy.stats import multivariate_t


def sample_mixture(n_samples, mix_prop, means, covariances):

    classes = np.argmax(np.random.multinomial(1, mix_prop, size = n_samples), 1)
    
    dimensionSize = 1 if np.isscalar(means[0]) else len(means[0])
    samples = np.empty((n_samples, dimensionSize))
    for i in range(n_samples):
        c = classes[i]
        m = means[c]
        cov = covariances[c]
        samples[i] = np.random.multivariate_normal(m, cov)
            
    return samples, classes 

def sample_tmixture(n_samples, mix_prop, locations, scales, dofs):

    classes = np.argmax(np.random.multinomial(1, mix_prop, size = n_samples), 1)

    dimensionSize = 1 if np.isscalar(locations[0]) else len(locations[0])
    samples = np.empty((n_samples, dimensionSize))
    for i in range(n_samples):
        c = classes[i]
        samples[i] = multivariate_t.rvs(loc = locations[c], shape = scales[c], df = dofs[c])
    return samples, classes 



#below are excerpts from sklearn's code source for GaussianMixture()

def compute_precision_cholesky(covariances):
    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar."
    )

    n_components, n_features, _ = covariances.shape
    precisions_chol = np.empty((n_components, n_features, n_features))
    for k, covariance in enumerate(covariances):
        try:
            cov_chol = linalg.cholesky(covariance, lower=True)
        except linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_chol[k] = linalg.solve_triangular(cov_chol, np.eye(n_features), lower=True).T

    return precisions_chol

def compute_log_det_cholesky(matrix_chol, n_features):
    """Compute the log-det of the cholesky decomposition of matrices.
    Parameters
    ----------
    Returns
    -------
    log_det_precision_chol : array-like of shape (n_components,)
        The determinant of the precision matrix for each component.
    """
    n_components, _, _ = matrix_chol.shape
    log_det_chol = np.sum( 
        np.log(matrix_chol.reshape(n_components, -1)[:, :: n_features + 1]), 1
        )
    return log_det_chol


def estimate_log_gaussian_prob(X, means, covariances):
    """Estimate the log Gaussian probability.
    Parameters
    ----------
    Returns
    -------
    log_prob : array, shape (n_samples, n_components)
    """
    n_samples, n_features = X.shape
    n_components, _ = means.shape

    precisions_chol = compute_precision_cholesky(covariances)
    # det(precision_chol) is half of det(precision)
    log_det = compute_log_det_cholesky(precisions_chol, n_features)

    log_prob = np.empty((n_samples, n_components))
    for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
        y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
        log_prob[:, k] = np.sum(np.square(y), axis=1)

    return -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det

def estimate_weighted_log_prob(X, weights, means, covariances):
    """Estimate the weighted log-probabilities, log P(X | Z) + log weights.
    Parameters
    ----------
    Returns
    -------
    weighted_log_prob : array, shape (n_samples, n_component)
    """
    return estimate_log_gaussian_prob(X, means, covariances) + np.log(weights)

def estimate_log_prob_resp(X, weights, means, covariances):
        """Estimate log probabilities and responsibilities for each sample.
        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.
        Parameters
        ----------
        Returns
        -------
        log_prob_norm : array, shape (n_samples,)
            log p(X)
        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
        """
        weighted_log_prob = estimate_weighted_log_prob(X, weights, means, covariances)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under="ignore"):
            # ignore underflow
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_resp

def estimate_resp(X, weights, means, covariances):
    log_prob_norm, log_resp = estimate_log_prob_resp(X, weights, means, covariances)
    return np.exp(log_resp)




#below are excerpts from https://github.com/jlparkI/mix_T/

def estimate_tmixture_resp(X, weights, locations, scales_cholesky, scales_inv_cholesky, dofs):
    weighted_log_prob = get_weighted_loglik(X, weights, locations, scales_cholesky, scales_inv_cholesky, dofs)
    log_prob_norm = logsumexp(weighted_log_prob, axis=1)
    with np.errstate(under="ignore"):
        log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
    return np.exp(log_resp)


def get_weighted_loglik(X, weights, locations, scales_cholesky, scales_inv_cholesky, dofs):
    """
    scales_inv : Q x d x d 
    scales_cholesky: d x d x Q to match the code of https://github.com/jlparkI/mix_T/ 
    """

    #scales_inv_cholesky = get_scale_inv_cholesky(scales_cholesky)
    dimensionSize = X.shape[1]; n_components = len(weights)
    sq_maha_dist = sq_maha_distance(X, locations, scales_inv_cholesky)
    
    loglik = get_loglikelihood(X, sq_maha_dist, dofs, scales_cholesky, weights)
    return loglik + np.log(weights)[np.newaxis,:] # n x Q + 1 x Q


def get_loglikelihood(X, sq_maha_dist, df_, scale_cholesky_, mix_weights_):
    sq_maha_dist = 1 + sq_maha_dist / df_[np.newaxis,:]
        
    #THe rest of this is just the calculations for log probability of X for the
    #student's t distributions described by the input parameters broken up
    #into three convenient chunks that we sum on the last line.
    sq_maha_dist = -0.5*(df_[np.newaxis,:] + X.shape[1]) * np.log(sq_maha_dist)
        
    const_term = gammaln(0.5*(df_ + X.shape[1])) - gammaln(0.5*df_)
    const_term = const_term - 0.5*X.shape[1]*(np.log(df_) + np.log(np.pi))
        
    scale_logdet = [np.sum(np.log(np.diag(scale_cholesky_[:,:,i])))
                        for i in range(len(mix_weights_))]
    scale_logdet = np.asarray(scale_logdet)
    return -scale_logdet[np.newaxis,:] + const_term[np.newaxis,:] + sq_maha_dist #nxQ


def get_scale_inv_cholesky(scale_cholesky):
    scale_inv_cholesky = np.empty(scale_cholesky.shape)
    for i in range(scale_cholesky.shape[2]):
        scale_inv_cholesky[:,:,i] = linalg.solve_triangular(scale_cholesky[:,:,i],
                    np.eye(scale_cholesky.shape[0]), lower=True).T
    return scale_inv_cholesky

#def squaredMahaDistance(X, locations, scales_inv):
    #n = X.shape[0]
    #n_components = len(locations)
    #sq_maha_dist = np.empty((n, n_components))
    #for i in range(n):
        #for k in range(n_components):
            #sq_maha_dist[i,k] = mahalanobis(X[i], locations[k], scales_inv[k])
    #return sq_maha_dist

def sq_maha_distance(X, loc_, scale_inv_cholesky_):
        sq_maha_dist = np.empty((X.shape[0], scale_inv_cholesky_.shape[2]))
        for i in range(sq_maha_dist.shape[1]):
            y = np.dot(X, scale_inv_cholesky_[:,:,i])
            y = y - np.dot(loc_[i,:], scale_inv_cholesky_[:,:,i])[np.newaxis,:]
            sq_maha_dist[:,i] = np.sum(y**2, axis=1)
        return sq_maha_dist