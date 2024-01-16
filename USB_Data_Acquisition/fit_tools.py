# This function calculates uncertainties perr from the covariance matrix pcov
# Axel Mellinger, 2022-04-03

import numpy as np

def err_corr_matrix(pcov):
    """
    Returns uncertainties of fit parameters and the correlation matrix.

    Parameters
    ----------
    pcov : array
        Second output of curve_fit()

    Returns
    -------
    perr  : array (1-dim)
        Contains the errors of the fit parameters
    pcorr : array
        Correlation coefficients

    """
    perr = np.sqrt(np.diag(pcov))
    X, Y = np.meshgrid(perr, perr)
    pcorr = pcov/X/Y
    return perr, pcorr