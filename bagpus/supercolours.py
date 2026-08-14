import numpy as np
from astropy.io import fits


def read_eigensystem(evecfile, filterfile, verbose=True):
    '''
    This routine reads in the eigenbasis of super colours in .fits format and
    spits out the wavelength, eigenvectors, variance  and mean spectrum.
    '''
    fits_path = str(evecfile)
    hdul = fits.open(fits_path) # Read in .fits file
    data = hdul[1].data # Saving data as numpy array
    hdul.close() # Once we're done with the .fits file, close

    # Save data in dictionary of np arrays
    # There are cleverer ways to do this, but this is read once and is not a big file
    ebasis = {}    
    ebasis['wave'] = data['WAVE_REST_SUPER'][0]
    ebasis['spec'] = data['EVECS'][0]
    ebasis['mean'] = data['MEANARR'][0]
    #ebasis['var'] = data['VARIANCE'][0] # this isn't needed
    ebasis['ind_wave']= data['IND_WAVE'][0]
    ebasis['minz'] = data['MINZ'][0]
    ebasis['maxz'] = data['MAXZ'][0]
    ebasis['dz'] = data['DZ'][0]
    #if no_filter_names != True:
    #    filternames = data['FILTERNAMES_SUPER'][0]
    #else:
    #    filternames=' '

    # Print some info about the eigensystem
    if verbose:
        print(str(len(ebasis['spec']))+' eigenvectors extracted')
        print('Minimum redshift of the eigenbasis: '+str(ebasis['minz']))
        print('Maximum redshift of the eigenbasis: '+str(ebasis['maxz']))

    # Read in filter effective wavelengths into array
    f = open(filterfile).readlines()
    ll_eff = np.zeros(len(f))
    filt_list = []
    for i in range(0, len(f)):
        ll_eff[i] = float(f[i].split()[1])
        filt_list.append(str(f[i].split()[0]))

    ebasis['ll_eff']=ll_eff
    ebasis['filt_list']=filt_list

    return ebasis

def vwsc_fillflux(flux,z,ebasis):
    '''
    Place f_nu_obs into super-sampled array and then convert into f_lambda_rest.
    '''

    minz = ebasis['minz']
    maxz = ebasis['maxz']
    ll_obs = ebasis['ll_eff'] # effective wavelength of filter is our observed wavelength
    ind_wave = ebasis['ind_wave']
    dz = ebasis['dz']

    c_in_AA = 2.99792e18        

    nredshift = int((maxz-minz)/dz) + 1
    zbin = np.linspace(minz, maxz, nredshift)
    nz = len(zbin)
    n_band = len(ll_obs)

    ff = c_in_AA * flux / (ll_obs**2) # f_nu_obs to f_lambda_obs
    ff = ff * (1+z) # f_lambda_obs to f_lambda_rest
    
    # Find into which redshift bin the galaxy lands
    tmp = abs(z - zbin)
    ind_zz = tmp.argmin()

    # Find into which band bin to put flux into
    ind = np.arange(0,n_band,1)*nz+ind_zz

    fluxarr = np.zeros(nz*n_band)
    fluxarr[ind] = ff
  
    fluxarr  = fluxarr[ind_wave]

    return fluxarr

def normgappy(data, error, ebasis, cov=False, \
                reconstruct=False, verbose=False):
    """
    Performs robust PCA projection, including normalization estimation. 
    Parameters
    ----------
    data : ndarray
        1D spectrum or 2D specta with 'float' type.
    error : ndarray
        1D or 2D corresponding 1-sigma error array. Zeros indicate masked data.
    espec : ndarray
        2D array of eigenspectra, possibly truncated in dimension.
    mean : ndarray
        1D mean spectrum of the eigenspectra.
    cov: bool, optional
        Return covariance matrix.
        Default is ''False''.
    reconstruct : bool, optional
        Fill in missing values with PCA estimation.
        Default is ''False''.
    verbose : bool, optional
        Enable for status and debug messages.
        Default is ''False''
    Returns
    -------
    pcs : ndrray
        1D or 2D array of Principal Components with 'float' type.
    norm: float or ndarray
        Normalization estimates.
    data : ndrray
        If reconstruct enabled, 1D or 2D reconstructed spectra.
    ccov : ndarray
        If cov enabled, 2D or 3D covariance matrices.
    Credits
    -------
    Based on original IDL code by Gerard Lemson and Vivienne Wild.
    Converted to python by John Weaver. 
    Available as a standalone package here: https://github.com/astroweaver/pygappy
    """
    
    espec = ebasis['spec']
    mean = ebasis['mean']

    # Sanity checks
    if (np.size(data) == 0) | (np.size(error) == 0) | (np.size(espec) == 0) | (
            np.size(mean) == 0):
        print('[pca_normgappy] ERROR: incorrect input lengths')
        return None

    tmp = np.shape(espec)  # number of eigenvectors
    if np.size(tmp) == 2:
        nrecon = tmp[0]
    else:
        nrecon = 1
    nbin = np.shape(espec)[-1]  # number of data points
    tmp = np.shape(data)  # number of observations to project
    if np.size(tmp) == 2:
        ngal = tmp[0]
    else:
        ngal = 1

    
    # Dimension mismatch check
    if np.shape(data)[-1] != nbin:
        print(
            '[pca_normgappy] ERROR: "data" must have the same dimension as eigenvectors'
        )
        return None
    if np.shape(error)[-1] != nbin:
        print(
            '[pca_normgappy] ERROR: "error" must have the same dimension as eigenvectors'
        )
        return None
    if np.shape(mean)[0] != nbin:
        print(
            '[pca_normgappy] ERROR: "mean" must have the same dimension as eigenvectors'
        )
        return None

    # Project each galaxy in turn
    pcs = np.zeros((ngal, nrecon), float)
    norm = np.zeros(ngal, float)
    if cov is not None:
        ccov = np.zeros((ngal, nrecon, nrecon))

    if ngal == 1:
        data = data[np.newaxis, :]
        error = error[np.newaxis, :]

    for j in np.arange(0, ngal):

        if verbose:
            print('[pca_normgappy] STATUS: processing spectrum ')

        # Calculate weighting array from 1-sig error array
        # ! if all bins have error=0 continue to next spectrum
        weight = np.zeros(nbin)
        ind = error[j, :].nonzero()[0]
        if np.size(ind) != 0:
            try:
                weight[ind] = 1. / (error[j, :][ind]**2)
            except:
                if verbose:
                    print(
                        '[pca_normgappy] ERROR: error array problem in spectrum (setting pcs=0)'
                    )
                continue

        if np.isnan(weight).any() == True:
            if verbose:
                print(
                    '[pca_normgappy] ERROR: error array problem in spectrum (setting pcs=0)'
                )
            continue

        data_j = data[j, :]

        # Solve partial chi^2/partial N = 0
        Fpr = np.sum(weight * data_j * mean)  # eq 4 [2]
        Mpr = np.sum(weight * mean * mean)  # eq 5 [2]
        E = np.sum((weight * mean) * espec, axis=1)  # eq 6 [2]

        
        # Calculate the weighted eigenvectors, multiplied by the eigenvectors (eq. 4-5 [1])

        if nrecon > 1:
            # CONSERVED MEMORY NOT IMPLEMETED
            espec_big = np.repeat(espec[:, np.newaxis, :], nrecon, axis=1)
            M = np.sum(weight * np.transpose(espec_big, (1, 0, 2)) * espec_big, 2)

            # Calculate the weighted data array, multiplied by the eigenvectors (eq. 4-5 [1])
            F = np.dot((data_j * weight), espec.T)

            # Calculate new M matrix, this time accounting for the unknown normalization (eq. 11 [2])
            E_big = np.repeat(E[np.newaxis, :], nrecon, axis=0)
            F_big = np.repeat(F[:, np.newaxis], nrecon, axis=1)
            Mnew = Fpr * M - E_big * F_big

            # Calculate the new F matrix, accounting for unknown normalization
            Fnew = Mpr * F - Fpr * E
            
            # Solve for Principle Component Amplitudes (eq. 5 [1])
            try:
                Minv = np.linalg.inv(Mnew)
            except:
                if verbose:
                    print(
                        '[pca_normgappy] STATUS: problem with matrix inversion (setting pcs=0)'
                    )

                continue

            pcs[j, :] = np.squeeze(np.sum(Fnew * Minv, 1))
            norm[j] = Fpr / (Mpr + np.sum(pcs[j, :] * E))
            
            # Calculate covariance matrix (eq. 6 [1])
            if cov is True:
                M_gappy = np.dot((espec * (weight * norm[j]**2)), espec.T)
                ccov[j, :, :] = np.linalg.inv(M_gappy)

        else:  # if only one eigenvector
            M = np.sum(weight * espec * espec)
            F = np.sum(weight * data_j * espec)
            Mnew = M * Fpr - E * F
            Fnew = Mpr * F - E * Fpr
            pcs[j, 0] = Fnew / Mnew
            norm[j] = Fpr / (Mpr + pcs[j, 0] * E)
            if cov is True:
                ccov[j, 0, 0] = np.sum((1. / weight) * espec * espec)

        # If reconstruction of data array required,
        #   fill in regions with weight = 0 with PCA reconstruction
        if reconstruct is True:
            bad_pix = np.where(weight == 0.)
            count = np.size(bad_pix)
            if count == 0:
                continue

            rreconstruct = np.sum((pcs[j, :] * espec[:, bad_pix].T).T, 0)
            rreconstruct += mean[bad_pix]
            data[j, bad_pix] = reconstruct

    if ngal == 1:
        pcs = pcs[0]
        data = data[0]
        norm = norm[0]
        if cov:
            ccov = ccov[0]

    # Report to user
    # if verbose:
    #     print("[pca_normgappy] STATUS: Results...")
    #     for i, pc in enumerate(pcs):
    #         print(f"               PCA{i+1}: {pc:2.5f}")
    #     print(f"               Norm: {norm:2.5f}")

    # Return
    if reconstruct is True:
        if cov is True:
            return pcs, norm, data, ccov
        else:
            return pcs, norm, data

    elif cov is True:
        return pcs, norm, ccov
    else:
        return pcs, norm

