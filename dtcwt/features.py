import dtcwt
import dtcwt.utils
import numpy as np
import dtcwt.sampling
from matplotlib.pyplot import *
# The following are required for image resizing
import scipy
from scipy import misc
from scipy.interpolate import griddata
from scipy.ndimage import map_coordinates
from skimage.transform import ProjectiveTransform, warp, matrix_transform, resize, AffineTransform

"""Performs the mark II SLP operation on an image.
This relies on the weak linearity of DTCWT coefficient
phase with feature shift. rjw57's dtcwt toolbox is
required to use this class."""

class slp2:
    """Class for the SLP2 operation on a DTCWT pyramid.
    Includes a multiscale keypoint detector based on 
    this feature space.
    By S.C. Forshaw, August 2014.
    Modified March 2017 (overhaul + new functions)"""

    def __init__(self, initimg, nlevels=5, full=True, firstlevel=0, plots=False, verbose=False):
        # Will use these default settings unless told otherwise
        self.nlevels = nlevels
        self.full = full
        self.plots = plots
        self.image = initimg
        self.verbose = verbose
        if full:
            self.firstlevel = 0
        else:
            self.firstlevel = firstlevel-1

        # Copied/pasted from MATLAB, should regenerate these
        # natively at some point.
        # row index: transform level, column index: subband
        self.rotratio = (np.multiply(-1,
                        [[3.7962, 4.2812, 3.8327, 3.8327, 4.2812, 3.7962],
                        [4.5437, 5.0473, 4.5336, 4.5336, 5.0473, 4.5437],
                        [4.7176, 5.4489, 4.6981, 4.6981, 5.4489, 4.7176],
                        [4.7651, 5.5655, 4.7443, 4.7443, 5.5655, 4.7651],
                        [4.7692, 5.5909, 4.7507, 4.7507, 5.5909, 4.7692 ]]))

    def histnd(self, x, bins, wt=None):
        """Form an N-dimensional histogram of the list of N-vectors in x.
        x is an N by M matrix, where the number of vectors is M.
        Bilinear interpolation over an N-dimensional hypercube is
        used to smooth the histogram data and give optimal representation
        of the data distribution.  This is particularly important
        with coarse bin sizes, which are needed when N gets larger, to
        avoid the size of h becoming excessive.

        The bins of the histogram are specified by the N by 3 matrix:
        bins = [x1_min  x1_max  x1_delta; x2... ; x3... ; ... ; xN...]
        The vector [x1_min:x1_delta:x1_max] specifies the bin centres.
        Vectors of the exact bin centres are output using the cell array
        binvecs.

        wt is an optional M-vector which specifies the weight given to each
        sample of x when added to the histogram.  The default wt = ones(1,M).

        MATLAB implementation by Nick Kingsbury, 2005
        Python port by S. C. Forshaw, 2013"""

        # Initialisations and input size checks

        if len(np.shape(x)) == 1:  # i.e., if it is an 1xn vector
            # reshape it so its shape becomes (N,1) not (N, )
            x = x.reshape(1, x.shape[0])

        if len(np.shape(bins)) == 1:
            # Likewise for the bins vector
            bins = bins.reshape(1, bins.shape[0])

        # Assign the sizes to variables which control loops
        N = x.shape[0]
        M = x.shape[1]
        Nb = bins.shape[0]
        Mb = bins.shape[1]

        # Checking input sizes
        if Nb != N:
            raise ValueError('HISTND: bins vectors do not match x vectors in size')

        if Mb != 3:
            raise ValueError('HISTND: bin params must be [x_min x_max x_delta]')

        # No. of bins in each dimension (must be integer values)
        nbins = np.int32(np.round((bins[:, 1] - bins[:, 0])/bins[:, 2]))

        if np.any(nbins < 1):
            raise ValueError('HISTND: no. of bins not all positive')

        # unit M-vector. Not sure if this has to be an int, a double or what.
        uM = np.ones([1, M], dtype='float')
        if wt is None:
            wt = uM.copy()

        # Finished initialisations
        # Subtract xn_min from x so all bins start at zero and 
        # scale them by 1/xn_delta
        xr = (x - (bins[:, 0] * uM.T).T) / (bins[:, 2] * uM.T).T

        # Limit xr to the range 0 to nbins
        xr = np.maximum((np.minimum(xr, (nbins * uM.T).T)), 0)

        # Find integer xr, and set up weights for each component
        # using fractional parts of xr
        # Do not allow xi to exceed (nbins-1).
        xi = np.minimum(np.floor(xr), ((nbins - 1) * uM.T).T)

        w1 = xr - xi  # Fractional parts of each xr.
        w0 = 1 - w1  # 1 - fractional parts.

        # Initialise
        # The output histogram as a column vector
        h = np.zeros([np.prod(nbins + 1), 1])
        # Add 1 to accommodate upper and lower boundaries of each set of bins
        sh = nbins + 1
        binvecs = list(range(0, N))

        # Loop to define exact bin centres in binvecs
        # (which is a list in Python rather than a cell array in MATLAB)
        for n in binvecs:
            binvecs[n] = np.arange(0, nbins[n]) * bins[n, 2] + bins[n, 0]

        # Step sizes for t along each dimension
        p = np.append(np.array([1]), np.array(np.cumprod(sh[:])))

        # No. of vertices on the N-dim hypercube of near-neighbours
        ns = np.round(2**N)
        # Array of histogram increments for each input vector
        hi = np.zeros([ns, M])
        # Initialise array for zero fractional parts
        hi[0, :] = wt

        # Locations to be incremented by each hi vector
        t = np.ones([ns, M], dtype='int')

        # Loop to calculate the histogram increments
        # hi and the location vectors t. Each t vector specifies the
        # ns locations that will be incremented by each hi vector.
        s0 = np.array([1], dtype='int')
        for n in range(0, N):
            s1 = s0 + s0[-1]
            uS = np.ones([s0.shape[0], 1])
            hi[s1-1, :] = hi[s0-1, :] * (uS * w1[n, :])
            hi[s0-1, :] = hi[s0-1, :] * (uS * w0[n, :])
            t[s0-1, :] = t[s0-1, :] + ((p[n] * uS.T).T * xi[n, :])
            t[s1-1, :] = t[s0-1, :] + p[n]
            s0 = np.hstack((s0, s1))

        h = np.squeeze(h)  # Remove the singleton dimension

        # Increment the histogram with each of the M input vectors. 
        # This loop is critical for speed in the MATLAB version.
        for m in np.arange(0, M, 1):
            h[t[:, m]-1] = (h[t[:, m]-1] + hi[:, m])

        # Reshape h to N dimensions
        if N > 1:
            # A lot of axis swapping and hacking is going on here...
            h = np.swapaxes(np.reshape(h, (nbins+1).T, order='F'), 1, 0)

        return h  # binvecs is not returned   

    def init(self):
        """This will calculate the appropriate sampling locations.
        Rather than guessing what the DTCWT will do with 
        the image (which could be any size), it will apply the DTCWT 
        and find out for itself. This is costly now, but if working on
        multiple frames of the same size, it saves generating the meshes
        every frame. It also means it should work with the undecimated DTCWT."""

        # Put a check in here somewhere to see if DTCWT object already exists
        # First, we perform the DTCWT (Tfm is of the class Pyramid;
        # we access the high-pass parts using Tfm.highpasses).
        transform = dtcwt.Transform2d('near_sym_b_bp', 'qshift_b_bp')     
        Tfm = transform.forward(self.image, self.nlevels)

        # Provide the space which is used to generate the relative
        # sampling locations. Not to be confused with the 'sb'
        # used elsewhere to loop over subbands.
        sb = np.array(list(range(1,13)))

        # We use negative angles (and ADD the offset) 
        # because we are working in uv coordinates
        ru = 0.5*np.cos(sb*(np.pi/(-6)) + np.pi/12) + 0.5
        rv = 0.5*np.sin(sb*(np.pi/(-6)) + np.pi/12) + 0.5

        U = list(range(0,self.nlevels))
        V = list(range(0,self.nlevels))
        sampleLocs = list(range(0,self.nlevels))

        for level in range(self.firstlevel,self.nlevels):
            # Generate the grid to which we will add the relative sampling locations
            U[level], V[level] = (np.meshgrid(
                                  np.arange(Tfm.highpasses[level].shape[1]),
                                  np.arange(Tfm.highpasses[level].shape[0])))
            
            sampleLocs[level] = (np.zeros([Tfm.highpasses[level].shape[0],
                                 Tfm.highpasses[level].shape[1], 4, 6]))

            for sb in np.arange(0,6):
                sampleLocs[level][..., sb] = (np.dstack([U[level]+ru[sb],
                                                          V[level]+rv[sb],
                                                          U[level]+ru[sb+6],
                                                          V[level]+rv[sb+6]]))
                # Coordinates are in the format [centre-u, centre-v,
                # [upper-u/upper-v/lower-u/lower-v], subband]

        if self.verbose:
            print("SLP2 sampling locations calculated.\n")

        return sampleLocs

    def transform(self, image, sampleLocs):
        """ Method to actually perform the SLP mk 2 
            without recalculating interpolation locations"""
    
        # Specify the angles at which the DTCWT is selective to image features
        sbangle = (np.array([ np.pi/12, np.pi/4, 5*np.pi/12,
                             7*np.pi/12, 3*np.pi/4, 11*np.pi/12]))
        gamma = list(range(0,self.nlevels))


        transform = dtcwt.Transform2d('near_sym_b_bp', 'qshift_b_bp')
        Tfm = transform.forward(image, self.nlevels)

        # Check to make sure the SLP2 sampling locations match the specified image.
        if Tfm.highpasses[self.nlevels-1].shape[0:1] != sampleLocs[self.nlevels-1].shape[0:1]:
            raise ValueError('Precomputed SLP2 sampling locations do not match the dimensions of \n \
            the specified image. Please use a correctly-sized image or recompute sampling locations.')
        
        # Begin loop over scales
        for level in range(self.firstlevel,self.nlevels):
            
            # The following must be initialised as complex
            # or the imaginary part will be lost!
            samples = (np.zeros([Tfm.highpasses[level].shape[0]*2,
                                 Tfm.highpasses[level].shape[1],6], dtype='complex'))

            # This loop performs the actual interpolation
            for sb in range(0,6):
                samples[:,:,sb] = (np.squeeze(dtcwt.sampling.sample_highpass(
                  Tfm.highpasses[level], np.vstack((sampleLocs[level][..., 0, sb], 
                                                    sampleLocs[level][..., 2, sb])),
                                         np.vstack((sampleLocs[level][..., 1, sb], 
                                                    sampleLocs[level][..., 3, sb])), 
                                         method='bilinear', sbs=[sb])))

            # Conjugate multiply the samples in the upper half of the sampling circle with their counterparts in the lower half
            W = samples[0:Tfm.highpasses[level].shape[0],:,:]*np.conj(samples[Tfm.highpasses[level].shape[0]:,:,:])
            # This is where the phase rotation takes place
            # Uncomment for the intermediate alpha value
            # alpha = np.abs(W)**(1 - (1/rotratio[level,:])) * W ** (1 / rotratio[level,:])
            gamma[level] = np.abs(W)**(1 - (1/self.rotratio[level,:])) * W ** (1 / self.rotratio[level,:])*np.exp(1j*sbangle)
            gamma[level][np.isnan(gamma[level])] = 0

        if self.verbose:
            print("SLP2 coefficients computed.")

        return gamma # Return our final "cell array" (list of numpy arrays).
    
    def global_warp(self, slp2pyramid, Tfm, resample=True):
        # Initialise the returned list
        warped = [None,] * len(slp2pyramid)
        
        # Set up a transformation matrix with just the upper 2x2
        affine = np.eye(3)
        affine[0:2,0:2] = Tfm[0:2,0:2]
        
        # Begin loop over scales
        for level in range(0, len(slp2pyramid)):
            if slp2pyramid[level] is None:
                continue
            
            # Create grid of sampling locations
            U, V = np.meshgrid(np.arange(2**level, slp2pyramid[level].shape[1]*(2**level)*2, 2**(level+1)), np.arange(2**level, slp2pyramid[level].shape[0]*(2**level)*2, 2**(level+1)))
            
            # Create transformable position vectors from the complex SLP2 coefficients
            delta = (np.array([np.real(slp2pyramid[level]).flatten(), 
                               -np.imag(slp2pyramid[level]).flatten(), 
                               np.ones_like(np.real(slp2pyramid[level]).flatten())]))
            
            # Warp the coefficients themselves using only the upper 2x2 portion
            # of the matrix. this is done because the translation parameters
            # only affect the sampling locations
            warpedDelta = np.dot(affine, delta)
            warpedField = (warpedDelta[0,:] - 1j*warpedDelta[1,:]).reshape(slp2pyramid[level].shape)
            #sc = np.abs(warpedField).max()
            #sc = 1
            #warpedField = warpedField/sc
            
            if resample:
                # Warp the sampling locations using the transformation full matrix
                warpedLocs = np.dot(Tfm, np.array([U.flatten(), V.flatten(), np.ones_like(U.flatten())]))
            
                for sb in range(0,slp2pyramid[level].shape[-1]):
                    #warpedField[:,:,sb] = (warp(np.real(warpedField[:,:,sb]), inverse_map=ProjectiveTransform(Tfm)) 
                    #        - 1j*warp(np.imag(warpedField[:,:,sb]), inverse_map=ProjectiveTransform(Tfm)))
                    
                    # This works, but is probably slower than warp() would be if I could fix it
                    warpedField[:,:,sb] = griddata(warpedLocs[0:2,:].T, warpedField[:,:,sb].flatten(), (U.flatten(), V.flatten()), fill_value=0).reshape(slp2pyramid[level].shape[0], slp2pyramid[level].shape[1])
            
            # Place in the returned list
            warped[level] = warpedField
            
        return warped
      
    # SLP2 histogram generator
    def histgen(self, gamma, nbins=24, full=True, best=False):
    
        # Initialise the returned list
        finalghists = [None,] * len(gamma)
    
        # Begin loop over scales
        for level in range(self.firstlevel, len(gamma)):
            # Sorting and using only the 2 largest coefficients not implemented yet.
            udim = np.int(gamma[level].shape[0])
            vdim = np.int(gamma[level].shape[1])
            grid = np.zeros([vdim, udim, 6], dtype='complex')
            rows, cols = np.meshgrid(np.arange(0,vdim,1), np.arange(0,udim,1))

            # Binning array consists of:
            # Row 1: Row indices of SLP2 coefficients repeated 6 times
            # Row 2: Column indices of SLP2 coefficients repeated 6 times
            # Row 3: Phase angles of SLP2 coefficients 
            #        present at the locations in rows 1 and 2
            binningarray = np.vstack([\
            np.tile(rows.flatten('F'), 6), np.tile(cols.flatten('F'), 6),
            np.mod(np.angle(gamma[level].flatten('F')), np.pi)])

            # Present histnd() with the binning array, 
            # and weight by magnitude of SLP2 coefficients
            ghist = (self.histnd(binningarray, 
                                 np.array([[0, vdim-1, 1], 
                                           [0, udim-1, 1], 
                                           [0, np.pi, np.pi/nbins]]),
                                 np.sqrt(np.abs(gamma[level].flatten('F')))))
            # Make histogram wrap by placing all entries
            # from the final orientation bin in the zeroth bin
            ghist[:,:,0] = ghist[:,:,0] + ghist[:,:,-1]
            # Only retain the first 24 orientation bins
            finalghists[level] = ghist[:,:,0:-1] + 10**-8

        if self.verbose:
            print("SLP2 histograms generated successfully.")

        return finalghists

    def histvis(self, w, bs):
        # Make picture of positive SLP2 block histogram contents
        # Intended to emulate the behaviour of the common HOG visualiser

        # Get the number of orientation bins
        nbins = w.shape[-1]

        # Construct a "glyph" for each orientation
        bim1 = np.zeros([bs, bs])
        bim1[:,np.arange(np.round(bs/2), np.round(bs/2)+1).astype('int32')] = 1
        bim = np.zeros([bim1.shape[0], bim1.shape[1], nbins])
        bim[:,:,0] = bim1

        for i in range(1,nbins):
            bim[:,:,i] = scipy.misc.imrotate(bim1, 90 - i*(-180/nbins))
            
        # Make pictures of positive values bs adding up to weighted glyphs
        s = w.shape
        w[w<0] = 0
        visimg = np.zeros([bs*s[0], bs*s[1]])

        for i in range(1,s[0]+1):
            iis = np.arange((i-1)*bs+1, i*bs+1)-1
            for j in range(1,s[1]+1):
                jjs = np.arange((j-1)*bs+1, j*bs+1)-1
                for k in range(0,nbins):
                    visimg[np.ix_(iis,jjs)] = (visimg[np.ix_(iis,jjs)] 
                                               + bim[:,:,k]*w[i-1,j-1,k])
                    
        return visimg
        
    # Complex sorting by magnitude, N-dimensional case
    def complex_sort_nd(self, complexArray, axis):
        index = np.argsort(np.abs(complexArray), axis)
        return complexArray[list(np.mgrid[[slice(x) for x in complexArray.shape]][:-1])+[index]]
    
    # Complex sorting by magnitude 3-dimensional case
    def complex_sort_3d(self, complexArray, axis):
        ii = np.indices(complexArray.shape)
        si = np.argsort(np.abs(complexArray), axis=2)
        # There are still memory-saving improvements to be made
        return complexArray[ii[0], ii[1], si]
    
    def keypoints(self, featuremaps, method='gale', edge_suppression=None, threshold=5):
        """Bare working version of the SLP2 histogram keypoint detector.
        Now includes subpixel refinement.
        """
        # TODO: It would be nice to have the option of
        # 'gale' (cross product) or 'forshaw'
        # (fourier autocorrelation) methods to achieve this
        # Declare lists to contain the outputs of the
        # detector for each transform level
        rng = [None] * len(featuremaps)
        self.energymaps = [None] * len(featuremaps)
        peakmaps = [None] * len(featuremaps)
        kps = []
        
        if method == 'forshaw':
        # Main loop over levels
            for k in range(0, self.nlevels):
                # Check to see whether histograms are present at all levels
                if featuremaps[k] is None:
                    continue
                
                if featuremaps[k].dtype =='complex128':
                    raise TypeError('This keypoint detector requires fully real SLP2 histograms.')

                nbins = featuremaps[k].shape[-1]

                # Edge-suppressing cosine window specified here
                weights = np.cos(np.arange(-1*np.pi/2, np.pi/2, np.pi/nbins))**edge_suppression

                # FIXME: Better scale invariance is achieved
                # by having the histograms squared (assuming 
                # they were square-rooted earlier in the process
                featuremaps[k] = featuremaps[k]**2
                
                # FFT-based weighted normalised autocorrelation FIXME: scale invariance is in doubt at the moment
                self.energymaps[k] = np.sqrt(np.real(np.sum(np.fft.ifft(np.fft.fft(featuremaps[k]*(2**-(k-1)), nbins, 2)*\
                np.conj(np.fft.fft(featuremaps[k]*(2**-(k-1)), nbins, 2)), nbins, 2)*weights, axis=2, dtype='complex')\
                /np.sqrt(np.sum(featuremaps[k]**2, axis=2)/nbins)))
            
        else:
            # Default to Gale keypoint detector
            gamma_tilde = list(rng)
            # Main loop over levels
            for k in range(0, self.nlevels):
                # Check to see whether histograms are present at all levels
                if featuremaps[k] is None:
                    continue

                gamma_tilde[k] = (np.stack([np.real(featuremaps[k]), 
                                            np.imag(featuremaps[k]), 
                                            # np.zeros_like(np.real(featuremaps[k])) 
                                            # Zeros not needed because cross product will 
                                            # assume 2d vectors are coplanar in 3D
                                            ], axis=-1))
                
                mag_cp = np.zeros([featuremaps[k].shape[0], featuremaps[k].shape[1], 6, 6])
                max_abs = np.zeros([featuremaps[k].shape[0], featuremaps[k].shape[1], 6, 6])

                for d1 in range(0, 6):
                    for d2 in range(0, 6):
                        # mag_cp[:,:,d1,d2] = np.sqrt(np.sum(np.cross(gamma_tilde[k][:,:,d1,:], gamma_tilde[k][:,:,d2,:])**2, axis=-1))
                        mag_cp[:,:,d1,d2] = np.abs(np.cross(gamma_tilde[k][:,:,d1,:], gamma_tilde[k][:,:,d2,:]))
                        max_abs[:,:,d1,d2] = np.max(np.abs(featuremaps[k][:,:,[d1,d2]]), axis=-1)  + 10**-8

                self.energymaps[k] = np.sum(np.sum(mag_cp / max_abs, axis=-1), axis=-1)
        
        # Now method-independent steps
        for k in range(0, self.nlevels):
            if self.energymaps[k] is None:
                continue
            
            # Initialise the logical map
            peakmaps[k] = np.zeros([self.energymaps[k].shape[0],self.energymaps[k].shape[1]], dtype='int')
            
            # This is the searchable area of the keypoint energy map
            y = self.energymaps[k][1:-2:1,1:-2:1]
            
            # Test each location in the searchable area to see if it is larger than its 8 neighbours
            peakmaps[k][1:-2:1,1:-2:1] = \
            np.all(np.array([y > self.energymaps[k][0:-3:1,0:-3:1],\
            y > self.energymaps[k][1:-2:1,0:-3:1],\
            y > self.energymaps[k][2:-1:1,0:-3:1],\
            y > self.energymaps[k][0:-3:1,1:-2:1],\
            y > self.energymaps[k][2:-1:1,1:-2:1],\
            y > self.energymaps[k][0:-3:1,2:-1:1],\
            y > self.energymaps[k][2:-1:1,2:-1:1],\
            y > self.energymaps[k][1:-2:1,2:-1:1]]), axis=0)
            
            # Get the locations of non-zeros (i.e., peaks)
            uv = np.nonzero(peakmaps[k])
            values = self.energymaps[k][uv]
            
            # The locations need to be rearranged so that they are in [u,v] order
            locs = np.vstack([uv[1], uv[0]]).T

            # Put the keypoint locations in a list, along with their scale and energy values.
            # Note that 1 must be added to the locations before scaling them to image size due
            # to zero indexing. 1 must then be subtracted again.
            # This takes place after subpixel refinement.
            if self.verbose:
                print("Found " + repr(locs.shape[0]) + " keypoints at level " + repr(k+1) + ".\n")
                print("Refining keypoints...\n")
            """ IJ, UV coordinates need sorting out"""
            # Perform subpixel refinement, also returning the mask to filter out unstable points from the values vector
            locs, stable = self._refinePeaks(self.energymaps[k], locs[:,::-1]) # N.B., takes uv coordinates, not ij like the MATLAB version
            # The returned locs is therefore in ij coordinates and must be reversed again
            
            # Mask off the unstable energy values
            values = values[stable]
            
            # Stack the relevant attributes into the keypoint list element for this level
            kps.append(np.array(np.hstack([(locs[:,::-1]+1)*2**(k+1)-1, np.ones([locs.shape[0], 1])*2**(k+1), values[:,None]])))
            if self.verbose:
                print(repr(locs.shape[0]) + " keypoints were stable at level " + repr(k+1) + ".\n")
            
            # Eliminate keypoints with responses below the threshold
            print(values)
            print(kps[-1].shape)
            kps[-1] = kps[-1][values>threshold,:]
            print(kps[-1].shape)
            if self.verbose:
                print(repr(kps[-1].shape[0]) + " keypoints were above the threshold.\n")
                            
        # Convert the unwieldy list into a numpy array
        kps = np.vstack(kps)
        # Sort the array by energy. It is descending by default and reversed on return
        kps = kps[np.argsort(kps[:,-1]),:]
        
        return kps[::-1,:]
    
    def draw_keypoints(self, kps, img):
        """Function for drawing keypoints from multiple scales
        overlaid on an image. 
        Valid format is a vertical stack of keypoints formatted as follows:
        [u,v,scale,energy]

        By S.C. Forshaw, July 2014."""

        imshow(img, cmap=cm.gray)
        axis('image')
        t = np.hstack([np.nan, np.arange(0,21*np.pi/10,np.pi/10), np.nan])
        (plot(np.transpose(kps[:,0,None]+2*kps[:,2,None]*np.cos(t)[:,None].T), 
         np.transpose(kps[:,1,None]+2*kps[:,2,None]*np.sin(t)[:,None].T), color='y'))
            
        scatter(kps[:,0], kps[:,1], marker='+', color='m')
        
    def draw_maps(self): 
        """Plots the keypoint energy maps."""
        for p in range(self.firstlevel,self.nlevels):
            subplot(1,self.nlevels-self.firstlevel,p-self.firstlevel+1)
            imshow(self.energymaps[p])
            
    def _clamp(self, x, a, b):
        return np.minimum(np.maximum(x,a),b)

    def _refinePeaks(self, X, coords):
        """Takes a column of ij coords, and uses quadratic fitting to improve
        the peak location estimates.  If a peak moves by more than 1 in any 
        direction it is removed."""
        
        # Adapted from Pashmina's code
        # There is probably a more efficient way of setting this up.
        q = np.ones([9,1])
        z = np.array([[-1,0,1,-1,0,1,-1,0,1], [-1,-1,-1,0,0,0,1,1,1]]).T
        Q = np.hstack((q, z, 0.5*z*z, z[:,0].reshape(9,1)*z[:,1].reshape(9,1)))
        q[np.sum(np.abs(z),1)==1] = 2
        q[np.sum(np.abs(z),1)==0] = 4
        Q = Q*(q*np.ones([1,6]))
        
        # Compute the pseudoinverse of Q, and include the weighting operation in
        # the product too (no need to add an extra stage later)
        R = np.linalg.solve(Q.T.dot(Q),Q.T.dot(np.diagflat(q)))
        
        # Add on offsets to the 3x3 region, one region per column
        r = coords[:,0,None].T + z[:,0,None]
        c = coords[:,1,None].T + z[:,1,None]

        # Work out the parameters of each quadratic surface, one surface
        # per column.
        rr = self._clamp(r, 1, X.shape[0])
        cc = self._clamp(c, 1, X.shape[1])

        quadParams = R.dot(X[rr.astype(int), cc.astype(int)])
        moves = np.empty([2,quadParams.shape[1]])
        
        # Main speed-critical loop
        for n in range(0, quadParams.shape[1]):
            a = quadParams[:,n]
            gradx = np.array([[a[1]],[a[2]]])
            hessian = np.array([[a[3], a[5]], [a[5], a[4]]])
            moves[:,n] = np.linalg.solve(-1*hessian, gradx).T
        
        stable = np.logical_and(np.abs(moves[0,:])<1, np.abs(moves[1,:])<1)
        moves = moves[:,stable]
        coords = coords[stable,:]
        
        return coords + moves.T, stable

def slp2interleaved(img, nlevels=5, full=True, firstlevel=1, plots=False, verbose=False, sampleLocs=None, kpmethod='gale', edge_suppression=None):
    """Function with options to be specified as
    one would an slp2 object. The input image is scaled down
    by a factor of sqrt(2) and a second DTCWT & SLP performed.
    The results are interleaved so that the outputs are lists
    of numpy arrays in increasing order of coarseness.
    Keypoints are computed by default. In general, unless one 
    has a special reason to double the sampling rate in scale,
    one should use the slp2 class and its functions.
    """

    # Input parameters are as for an SLP object declaration
    scaledImage = list([None,None])
    gamma = list([None,None])
    hists = list([None,None])
    kps = list([None,None])
    if sampleLocs is None:
        sampleLocs = list([None,None])

    for n in range(2):
        udim = img.shape[1]/np.sqrt(2)**n
        vdim = img.shape[0]/np.sqrt(2)**n
        if np.mod(np.ceil(udim), 2) == 0:
            udim = np.ceil(udim)
            upad = 0
        else:
            udim = np.ceil(udim+1)
            upad = np.mod(np.ceil(udim), 2)

        if np.mod(np.ceil(vdim), 2) == 0:
            vdim = np.ceil(vdim)
            vpad = 0
        else:
            vdim = np.ceil(vdim+1)
            vpad = np.mod(np.ceil(vdim), 2)

        # Rescale the input image FIXME: and round to the nearest even integer
        scaledImage = scipy.misc.imresize(img, (int(vdim+vpad), int(udim+upad)), interp='bilinear')
        s = slp2(scaledImage, nlevels=nlevels, full=full, firstlevel=firstlevel, plots=plots, verbose=verbose)
        # If sampleLocs for this tree is None, no precomputed locations were given, so compute them
        if sampleLocs[n] is None:
            sampleLocs[n] = s.init()
        
        # Actual slp operations
        gamma[n] = s.transform(scaledImage, sampleLocs[n])
        hists[n] = s.histgen(gamma[n])
        if kpmethod == 'forshaw':
            print('Using Forshaw keypoint detector.')
            kps[n] = s.keypoints(hists[n], method='forshaw', edge_suppression=2, threshold=5)
        else:
            print('Using Gale keypoint detector')
            # FIXME: This produces keypoints with huge energy values for some reason, haven't found the cause yet
            kps[n] = s.keypoints(gamma[n], method='gale', edge_suppression=None, threshold=10000)
        # Scale the keypoints as necessary
        kps[n][:,0:2] = kps[n][:,0:2] * (np.sqrt(2)**n)
        kps[n][:,2] = kps[n][:,2] / (np.sqrt(2)**n)

    # rearrange the outputs in order of increasing coarseness
    gamma = [gamma[i][j] for j in range(len(gamma[0])) for i in (0,1)]
    hists = [hists[i][j] for j in range(len(hists[0])) for i in (0,1)]
    
    # Re-arrange keypoints in decreasing order of saliency
    kps = np.vstack(kps)
    kps = kps[kps[:,-1].argsort()[::-1],:]
    return kps, gamma, hists, sampleLocs