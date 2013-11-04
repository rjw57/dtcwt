Variant transforms
==================

In addition to the basic 1, 2 and 3 dimensional DT-CWT, this library also
supports a selection of variant transforms.

Rotational symmetry modified wavelet transform
----------------------------------------------

For some applications, one may prefer the subband responses to be more rotationally similar. 

In the original DTCWT, the 45 and 135 degree subbands have passbands whose centre frequencies 
are somewhat furtherfrom the origin than those of the other four subbands. This results from 
the combination of two highpass 1-D wavelet filters. The remaining subbands combine highpass
and lowpass 1-D filters, and their centre frequencies are a factor of approximately sqrt(1.8) 
closer to the origin of the frequency plane.

The dtwavexfm2b() function, when used with 'near_sym_b_bp' and 'qshift_b_bp' parameters, employs 
an alternative bandpass 1-D filter in place of the highpass filter for the appropriate subbands.

While the Hilbert transform property of the DTCWT is preserved, perfect reconstruction is lost.
However, in applications such as machine vision, where all subsequent operations on the image
take place in the transform domain, this is of relatively minor importance.

For full details, refer to:

N. G. Kingsbury. Rotation-invariant local feature matching with complex
wavelets. *In Proc. European Conference on Signal Processing (EUSIPCO)*,
pages 901–904, 2006. 2, 18, 21



