########################################################################################################################
# Assemble jacobian.
# Note: TWO DMs ARE ASSUMED.
G_3D = np.zeros((self.num_wavelengths_estimating, 2 * self.n_pix_dh, 2 * self.num_actuators))
for i, wavelength in enumerate(self.estimating_wavelengths):
    # for wavelength in self.estimating_wavelengths:
    jacobian = hcipy.read_fits(self.jacobian_filenames[wavelength])
    # only take dark hole region of jacobian, multiply by contrast normalization to put in counts
    G_i = jacobian[:, np.tile(self.dark_zone, 2)] * e_field_contrast_normalization_lambda[
        wavelength]  

    G_3D[i, :, :] = G_i.T  # transpose to be 2n_pix x 2m_act

self.G_3D = G_3D  # re;im are stacked in blocks, not alternating
# Reorder to put real and imaginary comps for each pixel together:
G_reordered = np.zeros(G_3D.shape, G_3D.dtype)
G_reordered[:, ::2, :] = G_3D[:, :G_3D.shape[1] // 2, :]
G_reordered[:, 1::2, :] = G_3D[:, G_3D.shape[1] // 2:, :]
G_reordered = G_reordered.reshape(2 * self.n_pix_dh * self.num_wavelengths_estimating, 2 * self.num_actuators)

# Probably a smarter way to do this
i_s = np.arange(start=0, stop=2 * (self.num_wavelengths_estimating) * self.n_pix_dh, step=2 * self.n_pix_dh)
i_e = i_s + 1
i_se = np.array([[i, j] for i, j in zip(i_s, i_e)]).ravel()

# Rearrange to group pixels instead of wavelengths:
# G = [ lambda1 (Re{g_1}; Im{g_1};) ; lambda2(Re{g_1}; Im{g_1};) ; lambda_l(Re{g_1}; Im{g_1};); lambda1(Re{g_2}; Im{g_2};)...]
self.G = []
self.G = np.reshape([G_reordered[i_se + factor, :] for factor in range(0, 2 * self.n_pix_dh, 2)],
                    (2 * self.n_pix_dh * self.num_wavelengths_estimating, 2 * self.num_actuators))

########################################################################################################################
# dI matrix:
I_deltas = I_deltas.T.reshape(self.n_pix_dh, self.num_probes, 1)
self.dys = I_deltas



########################################################################################################################
# Estimate e-field

gdus = []
# First calculate Gu for each probe:
# for i, probe in enumerate(self.probes):
for i in range(self.num_probes):
    # THIS IS A BIT OF A HACK
    probe = self.probes[i, :] * self.probe_amp_scaling
    gdus.append(4 * self.G.dot(probe).reshape((-1, 1, 2 * self.num_wavelengths_estimating)))

# concatenating gdus so that its shape is (number of pixels)x(number of probes)x(number of wavelengths times two)
gdus = np.concatenate(gdus, axis=1)

# computing the pseudo-inverses of each matrix in gdus in 3 steps.
# First, Z is an array of square matrices and its shape is
# (number of pixels)x(number of wavelengths times two)x(number of wavelengths times two)
Z = np.matmul(gdus.transpose((0, 2, 1)), gdus)
# then computing the inverse of the square matrices in Z
alph = 1e-15
Z_inv = np.linalg.pinv(Z, alph)  # see if this is sufficient

# then the pseudo-inverses,
gdus_pinv = np.matmul(Z_inv, gdus.transpose((0, 2, 1)))

# print('gdus_pinv shape:',gdus_pinv.shape)

# computing the electric field by solving eq.26,
# should produce an array that is 1x(2*num wavelengths)x(number of pixels)
x_hat = np.matmul(gdus_pinv, self.dys)

# reshape to grid to save while in an easier format to split Re/Im
# NEED TO DO THIS FOR EACH WAVELENGTH
for i, wavelength in enumerate(self.estimating_wavelengths):

    # Create return variable in hcipy format
    E = hcipy.Field(np.zeros(wfsc_utils.focal_grid.size, dtype='complex'), wfsc_utils.focal_grid)

    E[self.dark_zone] = np.reshape(x_hat[:, 2 * i, :] + 1j * x_hat[:, 2 * i + 1, :], self.n_pix_dh)
