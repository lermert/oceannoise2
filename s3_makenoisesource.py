#  INPUT
noise_source_output_file = "microseisms/source1.h5"
grid_file = "sourcegrid_3D.npy"
src_model_file = "oceanmodels/pressure_PSD_2007-12-11-00.npz"
greens_function_file = "greens/force.pressure3D..MXZ.h5"
# Number of basis functions for representing spectra:
n_basis = 100
# represent spectra between fmin and fmax (noisi source needs all freq. between 
# 0 and Nyquist)
fmin = 0.0
# End input

import h5py
import numpy as np
from scipy.signal import tukey
from scipy.interpolate import interp1d
from sincbasis import SincBasis
from math import sin

def get_approx_surface_elements(lon, lat, r=6.378100e6):

    if len(lon) != len(lat):
        raise ValueError('Grid x and y must have same length.')

    # surfel
    surfel = np.zeros(lon.shape)
    colat = 90. - lat

    # find neighbours
    for i in range(len(lon)):

        # finding the relevant neighbours is very specific to how
        # the grid is set up here (in rings of constant colatitude)!
        # get the nearest longitude along the current colatitude
        current_colat = colat[i]
        if current_colat in [0., 180.]:
            # surface area will be 0 at poles.
            continue

        colat_idx = np.where(colat == current_colat)
        lon_idx_1 = np.argsort(np.abs(lon[colat_idx] - lon[i]))[1]
        lon_idx_2 = np.argsort(np.abs(lon[colat_idx] - lon[i]))[2]
        closest_lon_1 = lon[colat_idx][lon_idx_1]
        closest_lon_2 = lon[colat_idx][lon_idx_2]

        if closest_lon_1 > lon[i] and closest_lon_2 > lon[i]:
            d_lon = np.abs(min(closest_lon_2, closest_lon_1) - lon[i])

        elif closest_lon_1 < lon[i] and closest_lon_2 < lon[i]:
            d_lon = np.abs(max(closest_lon_2, closest_lon_1) - lon[i])

        else:
            if closest_lon_1 != lon[i] and closest_lon_2 != lon[i]:
                d_lon = np.abs(closest_lon_2 - closest_lon_1) * 0.5
            else:
                d_lon = np.max(np.abs(closest_lon_2 - lon[i]),
                               np.abs(closest_lon_1 - lon[i]))

        colats = np.array(list(set(colat.copy())))
        colat_idx_1 = np.argsort(np.abs(colats - current_colat))[1]
        closest_colat_1 = colats[colat_idx_1]
        colat_idx_2 = np.argsort(np.abs(colats - current_colat))[2]
        closest_colat_2 = colats[colat_idx_2]

        if (closest_colat_2 > current_colat
            and closest_colat_1 > current_colat):

            d_colat = np.abs(min(closest_colat_1,
                                 closest_colat_2) - current_colat)

        elif (closest_colat_2 < current_colat and
              closest_colat_1 < current_colat):
            d_colat = np.abs(max(closest_colat_1,
                                 closest_colat_2) - current_colat)

        else:
            if (closest_colat_2 != current_colat
                and closest_colat_1 != current_colat):
                d_colat = 0.5 * np.abs(closest_colat_2 - closest_colat_1)
            else:
                d_colat = np.max(np.abs(closest_colat_2 - current_colat),
                                 np.abs(closest_colat_1 - current_colat))

        surfel[i] = (np.deg2rad(d_lon) *
                     np.deg2rad(d_colat) *
                     sin(np.deg2rad(colat[i])) * r ** 2)

    return(surfel)

def run_s3(noise_source_output_file, grid_file,
           src_model_file, greens_function_file, n_basis, fmin):
    microseism_model = np.load(src_model_file)
    # frequency axis
    with h5py.File(greens_function_file, "r") as wf:
        df = wf["stats"].attrs["Fs"]
        n = wf["stats"].attrs["npad"]

    freqs_new = np.fft.rfftfreq(n, d=1. / df)
    source_grid = np.load(grid_file)
    latmin = source_grid[1].min()
    latmax = source_grid[1].max()
    lonmin = source_grid[0].min()
    lonmax = source_grid[0].max()

    # pressure model
    print('min frequency', microseism_model['freqs'].min())
    print('max frequency', microseism_model['freqs'].max())
    print('min latitude', microseism_model['lats'].min())
    print('max latitude', microseism_model['lats'].max())
    print('min longitude', microseism_model['lons'].min())
    print('max longitude', microseism_model['lons'].max())
    print("Pressure source on ", microseism_model["date"], "\nUnit ",
          microseism_model["pressure_PSD_unit"])
    freqs_old = microseism_model["freqs"]
    sampling_frequency = np.diff(freqs_old).mean()
    lats = microseism_model["lats"]
    
    lons = microseism_model["lons"]   
    p = microseism_model["pressure_PSD"]
    p = np.nan_to_num(p)  # set nans to 0

    # now, we first have to create a spectral basis for the new model
    sb = SincBasis(K=n_basis, N=len(freqs_new), freq=freqs_new,
                  fmin=fmin, fmax=freqs_new.max())

    spect_basis =np.zeros((n_basis, len(freqs_new)), dtype=np.float)

    # add vectors
    for i in range(n_basis):
        spect_basis[i, :] = sb.basis_func(k=i, n=len(freqs_new))

    # coefficient matrix
    n_loc = source_grid.shape[1]
    mod = np.zeros((n_loc, n_basis), dtype=np.float)

    # for each location, we have to fit the spectrum of the microseism model
    # this is a bit slow
    for i in range(n_loc):
        # nearest point, approximately (this is not exact, do we need more exact?)
        # To Do: Use a more exact distance calc, like Haversine
        # To Do: Interpolate between nearest locations? e.g. weighted average
        ix_lon = np.argmin((lons - source_grid[0, i]) ** 2)
        ix_lat = np.argmin((lats - source_grid[1, i]) ** 2)
        # print(ix_lon, ix_lat, source_grid[0, i], source_grid[1, i], lons[ix_lon], lats[ix_lat])
        coeffs = np.zeros(n_basis)
        spec = p[:, ix_lat, ix_lon].copy()

        # PSD-->amplitude
        spec *= sampling_frequency  # divide by time interval
        spec /= (50000. ** 2)  # divide by spatial sampling rate in 2 D APPROXIMATE
        spec = np.sqrt(spec)

        if i % 1000 == 0:
            print(i, end=",")
        
        f = interp1d(freqs_old, spec, bounds_error=False, fill_value=0, kind="cubic")
        # To Do : Create a taper depending on distance from source that falls of to 
        # zero in the outermost 1-2 wavelenghts of the source area
        spec_n = f(freqs_new)

        for j in range(n_basis):
            coeffs[j] = np.dot(spect_basis[j, :], spec_n)
        # and finally enter the coefficients in the new model
        mod[i, :] = coeffs

    # Save to an hdf5 file
    with h5py.File(noise_source_output_file, 'w') as fh:
        fh.create_dataset('coordinates', data=np.load(grid_file))
        fh.create_dataset('frequencies', data=freqs_new)
        fh.create_dataset('surface_areas',
                          data=get_approx_surface_elements(source_grid[0], source_grid[1]))
        fh.create_dataset("spectral_basis", data=spect_basis)
        fh.create_dataset("model", data=mod)

if __name__ == "__main__":

    run_s3(noise_source_output_file, grid_file,
           src_model_file, greens_function_file, n_basis, fmin)
