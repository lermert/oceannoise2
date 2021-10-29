# -----------------------------------------------------------------------------
# INPUT

# synthetics from axisem3d:
input_file = "greens/force.pressure3D..MXZ.h5"
nsrc_file = "microseisms/source1.h5"
duration_seconds = 7200.0 #2. * 86400. # 2 days
outname = "noise1"
scale_factor = 1.e-10  # scaling factor to correct for input force of sem sim.
# Include here also water compressibility and the moment of the force source :)
# To Do
compressibility_times_moment = 1.0  # unit: Nm/Pa, i.e. 1/m
# nr of sources for time domain dirac comb:
n_sources = 10
# -----------------------------------------------------------------------------


import numpy as np
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
from obspy.signal.invsim import cosine_taper
import matplotlib.pyplot as plt
from obspy import Trace
import h5py

def kristiinas_source_generator(duration_in_samples, n_sources=1, domain="time"):
 # ...plus other parameters as needed
    # put the function here that generates a random phase spectrum
    # according to parameters given by the user
    if domain == "time":

        a = np.zeros(duration_in_samples)
        rix = np.random.randint(0, duration_in_samples, n_sources)
        a[rix] = 1.0

    elif domain == "frequency":

        freq = np.fft.rfftfreq(n=duration_in_samples)
        a = np.random.random(freq.shape)
    else:
        raise ValueError("Unknown domain " + domain)

    return(a)


def generate_timeseries(input_file, duration_seconds, nsrc_file,
                        all_ns, fs, domain="frequency", n_sources=1):
    """
    Generate a long time series of noise at two stations
    (one station in case of autocorrelation)
    input_files: list of 2 wave field files
    all_conf: Configuration object
    nsrc: Path to noise source file containing spatial
    distribution of noise source pressure PSD
    all_ns: Numbers specifying trace lengths / FFT parameters:
    all_ns = (nt, n, n_corr, Fs) where nt is the length
    of the Green's function seismograms in samples,
    n is the length to which it is padded for FFT, 
    n_corr is the nr of samples of the resulting cross-correlation,
    and fs is the sampling rate.
    """

    td_taper = cosine_taper(all_ns[0])
    npad = all_ns[1]
    fs = all_ns[3]

    # first the Green's functions are opened
    wf1 = h5py.File(input_file, "r")

    # the noise source power spectral density model is opened
    nsrc = h5py.File(nsrc_file, "r")
    mod_coeff = np.asarray(nsrc['model'][:])
    spect_basis = np.asarray(nsrc['spectral_basis'][:])
    surf_area = nsrc["surface_areas"][:]
    # allocate arrays for the output
    duration_in_samples = int(round(duration_seconds * fs))
    trace1 = np.zeros(duration_in_samples)
    
    freq_axis = np.fft.rfftfreq(n=duration_in_samples, d=1. / fs)
    fd_taper = cosine_taper(len(freq_axis), 0.01)

    # loop over source locations
    # This loop should maybe be parallelized To Do
    for i in range(wf1["stats"].attrs["ntraces"]):
        # read green's functions
        g1 = np.ascontiguousarray(wf1["data"][i, :] * td_taper)
        # two possibilities here:
        # a) FFT Green's function, define the random spectrum in
        # Fourier domain, and multiply, then inverse FFT

        if domain == "frequency":
            # read source power spectral density
            source_amplitude = np.dot(mod_coeff[i, :],
                                      spect_basis)
            if source_amplitude.sum() == 0:
                continue

            # since the spectrum of the noise source model in noisi is differently sampled
            fdfreq = np.fft.rfftfreq(n=npad, d=1. / fs)
            f = interp1d(fdfreq, source_amplitude, kind="cubic", bounds_error=False, fill_value=0)
            source_amplitude = f(freq_axis) * fd_taper

            # call the function to get the random phase spectrum
            source_phase = kristiinas_source_generator(duration_in_samples,
                                                       domain="frequency")
            # By definiton: 
            # complex number z = A exp(i phi)
            P = source_amplitude * np.exp(1.j * 2. * np.pi * source_phase)  # phase now between 0 and 2 pi
            p = np.fft.irfft(P, n=duration_in_samples)            
            trace1 += fftconvolve(g1, p, mode="full")[0: duration_in_samples] * surf_area[i]

        # b) define a time series with random "onsets" in time domain,
        # and run convolution in the frequency domain by scipy
        # I may have driven my PhD advisor insane with my love for the time domain
        elif domain == "time":
            source_amplitude = np.dot(mod_coeff[i, :],
                                      spect_basis)
            source_amplitude = np.fft.irfft(source_amplitude, n=npad)
            if source_amplitude.sum() == 0:
                print("zero amplitude, skip")
                continue
            # steps here would be:
            # call the function that gets the random onset time series
            source_phase = kristiinas_source_generator(duration_in_samples,
                                                       domain="time",
                                                       n_sources=n_sources) / n_sources
            # it becomes a bit more complicated if there is spatial correlation
            # rint(source_amplitude.shape, source_phase.shape)
            source = fftconvolve(source_amplitude, source_phase, mode="full")[0: duration_in_samples]
            trace1 += fftconvolve(g1, source, mode="full")[0: duration_in_samples] * surf_area[i]

            if i % 500 == 0:
                print("Created {} of {} source spectra.".format(i, wf1["stats"].attrs["ntraces"]))

    return(trace1, source_phase)


# input_file = "greensfcts/pressure1d.displacement..MXZ.h5"
# nsrc_file = "noisesources/source1.h5"
# duration_seconds = 86400.
# outname = "pressure1d.displacement..MXZ"
# # nr of sources for time domain dirac comb:
# n_sources = 10

def run_s4(input_file, nsrc_file, duration_seconds, n_sources, outname):

    with h5py.File(input_file, "r") as wf_test:
        all_ns = [wf_test["stats"].attrs["nt"],
                  wf_test["stats"].attrs["npad"], 0,
                  wf_test["stats"].attrs["Fs"]]
        fs = wf_test["stats"].attrs["Fs"]

    # Time domain "Dirac comb" (the waveform of the noise wavelet is always the same)
    trace1, source =  generate_timeseries(input_file, duration_seconds, nsrc_file,
                                          all_ns, fs, domain="time",
                                          n_sources=n_sources)

    tr1 = Trace(data=trace1 * scale_factor / compressibility_times_moment)
    tr1.stats.sampling_rate = fs
    tr1.stats.channel="MXZ"
    tr1.write("{}.timedomain.mseed".format(outname), format="MSEED")

    # Frequency domain (the waveform of the noise wavelet is random)
    trace1, source =  generate_timeseries(input_file, duration_seconds, nsrc_file,
                                                 all_ns, fs, domain="frequency")

    tr1 = Trace(data=trace1 * scale_factor)
    tr1.stats.sampling_rate = fs
    tr1.stats.channel="MXZ"
    tr1.write("{}.fourierdomain.mseed".format(outname), format="MSEED")


if __name__ == "__main__":
    run_s4(input_file, nsrc_file, duration_seconds, n_sources, outname)