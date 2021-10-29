# read in from Claudia's runs
import numpy as np
from netCDF4 import Dataset
import os
import h5py
from glob import glob
from obspy import Trace
from obspy.geodetics import gps2dist_azimuth, degrees2kilometers
from obspy.signal.invsim import cosine_taper
import matplotlib.pyplot as plt
# input
# -------------------
indir = "../Axisem3D_inputs_outputs/output_3D_cartesian/stations/oceansurface_stations/"
sourcegrid = np.load("sourcegrid_3D.npy")
f_out = "force.pressure3D..MXZ.h5" #"force1d.pressure..MXZ.h5"
channel_index = 0
physical_quantity = "P"  # put P for pressure, DIS for displacement
reference_station = "net1.sta1"
stafile = "STATIONS_kristiina"
#nt_keep = 3000  # throw whatever comes after this timestep
earth_r = 6371000.0 
fmin=0.01
fmax=0.2
filtord=8
src_x = -16.5
src_y = 37.5
# -------------------

# read in stations -> rank and array index file
fstas = open(os.path.join(indir, "rank_station.info"), "r")
fstas = fstas.read().split("\n")
sta_to_rank = {}
for sta in fstas:
    if sta == "":
        continue
    inf = sta.split()
    sta_to_rank[inf[1]] = [inf[0], inf[2]]  # station string identifies rank, index in rank

# get station locations from STATIONS file
rec_lat = []
rec_lon = []
stanames = []
fstas = open(stafile, "r")
fstas = fstas.read().split("\n")
for sta in fstas:
    if sta == "":
        continue
    inf = sta.split()
    if len(inf) == 0: continue
    stanames.append("II." + inf[0])
    rec_lat.append(float(inf[2]))
    rec_lon.append(float(inf[3]))
rec_lat = np.array(rec_lat)
rec_lon = np.array(rec_lon)
# set up the hdf output file
f_out = h5py.File(f_out, "w")

# open one file to get time vector
testfile = Dataset(glob(os.path.join(indir, "*.nc*"))[0])
t = testfile["data_time"][:]

# DATASET NR 1: STATS
stats = f_out.create_dataset('stats', data=(0,))
stats.attrs['reference_station'] = reference_station
stats.attrs['data_quantity'] = physical_quantity
stats.attrs['ntraces'] = sourcegrid.shape[-1]
fs = round(1./ np.diff(t).mean(), 5)
stats.attrs['Fs'] = fs
stats.attrs['nt'] = len(t)
npts = len(t)
if npts % 2 == 1:
    npad = 2 * npts - 2
else:
    npad = 2 * npts
stats.attrs["npad"] = npad
stats.attrs['fdomain'] = False

# DATASET NR 2: Source grid
f_out.create_dataset('sourcegrid', data=sourcegrid)

# DATASET Nr 3: Seismograms itself
data_out = f_out.create_dataset('data', (sourcegrid.shape[1], len(t)),
                                dtype=np.float)

taper = cosine_taper(len(t), p=0.05)
taper[0: len(taper) // 2] = 1.
# for all noise source locations:

for i in range(sourcegrid.shape[-1]):
    x = sourcegrid[0, i]
    y = sourcegrid[1, i]
    # print(rec_lat.min(), rec_lat.max())
    # print(rec_lon.min(), rec_lon.max())
    # print(sourcegrid[1].min(), sourcegrid[1].max())
    # print(sourcegrid[0].min(), sourcegrid[0].max())

    # find the station(s) 
    # rec_lat and rec_lon are the actual same latitudes
    # as in the sourcegrid file. Therefore, we are using
    # this simple way of matching the locations to an index here.
    ix_d1 = np.argmin((rec_lat - y)**2 + (rec_lon - x)**2)
    stastr1 = stanames[ix_d1]
    # get the time series
    try:
        rank_ix1 = sta_to_rank[stastr1]
    except:
        print(stastr1)
        continue
    #print(rank_ix)
    dd = Dataset(os.path.join(indir, "axisem3d_synthetics.nc.rank{}".format(rank_ix1[0])))
    # print(dd["data_wave"].shape)
    seis1 = dd["data_wave"][rank_ix1[1], channel_index, :]
    seis1 *= taper
    trace1 = Trace(data=seis1)
    trace1.stats.sampling_rate = fs
    trace1.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=filtord)

    data_out[i, :] = trace1.data
    if i % 200 == 0:
        print("{} down, {} to go".format(i, sourcegrid.shape[-1] - i))
        dist = gps2dist_azimuth(src_x, src_y, x, y)[0] / 1000.0
        taxis = np.linspace(0., trace1.stats.npts / trace1.stats.sampling_rate, trace1.stats.npts)
        plt.plot(taxis, trace1.data / trace1.data.max() + dist/5., color="0.7", alpha=0.5)
    plt.xlabel("Seconds after source")
    plt.ylabel("Norm. waveforms")
    
plt.show()

f_out.close()








