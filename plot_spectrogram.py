from obspy import read
from obspy.signal import PPSD

tr = read("noise1.fourierdomain.mseed")
psd = PPSD(tr[0].stats, metadata={"sensitivity": 1.0},
    ppsd_length=900.,
    special_handling="ringlaser", period_step_octaves=0.00625)
psd.add(tr)
psd.plot()

tr = read("noise1.timedomain.mseed")
psd = PPSD(tr[0].stats, metadata={"sensitivity": 1.0},
    ppsd_length=900.,
    special_handling="ringlaser", period_step_octaves=0.00625)
psd.add(tr)
psd.plot()