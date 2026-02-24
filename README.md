# FRB plotter

Three standalone Python scripts for making publication-style plots of FRB dynamic spectra from common data formats:

- **HDF5** (`.h5`) → `plot_frb_h5.py`
- **SIGPROC filterbank** (`.fil`) → `plot_frb_fil.py`
- **NumPy array** (`.npy`) → `plot_frb_npy.py`

Each script can optionally:
- downsample in time/frequency,
- auto-find a pulse and cut a window around it,
- manually flag frequency bands,
- create a symmetric time axis centered on the pulse,
- save a PNG.

> Note: the scripts set `matplotlib` backend to `TkAgg` (GUI). On headless clusters you may need an X-forwarding session or change the backend to `Agg`.

---

## Files
- plot_frb_h5.py
- plot_frb_fil.py
- plot_frb_npy.py
- LICENSE
- requirements.txt
- README.md
