![nrutils](https://github.com/llondon6/nrutils_dev/blob/master/media/nrutils_banner.png?raw=true)

# Installation
### Use pip:
```bash
pip install git+https://github.com/llondon6/nrutils_dev.git
```
### Or clone, then install:
```bash
git clone https://github.com/llondon6/nrutils_dev.git
pip install .
```
### Or clone, then install in editable mode (useful if installing in a virtual environment):
```bash
git clone https://github.com/llondon6/nrutils_dev.git
pip install -e .
```

# Working with Numerical Relativity simulation catalogs

**nrutils** uses a class based system where the simulation (i.e. its related metadat) is encapsulated a "simulation catalog entry" or "scentry" class. General purpose waveforms are encapsulated by the `gwf` (graviational waveform) class, but the user will most likey be interested in the `gwylm` class, which loads desired spherical harmonic multipoles, and then (semi-automatically) performs low level processing of data (e.g. strain calculation, ffts, windowing, plotting). In the context of the gwylm class, individual multipoles are `gwf` objects.

## 2. Setup .ini files
If you installed `nrutils` in editable mode then you should edit the files in the cloned
repository otherwise you should edit the files installed in you python environment.

* nrutils/config/bam.template -> nrutils/config/bam.ini
* nrutils/config/sxs.template -> nrutils/config/sxs.ini
* nrutils/config/bam_gw150914_followup.template -> nrutils/config/bam_gw150914_followup.ini

## 3. (Optional) Building a catalogue

This is done using the command ```scbuild()``` to create a database of the NR simulations.
```to be expanded on soon```
