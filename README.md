![nrutils](https://github.com/llondon6/nrutils_dev/blob/master/media/nrutils_banner.png?raw=true)

**nrutils** is a low-level suite for working with Numerical Relativity data (i.e. gravitational waveforms) from binary black hole mergers.

# Table of Contents
1. [Installation](#installation)
2. [Working with Numerical Relativity simulation catalogs](#working-with-numerical-relativity-simulation-catalogs)
3. [Usage Examples](#examples)

# Installation (OSX+Linux)
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

**nrutils** uses a class based system where the simulation (i.e. its related metadata) is encapsulated a "simulation catalog entry" or "scentry" class. General purpose waveforms are encapsulated by the `gwf` (graviational waveform) class, but the user will most likey be interested in the `gwylm` class, which loads desired spherical harmonic multipoles, and then (semi-automatically) performs low level processing of data (e.g. strain calculation, ffts, windowing, plotting). In the context of the gwylm class, individual multipoles are `gwf` objects.

To get a sense of how these concepts work togerther please see the core usage example. **NOTE** that you will need to configure or build catalog files before trying to run this example. Please see sections below for how to do this.
* [core_ipython_example.ipynb](https://github.com/llondon6/nrutils_dev/blob/master/examples/core_ipython_example.ipynb)

## Working with catalog files
If you installed `nrutils` in editable mode then you should edit the files in the cloned
repository otherwise you should edit the files installed in your python environment.

* nrutils/config/bam.template -> nrutils/config/bam.ini
* nrutils/config/sxs.template -> nrutils/config/sxs.ini
* nrutils/config/bam_gw150914_followup.template -> nrutils/config/bam_gw150914_followup.ini

## Building your own catalog

This is done using the command ```scbuild()``` to create a database of the NR simulations. (Documentaion incomplete) 

The related example ipython notebook is:
* [core_ipython_example.ipynb](https://github.com/llondon6/nrutils_dev/blob/master/examples/core_ipython_example.ipynb)

## Using a stock catalog or one shared by a friend
The related example ipython notebook is:
* [reconfigure_catalog_example.ipynb](https://github.com/llondon6/nrutils_dev/blob/master/examples/reconfigure_catalog_example.ipynb)

# Examples

The best way to learn what's in nrutils is to brows through the package contents and skim over the comments. The second best way to figure out what's going on is to skim though the example ipython notbooks located in the examples folder:
* [Usage Examples](https://github.com/llondon6/nrutils_dev/blob/master/examples)
