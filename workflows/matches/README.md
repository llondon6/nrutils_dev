# Matches (Faithfulness) with NR waveforms
---

### WARNING!!!!

This workflow is intended as an example only. You should copy the workflow folder to your own directory, and then edit it according to your needs. This specific example uses LALSimulation versions of PhenomHM and PhenomD to compute matches; the waveform calls are in template_wfarr_fun and signal_wfarr_fun as found in src.py -- you'll want to change this for your purposes. Also note that LAL is sourced in "work". You will likely also want to change this.

### Requirements

LAL, nrutils, positive

### Outline

* Given an NR waveform (some identifying keyword), calculate matches.
* The NR waveform will be considered to be the *signal*.
* An aproximant (from LAL) will be the *template*.
* What will be varied? What will be fixed?
 * The template and signal will have the same fixed masses and spins.
 * Inclinations between 0 and pi will be considered.
 * For each inclination, a range of signal polarization and orbital phase will be considered.
 * The low-level match is optimized over template polarization and orbital phase.

### How to run

See the ```config.ini``` file for settings. When you've input the desired settings, run

```python
./run_thems_matches
```
