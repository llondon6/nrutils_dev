# Matches (Faithfulness) with NR waveforms
---

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
