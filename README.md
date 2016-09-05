![nrutils](https://github.com/llondon6/nrutils_dev/blob/master/media/nrutils_banner.png?raw=true)

## 1. Install Options
###Use pip:
```bash
pip install git+https://github.com/llondon6/nrutils_dev.git
```
###Or clone, then install:
```bash
git clone https://github.com/llondon6/nrutils_dev.git
pip install .
```
###Or clone, then install in editable mode:
```bash
git clone https://github.com/llondon6/nrutils_dev.git
pip install -e .
```

## 2. Setup .ini files
If you installed `nrutils` in editable mode then you should edit the files in the cloned
repository otherwise you should edit the files installed in you python environment.

1. nrutils/settings/paths.template -> nrutils/settings/paths.ini
2. nrutils/config/bam.template -> nrutils/config/bam.ini
3. nrutils/config/sxs.template -> nrutils/config/sxs.ini
4. nrutils/config/bam_gw150914_followup.template -> nrutils/config/bam_gw150914_followup.ini

## 3. (Optional) Building a catalogue

This is done using the command ```scbuild()``` to create a database of the NR simulations.
```to be expanded on soon```
