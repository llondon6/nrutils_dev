import romSpline
import h5py
import sys

# Usage example:
#   python h5splinetotxt.py <h5file> <group> <coorddataset>

h5file = sys.argv[1]
group = sys.argv[2]
coords_dataset = sys.argv[3]

s_h5 = romSpline.readSpline(h5file, group)

f = h5py.File(h5file)

coords = f[coords_dataset]
vals = s_h5(coords)

for t,f in zip(coords,vals):
    print "%.19g\t%.19g" % (t,f)
