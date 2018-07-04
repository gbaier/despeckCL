#!/usr/bin/python3

import zipfile
import urllib.request

import numpy as np
import matplotlib.pyplot as plt
import gdal


from IPython import embed

url = 'https://earth.esa.int/documents/653194/658149/'

filename = 'AIRSAR_Flevoland'
dataname = 'FLEVOL.STK'

def extract_from_archive(filename, dataname):
    with zipfile.ZipFile(filename) as zf:
        zf.extract(dataname)

def stk_reader(stk_filename):
    """ see http://gdal.org/frmt_airsar.html for description """
    data = gdal.Open(stk_filename)
    data = data.ReadAsArray()
    covmat = np.empty((3, 3, *data.shape[1:]), dtype=np.complex64)
    covmat[0, 0] = data[0]
    covmat[0, 1] = data[1]
    covmat[1, 0] = data[1].conj()
    covmat[0, 2] = data[2]
    covmat[2, 0] = data[2].conj()
    covmat[1, 1] = data[3]
    covmat[1, 2] = data[4]
    covmat[2, 1] = data[4].conj()
    covmat[2, 2] = data[5]
    return covmat

try:
    covmat = stk_reader(dataname)
except FileNotFoundError:
    urllib.request.urlretrieve(url + filename, filename + '.zip')
    extract_from_archive(filename + '.zip', dataname)
    covmat = stk_reader(dataname)

def plot_covmats(covmats, fig):
    nrows = len(covmats)
    ncols = covmats[0].shape[0]
    ax = None
    for nr, covmat in enumerate(covmats):
        # diagonal
        diag = np.abs(np.diagonal(covmat)) + 0.000001
        rgb_comp = 20*np.log10(diag)
        rgb_comp_norm = rgb_comp - rgb_comp.min()
        rgb_comp_norm /= rgb_comp_norm.max()
        for nc in range(ncols):
            ax = fig.add_subplot(nrows, ncols, nr*ncols + nc+1, sharex=ax,
                    sharey=ax)
            ax.imshow(rgb_comp_norm[:, :, nc], vmin=0, vmax=1)
            ax.set_title('channel {}'.format(nc+1))

fig = plt.figure()
plot_covmats([covmat], fig)
plt.show()
