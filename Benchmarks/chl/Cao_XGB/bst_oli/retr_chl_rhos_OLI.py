# -*- coding: utf-8 -*-
"""Description:
This script was used to call the BST model trained by Cao et al. (2020)
to estimate  chlorophyll-a concentrations in the lakes
from Landsat-8 rayleigh-corrected reflectance.

Some dependencies are required for running this script.
This script was examined in MacOS system(Catalina 10.15.6) and python 3.8.3.,
and just provides a reference to call use BST model, users can freely improve it.

Before using this script, users should be prepared an Rrc dataset in HDF-5 format
and corresponding water mask file (0-land, 1-water). If you do not have
a water mask, this script would use a threshold of Rrc(2201)<0.018
to get water pixels.

If you would like to learn more about the details of this model, please read
this paper: Cao et al., 2020, Remote sensing of environment,
A machine learning approach to estimate chlorophyll-a from Landsat-8
measurements in inland lakes

Authors: Zhigang Cao
Affiliate: Nanjing Institute of Geography and Limnology, 
Chinese Academy of Sciences
"""
import os
import time
import matplotlib
import numpy as np
import pylab as plt
import xgboost as xgb
from netCDF4 import Dataset


# using this option to release the memory in plotting
matplotlib.use('agg')

def nc_write(ncfile, dataset, data, wavelength=None, global_dims=None,
             new=False, attributes=None, keep=True, offset=None,
             replace_nan=False, metadata=None, 
             dataset_attributes=None, double=True,
             chunking=True, chunk_tiles=[10, 10], 
             chunksizes=None,
             format='NETCDF4', 
             nc_compression=True
             ):
    # This function was adapted from ACOLITE by Quiten V
    # https://github.com/acolite/acolite
    from numpy import isnan, nan, float32, float64
    from math import ceil

    if os.path.exists(os.path.dirname(ncfile)) is False:
        os.makedirs(os.path.dirname(ncfile))

    dims = data.shape

    if global_dims is None:
        global_dims = dims

    if chunking:
        if chunksizes is not None:
            chunksizes = (ceil(dims[0]/chunk_tiles[0]),
                          ceil(dims[1]/chunk_tiles[1]))

    if new:
        if os.path.exists(ncfile):
            os.remove(ncfile)
        nc = Dataset(ncfile, 'w', format=format)

        # set global attributes
        setattr(nc, 'data source', 'SeaDAS L2 products')
        setattr(nc, 'Algorithm', ' xgboost algorithm')
        setattr(nc, 'generated_on', time.strftime('%Y-%m-%d %H:%M:%S'))
        setattr(nc, 'contact', 'Zhigang CAO')
        setattr(nc, 'product_type', 'NetCDF4')
        setattr(nc, 'Institute', 'NIGLAS, CAS')
        setattr(nc, 'version', '0.1 beta')

        if attributes is not None:
            for key in attributes.keys():
                if attributes[key] is not None:
                    try:
                        setattr(nc, key, attributes[key])
                    except:
                        print('Failed to write attribute: {}'.format(key))

        # set up x and y dimensions
        nc.createDimension('x', global_dims[1])
        nc.createDimension('y', global_dims[0])
    else:
        nc = Dataset(ncfile, 'a', format=format)

    if (not double) & (data.dtype == float64):
        data = data.astype(float32)

    # write data
    if dataset in nc.variables.keys():
        # dataset already in NC file
        if offset is None:
            if data.dtype in (float32, float64):
                nc.variables[dataset][:] = nan
            nc.variables[dataset][:] = data
        else:
            if replace_nan:
                tmp = nc.variables[dataset][offset[1]:offset[1]+dims[0], offset[0]:offset[0]+dims[1]]
                sub_isnan = isnan(tmp)
                tmp[sub_isnan] = data[sub_isnan]
                nc.variables[dataset][offset[1]:offset[1] +
                                      dims[0], offset[0]:offset[0]+dims[1]] = tmp
                tmp = None
            else:
                nc.variables[dataset][offset[1]:offset[1] +
                                      dims[0], offset[0]:offset[0]+dims[1]] = data
    else:
        # new dataset
        var = nc.createVariable(dataset, data.dtype, ('y', 'x'), zlib=nc_compression,
                                chunksizes=chunksizes, fill_value=-32767)
        if wavelength is not None:
            setattr(var, 'wavelength', float(wavelength))
        # set attributes
        if dataset_attributes is not None:
            for att in dataset_attributes.keys():
                setattr(var, att, dataset_attributes[att])

        if offset is None:
            if data.dtype in (float32, float64):
                var[:] = nan
            var[:] = data
        else:
            if data.dtype in (float32, float64):
                var[:] = nan
            var[offset[1]:offset[1]+dims[0], offset[0]:offset[0]+dims[1]] = data
    if keep is not True:
        data = None
    # close netcdf file
    nc.close()
    nc = None


def read_mask(filename):
    # read a water mask generated in matlab using OSTU method.
    import h5py
    h5file = h5py.File(filename, 'r')
    mask_data = np.array(h5file['water_mask'][:], dtype=np.int8)
    h5file.close()
    print('>>> {} readed...'.format(filename))
    return mask_data


def read_img_data(filename):
    """To read a netcdf file and return a dictionary"""
    import h5py
    h5file = h5py.File(filename, 'r')
    RRC_443 = np.array(
        h5file['geophysical_data/rhos_443'][:], dtype=np.float32)
    RRC_482 = np.array(
        h5file['geophysical_data/rhos_482'][:], dtype=np.float32)
    RRC_561 = np.array(
        h5file['geophysical_data/rhos_561'][:], dtype=np.float32)
    RRC_655 = np.array(
        h5file['geophysical_data/rhos_655'][:], dtype=np.float32)
    RRC_865 = np.array(
        h5file['geophysical_data/rhos_865'][:], dtype=np.float32)
    RRC_1609 = np.array(
        h5file['geophysical_data/rhos_1609'][:], dtype=np.float32)
    RRC_2201 = np.array(
        h5file['geophysical_data/rhos_2201'][:], dtype=np.float32)
    lat = np.array(h5file['navigation_data/latitude'][:], dtype=np.float32)
    lon = np.array(h5file['navigation_data/longitude'][:], dtype=np.float32)
    all_data = {'RRC': np.array([RRC_443, RRC_482, RRC_561, RRC_655, RRC_865, RRC_1609, RRC_2201]),
                'LAT': lat, 'LON': lon}
    h5file.close()
    print('>>> {} readed...'.format(filename))
    return all_data


def apply_model_pixel(bst, rrc, water_mask):
    """calculate pixel by pixel
       using the water_mask to filter the land pixel and only calculate the water pixel
       this is a strategy to save time"""
    print('>>> chlora started at :'+time.strftime('%Y-%m-%d %H:%M:%S'))
    line, col = rrc.shape[1], rrc.shape[2]
    # make a coarse atmospheric correction
    rrc = rrc[0:6, :, :] - rrc[6, :, :]    
    # generate the band ratios and FAI
    bg = rrc[0, :, :] / rrc[2, :, :]
    rg = rrc[3, :, :] / rrc[1, :, :]
    nr = rrc[4, :, :] / rrc[3, :, :]
    FAI = rrc[4, :, :] - (rrc[3, :, :] + (rrc[5, :, :]-rrc[3, :, :])
                        * (865.0-655.0)/(1609.0-655.0))
    rrc = np.array([rrc[0, :, :], rrc[1, :, :], rrc[2, :, :],
                    rrc[3, :, :], rrc[4, :, :], rrc[5, :, :], bg, rg, nr, FAI])
    chlora = np.full((line, col), np.nan)
    # only proecss the water pixels
    idx = np.where(water_mask == 1)
    line_idx = idx[0]
    col_idx = idx[1]
    n_rows = len(line_idx)
    interval = int(n_rows / 20.0)
    if interval == 0: interval = 1
    for i in range(n_rows):
        row, col = line_idx[i], col_idx[i]
        dtrain = xgb.DMatrix(np.array([rrc[:, row, col], ]))
        chlora[row, col] = bst.predict(dtrain)
        if i % interval == 0:
            print('Scans %d in %d => %d processed.' % (i, n_rows, i/n_rows * 100))
    # remove the minus value
    chlora[chlora < 0] = np.nan
    print('>>> chlora ended at :' + time.strftime('%Y-%m-%d %H:%M:%S'))
    return chlora


def output_retrieval(l2r_ncfile, lat, lon, chlora):
    """Write the Chla, Lat, Lon to a nc file with a compression netcdf4 format"""
    l2r_nc_new = True
    nc_write(l2r_ncfile, 'LAT', lat, new=l2r_nc_new,
             dataset_attributes={'long_name': 'latitude', 'units': 'degree + north'})
    l2r_nc_new = False
    nc_write(l2r_ncfile, 'LON', lon, new=l2r_nc_new,
             dataset_attributes={'long_name': 'longitude', 'units': 'degree +east'})
    nc_write(l2r_ncfile, 'chlora', chlora, new=l2r_nc_new,
             dataset_attributes={'long_name': 'chlorophll-a', 'valid_min': 1.0,
                                 'valid_max': 100.0, 'units': 'mg m^-3'})
    print('	+++ Chla written in ' + l2r_ncfile)


def plot_chla(chlora, vmin, vmax, outname):
    """Plot and save the image of Chla with a jet scratch color bar
        chlora: np.array of chla retrievals
        vmin and vmax: ranges used for presentation
        outname: outfile name of a RGB image.
    """
    # set negtive chl pixels to be transparent
    fig = plt.figure()
    canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(fig)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # chlora[np.isnan(chlora)] = -32767
    cmap = plt.cm.jet
    im = ax.imshow(chlora, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, orientation='vertical', label='Chl-a ($\mu$g/L)',
                 shrink=0.65, aspect=15)
    # ax.axis('off')
    canvas.print_figure(outname, dpi=300)
    print('>>> chlora png generated at :' + time.strftime('%Y-%m-%d %H:%M:%S'))


"""Main program to estimate the chlrophyll-a from Landsat-8 OLI Rrc data"""
if __name__ == '__main__': 
    model_path = 'chl_bst_model_release.model'
    assert os.path.exists(model_path), 'Please check path of BST model!'
    # load BST model
    np.seterr(divide='ignore', invalid='ignore')
    bst = xgb.Booster({'nthread': 4})  # init model
    bst.load_model(model_path)
    # define the input and output directory
    l2_file = 'LC08_L1TP_121038_20151011_20170403_01_T1_Chaohu_seadas_l2.h5'
    outputdir = './'
    # generate the output filename
    chl_file = outputdir + os.path.sep + \
        '_'.join(os.path.basename(l2_file).split('_')[0:5]) + '_chla.nc'
    if os.path.exists(chl_file):
        print('{} existed, skip.'.format(chl_file))
    else:
        # read Rrc data
        img_data = read_img_data(l2_file)
        RRC = img_data['RRC']
        lat = img_data['LAT']
        lon = img_data['LON']
        # find the corresponding water mask file
        water_maskfile = os.path.splitext(l2_file)[0] + '_waterMask.h5'
        if not os.path.exists(water_maskfile):
            water_mask = np.zeros(lat.shape, dtype=np.float)
            # Rrc(2201) > 0.018 as clouds or lands
            water_mask[RRC[6, :, :] < 0.018] = 1
        else:
            water_mask = read_mask(water_maskfile)
        # apply the model to retrieve
        chlora = apply_model_pixel(bst, RRC, water_mask)
        # writing data to a nc file
        output_retrieval(chl_file, lat, lon, chlora)
        print('>>> ' + chl_file + ' exported!')
    # plot the chl image
    l2r_pltfile = os.path.splitext(chl_file)[0] + '.png'
    if not os.path.exists(l2r_pltfile):
        nc = Dataset(chl_file, 'r')
        chlora = nc['chlora'][:]
        nc.close()
        nc = None
        plot_chla(chlora, 0, 80, l2r_pltfile)
        print('>>> ' + l2r_pltfile + ' exported!')
