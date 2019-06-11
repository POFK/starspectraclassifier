# coding: utf-8
import os
import time
import numpy as np
from scipy import interpolate
from loadspec import rdspec

Dir = '/nfs/P100/SDSSV_Classifiers/data/optical'
OutDir = '/nfs/P100/SDSSV_Classifiers/processed_dataset'
name_type = {'wdsb2': 0, 'wd': 1, 'yso': 2, 'hotstars': 3, 'fgkm': 4, 'cv': 5}
frac = np.array([0.8, 0.9, 1.0])
w_start = 3900.
w_end = 9000.
w_len = 3000
wavenew = np.linspace(w_start, w_end, w_len / 1)
dt = np.dtype([('index', 'i4'), ('label', 'i4'),
               ('flux_norm', np.float32, w_len)])


def mask(func):
    def wrap(*args, **kwargs):
        spec = func(*args, **kwargs)
        bool_m = np.logical_not(np.isnan(spec.err) + np.isnan(spec.flux))
        spec.flux = spec.flux[bool_m]
        spec.err = spec.err[bool_m]
        spec.wave = spec.wave[bool_m]
        return spec

    return wrap


class DATA(object):
    """data clean. """

    def __init__(self, dir=Dir, outdir=OutDir):
        """TODO: Docstring for __init__.
        :dir: TODO
        :outdir: TODO
        :returns: TODO
        """
        self.Test = False

    def checkdir(self, dir):
        """TODO: Docstring for checkdir.
        :dir: TODO
        :returns: TODO
        """
        isExists = os.path.exists(dir)
        if not isExists:
            os.makedirs(dir)

    def getfilename(self, dir=Dir):
        """TODO: Docstring for getfilename.
        :dir: data path
        :returns: fndict, file name dict
        """
        fndict = {}
        sptypes = os.listdir(dir)
        print sptypes
        for i in xrange(len(sptypes) - 1, -1, -1):
            if not os.path.isdir(os.path.join(dir, sptypes[i])):
                sptypes.remove(sptypes[i])
        print sptypes
        for sptype in sptypes:
            fnlist = os.listdir(os.path.join(dir, sptype))
            n1 = len(fnlist)
            for i in xrange(len(fnlist) - 1, -1, -1):
                if fnlist[i][-5:] != '.fits' and fnlist[i][-4:] != '.fit':
                    fnlist.remove(fnlist[i])
            n2 = len(fnlist)
            fndict[sptype] = {
                'len': len(fnlist),
                'dir': os.path.join(dir, sptype),
                'fname': np.array(fnlist)
            }
            print sptype, n1, '-->', n2
        return fndict

    def seltrainset(self, fndict):
        """Select the training, valid and test set from filename class
        :fndict: filename class, outputed by "getfilename" function
        :returns: fndict, add training set filename in fndict
        """
        for i in name_type.keys():
            size = fndict[i]['len']
            random_ind = np.random.permutation(np.arange(size))
            ind_dataset = np.ceil(frac * size).astype(np.int32)
            fndict[i]['train'] = fndict[i]['fname'][
                random_ind[:ind_dataset[0]]]
            fndict[i]['valid'] = fndict[i]['fname'][
                random_ind[ind_dataset[0]:ind_dataset[1]]]
            fndict[i]['test'] = fndict[i]['fname'][
                random_ind[ind_dataset[1]:ind_dataset[2]]]
            fndict[i]['len_train'] = len(fndict[i]['train'])
            fndict[i]['len_valid'] = len(fndict[i]['valid'])
            fndict[i]['len_test'] = len(fndict[i]['test'])
            assert fndict[i]['len_train'] + fndict[i]['len_valid'] + fndict[i][
                'len_test'] == fndict[i]['len']
            savepath = os.path.join(OutDir, 'filename')
            self.checkdir(savepath)
            np.savetxt(
                os.path.join(savepath, i + '_train.txt'),
                fndict[i]['train'],
                fmt='%s')
            np.savetxt(
                os.path.join(savepath, i + '_valid.txt'),
                fndict[i]['valid'],
                fmt='%s')
            np.savetxt(
                os.path.join(savepath, i + '_test.txt'),
                fndict[i]['test'],
                fmt='%s')
        return fndict

    @mask
    def read(self, fname):
        """TODO: Docstring for read.

        :fname: TODO
        :returns: TODO

        """
        data = rdspec(fname)
        return data

    def loaddata(self, fndict, fmt='train'):
        """TODO: Docstring for loaddata.

        :fndict: TODO
        :fmt: 'train', 'valid' or 'test'
        :returns: TODO

        """
        data = {}
        for key in name_type.keys():
            print key, name_type[key]
            sub = {}
            sptype = key
            for i in xrange(fndict[key]['len_' + fmt]):
                if self.Test:
                    if i >= 10:  # for test
                        break
                ind_f = i
                name = '%s_%d' % (key, i)
                spec = self.read(
                    os.path.join(fndict[key]['dir'], fndict[key][fmt][ind_f]))
                sub[name] = {
                    'index': ind_f,
                    'sptype': key,
                    'label': name_type[key],
                    'wave': spec.wave,
                    'flux': spec.flux,
                    #   'err': spec.err
                }
                data[key] = sub
        #       print name, data[name]['index'], data[name]['label'], data[name]['sptype']
        return data

    def interpolate_flux(self, wavelen, flux, wavenew):
        """TODO: Docstring for interpolate_flux.

        :wavelen: TODO
        :flux: TODO
        :wavenew: TODO
        :returns: TODO

        """
        f_linear = interpolate.interp1d(wavelen, flux)
        return f_linear(wavenew)

    def smooth4test(self, wavenew, data, sigma=200):
        """This function is used to interpolate and smooth the flux spectrum. 
        Unlike the "smooth" function used to create the training set, it is simplified for easier usage.

        :sigma: 20nm
        :wavenew: TODO
        :data: TODO
        :returns: TODO

        """
        step = wavenew[1] - wavenew[0]
        ones = np.ones_like(wavenew, dtype=np.float64)
        dis = np.arange(int(sigma / step) * 3 + 1, dtype=np.float64)
        dis -= dis[-1] / 2.
        dis *= step
        kernel = np.exp(-0.5 * (dis / sigma)**2.)
        w_conv = np.convolve(ones, kernel, mode='same')

        data_arr_len = len(data)
        data_arr = np.empty(data_arr_len, dtype=dt)

        for i in range(len(data)):
            spec = data[i]
            try:
                flux_ip = self.interpolate_flux(spec['wave'],
                                                spec['flux'],
                                                wavenew)
            except ValueError, err:
                print 'mask:', i, spec['wave'].min(), spec['wave'].max()
                data_arr[i]['label'] = -1
                continue
            flux_s = np.convolve(flux_ip, kernel, mode='same') / w_conv

            data_arr[i]['flux_norm'] = (flux_ip / flux_s).astype(
                np.float32)
            data_arr[i]['label'] = 0
            data_arr[i]['index'] = i
        bool_mask = data_arr['label'] != -1
        data_arr = data_arr[bool_mask]
        print 'mask %d elements from %d spectra...' % (
            data_arr_len - bool_mask.sum(), data_arr_len)
        return data_arr


    def smooth(self, wavenew, data, sigma=200):
        """TODO: Docstring for smooth.

        :sigma: 20nm
        :wavenew: TODO
        :data: TODO
        :returns: TODO

        """
        step = wavenew[1] - wavenew[0]
        ones = np.ones_like(wavenew, dtype=np.float64)
        dis = np.arange(int(sigma / step) * 3 + 1, dtype=np.float64)
        dis -= dis[-1] / 2.
        dis *= step
        kernel = np.exp(-0.5 * (dis / sigma)**2.)
        w_conv = np.convolve(ones, kernel, mode='same')

        dataset = {}
        for key in name_type.keys():
            data_arr_len = len(data[key])
            data_arr = np.empty(data_arr_len, dtype=dt)
            for name in data[key].iterkeys():
                index = data[key][name]['index']
                try:
                    flux_ip = self.interpolate_flux(data[key][name]['wave'],
                                                    data[key][name]['flux'],
                                                    wavenew)
                except ValueError, err:
                    print 'mask:', key, name, index, data[key][name][
                        'wave'].min(), data[key][name]['wave'].max()
                    data_arr[index]['label'] = -1
                    continue
                flux_s = np.convolve(flux_ip, kernel, mode='same') / w_conv
                #               data[key][name]['flux'] = flux_ip.astype(np.float32)
                #               data[key][name]['flux_s'] = flux_s.astype(np.float32)
                #               data[key][name]['flux_norm'] = (flux_ip / flux_s).astype(np.float32)
                data_arr[index]['flux_norm'] = (flux_ip / flux_s).astype(
                    np.float32)
                data_arr[index]['index'] = index
                data_arr[index]['label'] = name_type[key]
            bool_mask = data_arr['label'] != -1
            data_arr = data_arr[bool_mask]
            print 'mask %d elements from %d %s spectra...' % (
                data_arr_len - bool_mask.sum(), data_arr_len, key)
            dataset[key] = data_arr
        return dataset

    def get_musigma(self, dataset):
        """TODO: Docstring for get_musigma.

        :dataset: dict, result of function `smooth`
        :returns: TODO

        """
        data = []
        for key in name_type.keys():
            data.append(dataset[key]['flux_norm'])
        data = np.vstack(data)
        self.mu = data.mean(axis=0)
        self.sigma = data.std(axis=0)

    def oversampling(self, subset, num):
        """oversampling for the training set.

        :dataset: TODO
        :num: TODO
        :returns: TODO

        """
        shape = len(subset)
        assert num > shape
        num = num - shape
        randomindex = np.random.randint(0, shape, num)
        data = subset[randomindex]
        data = np.hstack([subset, data])
        return data

    def save(self, path, subdataset):
        """TODO: Docstring for save.

        :path: such as training.npy
        :dataset: TODO
        :returns: TODO

        """
        DIR = os.path.join(OutDir, 'dataset')
        np.save(os.path.join(DIR, path), subdataset)

    def __call__(self, fmt='train', sigma=200):
        """TODO: Docstring for __call__
        :returns: TODO

        """
        print '=' * 80
        print fmt
        print '=' * 80
        fndict = self.getfilename()
        fndict = self.seltrainset(fndict)
        num_max = []
        data = self.loaddata(fndict, fmt=fmt)
        data_set = self.smooth(wavenew, data, sigma=sigma)

        for key in name_type.keys():
            num_max.append(data_set[key].shape[0])
        num_max = np.max(num_max)
        for key in name_type.keys():
            if fmt == 'train':
                self.get_musigma(data_set)
                np.save(os.path.join(OutDir, 'wavelen.npy'), wavenew)
                np.save(os.path.join(OutDir, 'Norm_mu.npy'), self.mu)
                np.save(os.path.join(OutDir, 'Norm_std.npy'), self.sigma)
                print fmt, key, num_max, len(
                    data_set[key]), fndict[key]['len_train']
                if data_set[key].shape[0] < num_max:
                    if self.Test:
                        subdataset = self.oversampling(data_set[key], 20)
                    else:
                        subdataset = self.oversampling(data_set[key], num_max)
                    self.save(fmt + '_' + key, subdataset)
                else:
                    self.save(fmt + '_' + key, data_set[key])
            else:
                self.save(fmt + '_' + key, data_set[key])


if __name__ == '__main__':
    da = DATA()
    #   da.Test = True
    da(fmt='train', sigma=200)
    da(fmt='valid', sigma=200)
    da(fmt='test', sigma=200)
