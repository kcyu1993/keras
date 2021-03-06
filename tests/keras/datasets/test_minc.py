import pytest

from keras.datasets.minc import *
import datetime


# Test minc 2500
def save_data(type='hdf5'):
    loader = Minc2500()
    print("Loading the data")
    tstart = datetime.datetime.now()
    data = loader.loadwithsplit()
    tend = datetime.datetime.now()
    print('loads split from raw image file within {}s'.format((tend - tstart).total_seconds()))
    print('saving data {}'.format(type))
    tstart = datetime.datetime.now()
    loader.save(data, output_file='minc-2500.plk', ftype=type)
    tend = datetime.datetime.now()
    print('saved within {}s'.format((tend - tstart).total_seconds()))


def test_load():
    # loader = Minc2500()
    # print("Loading the data")
    #
    # tstart = datetime.datetime.now()
    # data = loader.loadwithsplit()
    # tend = datetime.datetime.now()
    # print('loads split from raw image file within {}s'.format((tend - tstart).total_seconds()))
    # print('saving data ')
    # tstart = datetime.datetime.now()
    # loader.save(data, ftype='hdf5')
    # tend = datetime.datetime.now()
    # print('saved within {}s'.format((tend - tstart).total_seconds()))

    loader2 = Minc2500()

    tstart = datetime.datetime.now()
    data = loader2.loadfromfile(filename='minc-2500.plk.hdf5')
    # print(data)
    tend = datetime.datetime.now()
    print('loads split from hdf5 file within {}s'.format((tend - tstart).total_seconds()))
    for d in data:
        print(d.shape)
    # print('loads from saved pickle')
    # tstart = datetime.datetime.now()
    # loader2.loadfromfile("minc-2500.plk.gz")
    # tend = datetime.datetime.now()
    # print('loads split from gzip pickle file within {}s'.format((tend - tstart).total_seconds()))
    #
    # print('loads from saved raw pickle')
    # tstart = datetime.datetime.now()
    # loader2.loadfromfile("minc-2500.plk")
    # tend = datetime.datetime.now()
    # print('loads split from gzip pickle file within {}s'.format((tend - tstart).total_seconds()))

def test_hdf5():
    loader = Minc2500()
    loader.test_hd5f()
    loader.test_hd5f_load()


# Test Minc Original
def test_minc_original_loader():
    save_dir = os.path.join(get_dataset_dir(), 'debug')
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    loader = MincOriginal()
    gen = loader.generator(save_dir=save_dir)
    a = gen.next()


if __name__ == '__main__':
    # save_data('gz')
    pytest.main([__file__])





