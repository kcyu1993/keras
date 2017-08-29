
import sys
import numpy as np
from third_party.openai.weightnorm import SGDWithWeightnorm

from kyu.models.resnet import ResNet50_o2, ResNet50_o1

from kyu.theano.general.finetune import get_tmp_weights_path

from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGeneratorAdvanced
import keras.backend as K

from kyu.models.vgg import VGG16_o1, VGG16_o2
from kyu.utils.example_engine import ExampleEngine
from kyu.utils.imagenet_utils import preprocess_image_for_imagenet
from kyu.datasets.xray import XrayLoader



# Absolute paths
XRAY_PATH = '/home/kyu/cvkyu/swisscom/data'

TARGET_SIZE = (224, 224)
RESCALE_SMALL = 256

BATCH_SIZE = 32
NB_EPOCH = 1000
VERBOSE = 1
SAVE_LOG = True
VALIDATION = True

# Xray Loader

xray_loader = XrayLoader()
gen = ImageDataGeneratorAdvanced(TARGET_SIZE, RESCALE_SMALL, True,
                                 horizontal_flip=True,
                                 preprocessing_function=preprocess_image_for_imagenet
                                 # channelwise_std_normalization=True
                                 )

train = xray_loader.generator('train', batch_size=BATCH_SIZE)
test = xray_loader.generator('test', batch_size=BATCH_SIZE)
# train = xray_loader.generator('train', batch_size=BATCH_SIZE, image_data_generator=gen)
# test = xray_loader.generator('test', batch_size=BATCH_SIZE, image_data_generator=gen)


def fit_model(model, load=False, save=True, title='swisscom', nb_epoch=10):

    # sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=False)
    sgd = SGDWithWeightnorm(lr=0.0001, decay=1e-6, momentum=0.9)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    engine = ExampleEngine(train, model, test,
                           load_weight=load, save_weight=save, save_log=SAVE_LOG,
                           lr_decay=True, early_stop=True, tensorboard=True,
                           batch_size=BATCH_SIZE, nb_epoch=nb_epoch, title=title, verbose=VERBOSE)

    if SAVE_LOG:
        sys.stdout = engine.stdout
    model.summary()
    engine.fit(batch_size=BATCH_SIZE, nb_epoch=nb_epoch, augmentation=False)
    # score = engine.model.evaluate(X_test, Y_test, verbose=0)

    # engine.plot_result('loss')
    engine.plot_result()
    # print('Test loss: {} \n Test accuracy: {}'.format(score[0], score[1]))
    if SAVE_LOG:
        sys.stdout = engine.stdout.close()


def runroutine1():
    """
    Baseline result: VGG16
    
    
    Returns
    -------

    """
    nb_finetune = 2
    nb_train = 100
    denses = [256,256,256,]
    K.clear_session()
    random_key = np.random.randint(1, 1000)
    sess = K.get_session()
    with sess.as_default():
        print("Fine tune process ")
        model = VGG16_o1(denses, nb_classes=2, input_shape=(224,224,3), load_weights=True, freeze_conv=True)
        # model = ResNet50_o1(denses, nb_classes=2, input_shape=(224,224,3), load_weights=True, freeze_conv=True)
        fit_model(model, nb_epoch=nb_finetune)
        model.save_weights(get_tmp_weights_path(model.name + str(random_key)))
    K.clear_session()
    sess2 = K.get_session()
    with sess2.as_default():
        print("Train proccess")
        model = VGG16_o1(denses, nb_classes=2, input_shape=(224,224,3), load_weights=True, freeze_conv=False)
        # model = ResNet50_o1(denses, nb_classes=2, input_shape=(224,224,3), load_weights=True, freeze_conv=False)
        model.load_weights(get_tmp_weights_path(model.name + str(random_key)))
        fit_model(model, nb_epoch=nb_train)


def runroutine2():
    """
    Baseline result: second order


    Returns
    -------

    """
    nb_finetune = 5
    nb_train = 100
    o2ts = [128,64,32,32]
    K.clear_session()
    random_key = np.random.randint(1, 1000)
    sess = K.get_session()
    with sess.as_default():
        print("Fine tune process ")
        model = VGG16_o2(o2ts, mode=1, nb_classes=2, input_shape=(224,224,3), cov_mode='pmean',
                         cov_branch_output=64, last_avg=False, freeze_conv=True, nb_branch=2, concat='concat',
                         last_conv_feature_maps=[512])
        fit_model(model, nb_epoch=nb_finetune)
        model.save_weights(get_tmp_weights_path(model.name + str(random_key)))
    K.clear_session()
    sess2 = K.get_session()
    with sess2.as_default():
        print("Train proccess")
        model = VGG16_o2(o2ts, mode=1, nb_classes=2, input_shape=(224,224,3), cov_mode='pmean',
                         cov_branch_output=64, last_avg=False, freeze_conv=False, nb_branch=2, concat='concat',
                         last_conv_feature_maps=[512])
        model.load_weights(get_tmp_weights_path(model.name + str(random_key)))
        fit_model(model, nb_epoch=nb_train)


def run_resnet():
    """
    Baseline result: second order


    Returns
    -------

    """
    nb_finetune = 4
    nb_train = 100
    o2ts = [256,128,64,64]
    K.clear_session()
    random_key = np.random.randint(1, 1000)
    sess = K.get_session()
    with sess.as_default():
        print("Fine tune process ")
        model = ResNet50_o2(o2ts, mode=1, nb_classes=2, input_shape=(224,224,3), cov_mode='pmean',
                            cov_branch='o2t_no_wv',
                            cov_branch_output=64, last_avg=False, freeze_conv=True, nb_branch=2, concat='concat',
                            last_conv_feature_maps=[512])
        fit_model(model, nb_epoch=nb_finetune)
        model.save_weights(get_tmp_weights_path(model.name + str(random_key)))
    K.clear_session()
    sess2 = K.get_session()
    with sess2.as_default():
        print("Train proccess")
        model = ResNet50_o2(o2ts, mode=1, nb_classes=2, input_shape=(224,224,3), cov_mode='pmean',
                            cov_branch='o2t_no_wv',
                            cov_branch_output=64, last_avg=False, freeze_conv=True, nb_branch=2, concat='concat',
                            last_conv_feature_maps=[512])
        model.load_weights(get_tmp_weights_path(model.name + str(random_key)))
        fit_model(model, nb_epoch=nb_train)


def test_loader():
    train = xray_loader.generator('train', batch_size=BATCH_SIZE,
                                  image_data_generator=gen,
                                  save_to_dir='/home/kyu/cvkyu/plots',
                                  save_prefix='xray', save_format='png')
    train.next()

if __name__ == '__main__':
    runroutine1()
    # runroutine2()
    # run_resnet()
    # runroutine1()
    # test_loader()