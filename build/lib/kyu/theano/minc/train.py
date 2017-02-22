from keras.utils.data_utils import get_absolute_dir_project
from kyu.datasets.minc import Minc2500
from kyu.utils.example_engine import ExampleEngine

LOG_PATH = get_absolute_dir_project('model_saved/log/minc2500')
# global constants
NB_CLASS = 23         # number of classes
LEARNING_RATE = 0.01
MOMENTUM = 0.9
BATCH_SIZE = 128
ALPHA = 0.0001
BETA = 0.75
GAMMA = 0.1
DROPOUT = 0.5
WEIGHT_DECAY = 0.0005
NB_EPOCH = 20
LRN2D_norm = True       # whether to use batch normalization
# Theano - 'th' (channels, width, height)
# Tensorflow - 'tf' (width, height, channels)
DIM_ORDERING = 'th'

TARGET_SIZE=(224,224)
### FOR model 1
if K.backend() == 'tensorflow':
    INPUT_SHAPE = TARGET_SIZE + (3,)
    K.set_image_dim_ordering('tf')
else:
    INPUT_SHAPE=(3,) + TARGET_SIZE
    K.set_image_dim_ordering('th')


def run_minc2500_model(model, title="", load_w=True, save_w=True, plot=False, verbose=2, imagegenerator=None):
    print("loading model from generator ")
    # tr, va, te = loads()
    loader = Minc2500()

    tr_iterator = loader.generator(input_file='train1.txt', batch_size=16, target_size=TARGET_SIZE, gen=imagegenerator)
    te_iterator = loader.generator(input_file='test1.txt', target_size=TARGET_SIZE, gen=imagegenerator)

    model.summary()
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    engine = ExampleEngine(data=tr_iterator, model=model, validation=te_iterator,
                           load_weight=load_w, save_weight=save_w, title=title)
    history = engine.fit(nb_epoch=NB_EPOCH)
    if plot:
        engine.plot_result()
