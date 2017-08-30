"""
Inference

"""
import argparse

from kyu.configs.experiment_configs.first_order import get_fo_vgg_exp

from kyu.configs.experiment_configs.running_configs import get_running_config_no_debug_withSGD
from kyu.configs.experiment_configs.simple_second_order_config import get_single_o2transform
from kyu.datasets.dtd import DTD
from kyu.datasets.minc import Minc2500_v2
from kyu.engine.trainer import ClassificationTrainer
from kyu.models import get_model
from kyu.utils.image import get_vgg_image_gen, get_resnet_image_gen
from kyu.utils.io_utils import ProjectFile

TEST_PATH = '/home/kyu/cvkyu/secondstat/output/run/cls/Minc2500/vgg/' \
            'SO-VGG_originalOriginal-O2T-comparing_regularizer_l1_l2-branch2_2017-08-29T14:45:55/' \
            'SO-VGG_originalOriginal-O2T-comparing_regularizer_l1_l2-branch2_2017-08-29T14:45:55.weights'

TEST_PATH_2 = '/home/kyu/cvkyu/secondstat/output/run/cls/Minc2500/vgg/SO-VGG original ' \
              'minc2500_2017-08-25T16:06:26/SO-VGG original minc2500_2017-08-25T16:06:26.weights'

def get_dirhelper(dataset_name, model_category, **kwargs):
    return ProjectFile(root_path='/home/kyu/cvkyu/test_inference', dataset=dataset_name, model_category=model_category,
                       **kwargs)


def get_argparser(description='inference'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--dataset', type=str, required=True, help='dataset name: support dtd, minc2500')
    parser.add_argument('-m', '--model_class', help='model class should be in vgg, resnet', default='vgg', type=str)
    parser.add_argument('-me', '--model_exp', help='model experiment', default=1, type=int)
    parser.add_argument('-dbg', '--debug', type=bool, help='True for entering TFDbg mode', default=False)
    parser.add_argument('-c', '--comments', help='comments if any', default='', type=str)
    parser.add_argument('-p', '--weight_path', help='weight path of the model', default=None, type=str, required=True)
    return parser


def cnn_inference(dataset, model_class, model_exp_fn, model_exp, title='',
                  weight_path=None,
                  comments='', debug=False, tensorboard=False):

    model_config = model_exp_fn(model_exp)
    model_config.class_id = model_class
    running_config = get_running_config_no_debug_withSGD(
        title=title,
        model_config=model_config
    )
    if debug:
        running_config.tf_debug = True
    running_config.tensorboard = tensorboard
    running_config.comments = comments

    if dataset == 'dtd':
        data = DTD('/home/kyu/.keras/datasets/dtd', name='DTD')
    elif dataset == 'minc2500' or dataset == 'minc-2500':
        data = Minc2500_v2('/home/kyu/.keras/datasets/minc-2500', image_dir=None)
    else:
        raise NotImplementedError
    dirhelper = get_dirhelper(data.name, model_config.class_id)

    inference_with_model_data(data, model_config, dirhelper, weight_path, running_config)


def inference_with_model_data(data, model_config, dirhelper, weight_path, running_config):
    """
    Inference pipeline

    Parameters
    ----------
    data
    model_config
    dirhelper
    weight_path
    running_config

    Returns
    -------

    """

    model_config.nb_class = data.nb_class
    if model_config.class_id == 'vgg':
        data.image_data_generator = get_vgg_image_gen(model_config.target_size,
                                                      running_config.rescale_small,
                                                      running_config.random_crop,
                                                      running_config.horizontal_flip)
    else:
        data.image_data_generator = get_resnet_image_gen(model_config.target_size,
                                                         running_config.rescale_small,
                                                         running_config.random_crop,
                                                         running_config.horizontal_flip)
    dirhelper.build(running_config.title + model_config.name)

    model = get_model(model_config)
    model.summary()
    model.compile('SGD', loss='categorical_crossentropy', metrics=['acc'])
    print("Load weights from {}".format(weight_path))
    model.load_weights(weight_path)

    test_data = data.get_test()
    history = model.evaluate_generator(test_data, steps=test_data.n / running_config.batch_size)
    print("loss {} acc {}".format(history[0], history[1]))
    # trainer = ClassificationTrainer(model, data, dirhelper,
    #                                 model_config=model_config, running_config=running_config,
    #                                 save_log=True,
    #                                 logfile=dirhelper.get_log_path())
    #
    # trainer.model.summary()
    # # trainer.plot_model()
    # model_config.freeze_conv = False
    # running_config.load_weights = True
    # running_config.init_weights_location = dirhelper.get_weight_path()


if __name__ == '__main__':
    parser = get_argparser('Inference general pipeline')
    # parser.add_argument('-mef', '--model_exp_fn', help='model config function', type=str)

    args = parser.parse_args()
    # Only  support VGG FO
    model_exp_fn = get_single_o2transform

    cnn_inference(model_exp_fn=model_exp_fn, **vars(args))
