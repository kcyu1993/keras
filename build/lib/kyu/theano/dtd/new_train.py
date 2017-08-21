"""
New Train pipeline with trainer

"""
import copy

from kyu.datasets.dtd import DTD
from kyu.engine.configs import RunningConfig
from kyu.engine.trainer import ClassificationTrainer
from kyu.models import get_model
from kyu.theano.dtd.configs import get_o2t_testing
from kyu.utils.io_utils import ProjectFile


def get_running_config(title='DTD_testing', model_config=None):
    return RunningConfig(
        _title=title,
        nb_epoch=100,
        batch_size=32,
        verbose=2,
        lr_decay=True,
        sequence=8,
        patience=8,
        early_stop=True,
        save_weights=True,
        load_weights=False,
        init_weights_location=None,
        save_per_epoch=True,
        tensorboard=None,
        optimizer='SGD',
        lr=0.01,
        model_config=model_config,
    )


def finetune_with_model(model_config, nb_epoch_finetune, running_config):
    data = DTD('/home/kyu/.keras/datasets/dtd', name='DTD')
    dirhelper = ProjectFile(root_path='/home/kyu/cvkyu/secondstat', dataset=data.name, model_category='VGG16')
    dirhelper.build(running_config.title)

    if nb_epoch_finetune > 0:
        # model_config2 = copy.copy(model_config)
        model_config.freeze_conv = True
        model = get_model(model_config)

        trainer = ClassificationTrainer(model, data, dirhelper,
                                        model_config=model_config, running_config=running_config,
                                        save_log=True,
                                        logfile=dirhelper.get_log_path())
        trainer.fit()
        trainer.plot_result()
        # trainer.plot_model()
        model_config.freeze_conv = False
        running_config.load_weights = True
        running_config.init_weights_location = dirhelper.get_weight_path()

    model = get_model(model_config)

    trainer = ClassificationTrainer(model, data, dirhelper,
                                    model_config=model_config, running_config=running_config,
                                    save_log=True,
                                    logfile=dirhelper.get_log_path())
    trainer.build()
    trainer.model.summary()
    trainer.fit()
    trainer.plot_result()



if __name__ == '__main__':
    title = 'DTD-Testing-o2transform'
    model_config = get_o2t_testing(1)
    running_config = get_running_config(title, model_config)

    finetune_with_model(model_config, 2, running_config)


