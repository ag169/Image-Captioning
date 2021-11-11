from .resnet50_lstm import ResNet50LSTM
from .efficientnetb4_lstm import EfficientNetB4LSTM
from .resnext50_lstm import ResNeXT50LSTM
from .resnext50_gru import ResNeXT50GRU

models = {
    'r50_lstm': ResNet50LSTM,
    'eb4_lstm': EfficientNetB4LSTM,
    'rnxt50_lstm': ResNeXT50LSTM,
    'rnxt50_gru': ResNeXT50GRU,
}


def get_model(model, params=None):
    try:
        model_class = models[model]
    except KeyError:
        print('Model not implemented')
        raise NotImplementedError

    if params:
        return model_class(**params)
    else:
        return model_class()

