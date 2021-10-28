from .resnet50_lstm import ResNet50LSTM
from .efficientnetb4_lstm import EfficientNetB4LSTM


models = {
    'r50_lstm': ResNet50LSTM,
    'eb4_lstm': EfficientNetB4LSTM
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

