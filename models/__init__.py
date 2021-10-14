from .resnet50_lstm import ResNet50LSTM


models = {
    'r50_lstm': ResNet50LSTM,
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

