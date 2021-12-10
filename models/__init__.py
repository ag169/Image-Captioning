from .resnet50_lstm import ResNet50LSTM
from .efficientnetb4_lstm import EfficientNetB4LSTM
from .resnext50_lstm import ResNeXT50LSTM
from .resnext50_gru import ResNeXT50GRU
from .resnet50_lstm_attention import ResNet50LSTMAttention
from .resnet50_lstm_attention_v2 import ResNet50LSTMAttention as ResNet50LSTMAttention2
from .resnext_50_gru_attention_v2 import ResNeXT50GRUAttention
from .mnv2_gru_attention_v2 import MobilenetV2GRUAttention
from .efficientnetb4_gru_attention_v2 import EfficientNetB4GRUAttention
from .efficientnetb4_gru import EfficientNetB4GRU

models = {
    'r50_lstm': ResNet50LSTM,
    'eb4_lstm': EfficientNetB4LSTM,
    'gru': EfficientNetB4GRU,
    'rnxt50_lstm': ResNeXT50LSTM,
    'rnxt50_gru': ResNeXT50GRU,
    'r50_lstm_attn': ResNet50LSTMAttention,
    'r50_lstm_attn2': ResNet50LSTMAttention2,
    'rnxt50_gru_attn2': ResNeXT50GRUAttention,
    'mnv2_gru_attn2': MobilenetV2GRUAttention,
    'eb4_gru_attn2': EfficientNetB4GRUAttention,
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

