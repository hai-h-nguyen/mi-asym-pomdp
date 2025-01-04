from .rnn_vanilla import RNN, LSTM, GRU
from .gpt2_vanilla import GPT2
from .mlp_vanilla import MLP


SEQ_MODELS = {RNN.name: RNN, LSTM.name: LSTM, GRU.name: GRU, MLP.name: MLP,
              GPT2.name: GPT2}
