from __future__ import absolute_import
from recurrentshop import LSTMCell, RecurrentSequential,RecurrentModel
from  seq2seq.cells import LSTMDecoderCell, AttentionDecoderCell
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, TimeDistributed, Bidirectional, Input, Activation


def AttentionSeq2Seq(output_dim, output_length, batch_input_shape=None,
                     batch_size=None, input_shape=None, input_length=None,
                     input_dim=None, hidden_dim=None, depth=1,
                     bidirectional=True, unroll=False, stateful=False, dropout=0.0,):
    '''
    This is an attention Seq2seq model based on [3].
    Here, there is a soft allignment between the input and output sequence elements.
    A bidirection encoder is used by default. There is no hidden state transfer in this
    model.
    The  math:
            Encoder:
            X = Input Sequence of length m.
            H = Bidirection_LSTM(X); Note that here the LSTM has return_sequences = True,
            so H is a sequence of vectors of length m.
            Decoder:
    y(i) = LSTM(s(i-1), y(i-1), v(i)); Where s is the hidden state of the LSTM (h and c)
    and v (called the context vector) is a weighted sum over H:
    v(i) =  sigma(j = 0 to m-1)  alpha(i, j) * H(j)
    The weight alpha[i, j] for each hj is computed as follows:
    energy = a(s(i-1), H(j))
    alpha = softmax(energy)
    Where a is a feed forward network.
    
    '''

    if isinstance(depth, int):
        depth = (depth, depth)
    if batch_input_shape:
        shape = batch_input_shape
    elif input_shape:
        shape = (batch_size,) + input_shape
    elif input_dim:
        if input_length:
            shape = (batch_size,) + (input_length,) + (input_dim,)
        else:
            shape = (batch_size,) + (None,) + (input_dim,)
    else:
        # TODO Proper error message
        raise TypeError
    if hidden_dim is None:
        hidden_dim = output_dim

    _input = Input(batch_shape=shape)
    _input._keras_history[0].supports_masking = True

    encoder = RecurrentSequential(unroll=unroll, stateful=stateful,
                                  return_sequences=True)
    encoder.add(LSTMCell(hidden_dim, batch_input_shape=(shape[0], shape[2])))

    for _ in range(1, depth[0]):
        encoder.add(Dropout(dropout))
        encoder.add(LSTMCell(hidden_dim))

    if bidirectional:
        encoder = Bidirectional(encoder, merge_mode='sum')
        encoder.forward_layer.build(shape)
        encoder.backward_layer.build(shape)
        # patch
        encoder.layer = encoder.forward_layer

    encoded = encoder(_input)
    
    decoder= RecurrentSequential(decode=True, output_length=output_length,
                                  unroll=unroll, stateful=stateful)
    decoder.add(Dropout(dropout, batch_input_shape=(shape[0], shape[1], hidden_dim)))
    if depth[1] == 1:
        decoder.add(AttentionDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim))
    else:
        decoder.add(AttentionDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim))
        for _ in range(depth[1] - 2):
            decoder.add(Dropout(dropout))
            decoder.add(LSTMDecoderCell(output_dim=hidden_dim, hidden_dim=hidden_dim))
        decoder.add(Dropout(dropout))
        decoder.add(LSTMDecoderCell(output_dim=output_dim, hidden_dim=hidden_dim))
    #decoder.add(Activation('sigmoid'))

    inputs = [_input]
    decoded = decoder(encoded)
    model = Model(inputs, decoded)
    return model
