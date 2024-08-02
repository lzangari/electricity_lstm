from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Input,
    RepeatVector,
    TimeDistributed,
    AdditiveAttention,
    Reshape,
    Flatten,
)
from tensorflow.keras.optimizers import Adam, schedules
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras import losses
from keras.layers import concatenate


# disable_eager_execution()
# LSTM-based model
def build_lstm_based_model(
    model_type,
    input_shape,
    output_size,
    pred_length,
    units=50,
    dropout=0.2,
    learning_rate=0.005,
    alpha=0.1,
):
    if "naive" in model_type:
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(units=units, return_sequences=True))
        model.add(Dropout(dropout))
        # model.add(TimeDistributed(Dense(output_size)))
        model.add(TimeDistributed(Dense(units=output_size)))
        # flatten it
        model.add(Flatten())
        # map to the desired output size
        model.add(Dense(units=pred_length * output_size))
        model.add(Reshape((pred_length, output_size)))

    elif "stacked" in model_type:
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(units=units, return_sequences=True))
        model.add(LSTM(units=units, return_sequences=True, kernel_regularizer=l2(1e-4)))
        model.add(LeakyReLU(negative_slope=alpha))
        model.add(Dropout(dropout))
        model.add(TimeDistributed(Dense(output_size)))
        # model.add(Dense(pred_length * output_size))

    optimizer = Adam(learning_rate=learning_rate, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss="mse")
    # model.summary()
    return model


# Hyperparameter tuning for LSTM-based model
def build_lstm_based_model_with_hp(
    hp, model_type, input_shape, output_size, pred_length
):
    units = hp.Int("units", min_value=30, max_value=170, step=30)
    dropout = hp.Float("dropout", min_value=0.15, max_value=0.3, step=0.05)
    learning_rate = hp.Float(
        "learning_rate", min_value=0.0001, max_value=0.024, step=0.0004
    )
    if "stacked" in model_type:
        alpha = hp.Float("alpha", min_value=0.1, max_value=0.2, step=0.1)

    if "naive" in model_type:
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(units=units, return_sequences=True))
        model.add(Dropout(dropout))
        # model.add(TimeDistributed(Dense(output_size)))
        model.add(TimeDistributed(Dense(units=output_size)))
        # flatten it
        model.add(Flatten())
        # map to the desired output size
        model.add(Dense(units=pred_length * output_size))
        model.add(Reshape((pred_length, output_size)))

    elif "stacked" in model_type:
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(units=units, return_sequences=True))
        model.add(LSTM(units=units, return_sequences=True, kernel_regularizer=l2(1e-4)))
        model.add(LeakyReLU(negative_slope=alpha))
        model.add(Dropout(dropout))
        model.add(TimeDistributed(Dense(output_size)))
        # model.add(Dense(pred_length * output_size))

    optimizer = Adam(learning_rate=learning_rate, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss="mse")

    return model


# Seq2Seq LSTM model
def build_seq2seq_lstm_model(
    input_shape,
    output_size,
    pred_length,
    units=50,
    dropout=0.2,
    learning_rate=0.005,
    regularization=True,
):
    # Encoder
    encoder_inputs = Input(shape=input_shape)
    # LSTM layer with return_state=True to get the hidden and cell states
    encoder_lstm, state_h, state_c = LSTM(
        units, return_state=True, return_sequences=True
    )(encoder_inputs)
    if regularization:
        # leaky relu activation function
        encoder_lstm = LeakyReLU(negative_slope=0.1)(encoder_lstm)
        # dropout layer
        encoder_lstm = Dropout(dropout)(encoder_lstm)
    # repeat the hidden state of the encoder LSTM for pred_length times to be used as input to the decoder
    decoder_input = RepeatVector(pred_length)(state_h)
    # LSTM layer with return_sequences=True to get the output at each time step
    decoder_lstm = LSTM(units, return_sequences=True, return_state=False)(
        decoder_input, initial_state=[state_h, state_c]
    )
    if regularization:
        # leaky relu
        decoder_lstm = LeakyReLU(negative_slope=0.1)(decoder_lstm)
        # dropout
        decoder_lstm = Dropout(dropout)(decoder_lstm)
    # create a context vector using the attention mechanism for capturing the relevant information from the encoder output at each time step
    context_vector = AdditiveAttention()([decoder_lstm, encoder_lstm])
    # concatenate the context vector with the decoder output for each time step
    decoder_combined_context = concatenate([context_vector, decoder_lstm])
    # time distributed dense layer to get the output at each time step
    out = TimeDistributed(Dense(output_size))(decoder_combined_context)
    # define the model with encoder input and decoder output
    model = Model(inputs=encoder_inputs, outputs=out)
    # compile the model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipvalue=1.0), loss="mse"
    )

    return model


# rmse and mape


def build_seq2seq_lstm_model_with_hp(
    hp, input_shape, output_size, pred_length, regularization=True, scheduler=True
):
    units = hp.Int("units", min_value=30, max_value=150, step=30)
    learning_rate = hp.Float(
        "learning_rate", min_value=0.0001, max_value=0.025, step=0.0003
    )
    if regularization:
        dropout = hp.Float("dropout", min_value=0.1, max_value=0.3, step=0.05)
        alpha = hp.Float("alpha", min_value=0.01, max_value=0.05, step=0.02)

    # Encoder
    encoder_inputs = Input(shape=input_shape)
    # LSTM layer with return_state=True to get the hidden and cell states
    encoder_lstm, state_h, state_c = LSTM(
        units, return_state=True, return_sequences=True
    )(encoder_inputs)
    if regularization:
        # leaky relu activation function
        encoder_lstm = LeakyReLU(negative_slope=alpha)(encoder_lstm)
        # dropout layer
        encoder_lstm = Dropout(dropout)(encoder_lstm)
    # repeat the hidden state of the encoder LSTM for pred_length times to be used as input to the decoder
    decoder_input = RepeatVector(pred_length)(state_h)
    # LSTM layer with return_sequences=True to get the output at each time step
    decoder_lstm = LSTM(units, return_sequences=True, return_state=False)(
        decoder_input, initial_state=[state_h, state_c]
    )
    if regularization:
        # leaky relu
        decoder_lstm = LeakyReLU(negative_slope=alpha)(decoder_lstm)
        # dropout
        decoder_lstm = Dropout(dropout)(decoder_lstm)
    # create a context vector using the attention mechanism for capturing the relevant information from the encoder output at each time step
    context_vector = AdditiveAttention()([decoder_lstm, encoder_lstm])
    # concatenate the context vector with the decoder output for each time step
    decoder_combined_context = concatenate([context_vector, decoder_lstm])
    # time distributed dense layer to get the output at each time step
    out = TimeDistributed(Dense(output_size))(decoder_combined_context)
    # define the model with encoder input and decoder output
    model = Model(inputs=encoder_inputs, outputs=out)
    # compile the model
    if scheduler:
        learning_rate_schedule = schedules.ExponentialDecay(
            initial_learning_rate=learning_rate, decay_steps=100, decay_rate=0.96
        )
    else:
        learning_rate_schedule = learning_rate

    model.compile(
        optimizer=Adam(learning_rate=learning_rate_schedule, clipvalue=1.0),
        loss=losses.Huber(),
    )
    # loss="mse", losses.Huber()
    print(f"model summary: {model.summary()}")

    return model
