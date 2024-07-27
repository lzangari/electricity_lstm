# build the naive LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.regularizers import l2


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
    if model_type == "lstm_naive":
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(units=units, activation="relu"))
        model.add(Dropout(dropout))
        model.add(Dense(pred_length * output_size))

    elif model_type == "lstm_stacked":
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(units=units, return_sequences=True))
        model.add(LSTM(units=units, kernel_regularizer=l2(1e-4)))
        model.add(LeakyReLU(negative_slope=alpha))
        model.add(Dropout(dropout))
        model.add(Dense(pred_length * output_size))

    optimizer = Adam(learning_rate=learning_rate, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss="mse")

    return model


# Hyperparameter tuning for LSTM-based model
def build_lstm_based_model_with_hp(
    hp, model_type, input_shape, output_size, pred_length
):
    units = hp.Int("units", min_value=30, max_value=240, step=30)
    dropout = hp.Float("dropout", min_value=0.1, max_value=0.25, step=0.05)
    learning_rate = hp.Float(
        "learning_rate", min_value=0.001, max_value=0.03, step=0.002
    )
    if model_type == "lstm_stacked":
        alpha = hp.Float("alpha", min_value=0.1, max_value=0.3, step=0.1)

    if model_type == "lstm_naive":
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(units=units, activation="relu"))
        model.add(Dropout(dropout))
        model.add(Dense(pred_length * output_size))

    elif model_type == "lstm_stacked":
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(units=units, return_sequences=True))
        model.add(LSTM(units=units, kernel_regularizer=l2(1e-4)))
        model.add(LeakyReLU(negative_slope=alpha))
        model.add(Dropout(dropout))
        model.add(Dense(pred_length * output_size))

    optimizer = Adam(learning_rate=learning_rate, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss="mse")

    return model


# Seq2Seq LSTM model
