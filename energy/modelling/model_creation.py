# build the naive LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


# Naive LSTM model
def build_naive_lstm_model(input_shape, output_size, units=50, dropout=0.2, clip=True):
    """Builds a naive LSTM model with a single LSTM layer and a single Dense layer.

    Args:
        input_shape (int): input shape of the model.
        output_size (int): output size of the model.
        units (int, optional): the number of units in the LSTM layer. Defaults to 50.
        dropout (float, optional): the dropout rate. Defaults to 0.2.
        clip (bool, optional): whether to use gradient clipping. Defaults to True.

    Returns:
        Sequential: the compiled LSTM model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=units, activation="relu"))
    model.add(Dropout(dropout))
    model.add(Dense(output_size))

    if clip:
        # Compile with gradient clipping
        optimizer = Adam(clipvalue=1.0)
        model.compile(optimizer=optimizer, loss="mse")
    else:
        model.compile(optimizer="adam", loss="mse")

    return model


# Stacked LSTM model
def build_stacked_lstm_model(
    input_shape, output_size, units=50, dropout=0.2, clip=True
):
    """Builds a stacked LSTM model with two LSTM layers and a single Dense layer.

    Args:
        input_shape (int): input shape of the model.
        output_size (int): output size of the model.
        units (int, optional): the number of units in the LSTM layer. Defaults to 50.
        dropout (float, optional): the dropout rate. Defaults to 0.2.
        clip (bool, optional): whether to use gradient clipping. Defaults to True.

    Returns:
        Sequential: the compiled LSTM model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units=units, activation="relu", return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units, activation="relu"))
    model.add(Dropout(dropout))
    model.add(Dense(output_size))

    if clip:
        # Compile with gradient clipping
        optimizer = Adam(clipvalue=1.0)
        model.compile(optimizer=optimizer, loss="mse")
    else:
        model.compile(optimizer="adam", loss="mse")

    return model


# Seq2Seq LSTM model
