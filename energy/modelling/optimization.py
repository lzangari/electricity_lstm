import os
import json
from tqdm import tqdm
import numpy as np

from sklearn.model_selection import TimeSeriesSplit

from energy.utils.utils import create_sequences


def optimize_seq_length(
    data,
    seq_lengths,
    pred_length,
    output_size,
    built_model,
    path,
    epochs=10,
    batch_size=32,
    num_folds=3,
    verbose=2,
    name="NAIVE_quarterhour",
):
    """Optimize the sequence length for the LSTM model using KFold cross-validation.

    Args:
        data (dataframe): The input data to be sequenced.
        seq_lengths (list): A list of sequence lengths to try.
        pred_length (int): The length of the output sequences.
        output_size (int): The number of output features.
        built_model (function): The function to build the LSTM model.
        path (str): The path to save the best sequence length.
        epochs (int, optional): Number of epochs to train the model. Defaults to 10.
        batch_size (int, optional): Batch size for training the model. Defaults to 32.
        num_folds (int, optional): Number of folds for cross-validation. Defaults to 3.
        verbose (int, optional): Verbosity mode. Defaults to 2.
        name (str, optional): The name of the file to save the best sequence length. Defaults to "NAIVE_quarterhour".

    Returns:
        int: The best sequence length.
        list: A list of losses for each sequence length.
        list: A list of calculated sequence lengths.
    """
    # Use KFold cross-validation to find the best sequence length
    tscv = TimeSeriesSplit(n_splits=num_folds)
    # Initialize the best sequence length and loss
    best_seq_length = seq_lengths[0]
    # Set the best loss to infinity
    best_loss = float("inf")
    losses = []
    calculated_seq_lengths = []

    for seq_length in tqdm(seq_lengths):
        total_loss = 0
        # store the loss for each fold
        fold_losses = []
        for train_index, val_index in tscv.split(data):
            # train and validation data for the current fold
            train_data, val_data = data[train_index], data[val_index]
            # Create sequences for the current fold for training and validation data
            X_train, y_train = create_sequences(
                train_data, seq_length, pred_length, output_size
            )
            X_val, y_val = create_sequences(
                val_data, seq_length, pred_length, output_size
            )

            # Build the model and train it on the training data
            model = built_model(
                (X_train.shape[1], X_train.shape[2]), pred_length * output_size
            )
            # fit the model
            _ = model.fit(
                X_train,
                y_train.reshape(y_train.shape[0], -1),
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val.reshape(y_val.shape[0], -1)),
                verbose=verbose,
            )

            # evaluate the model on the validation data and calculate the loss
            loss = model.evaluate(
                X_val, y_val.reshape(y_val.shape[0], -1), verbose=verbose
            )

            if np.isnan(loss):
                print(
                    f"NaN loss encountered for sequence length {seq_length}. Skipping this length."
                )
                continue

            total_loss += loss
            fold_losses.append(loss)
            print(f"Sequence Length: {seq_length}, Fold Loss: {total_loss / num_folds}")

        if fold_losses:
            # calculate the average loss for the current sequence
            # avg_loss = total_loss / num_folds
            avg_loss = total_loss / len(fold_losses)
            print(f"Sequence Length: {seq_length}, Average Loss: {avg_loss}")
            losses.append(avg_loss)
            calculated_seq_lengths.append(seq_length)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_seq_length = seq_length

    # save the best sequence length with its loss
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

    with open(f"{path}/{name}_best_sequence_length.json", "w") as f:
        json.dump({"best_seq_length": best_seq_length, "loss": best_loss}, f)

    return best_seq_length, losses, calculated_seq_lengths
