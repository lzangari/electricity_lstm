import os
import json
from tqdm import tqdm
import numpy as np

from functools import partial
from sklearn.model_selection import TimeSeriesSplit

from keras_tuner import BayesianOptimization

from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, TensorBoard

from energy.utils.utils import create_sequences


def optimize_seq_length(
    data,
    seq_lengths,
    pred_length,
    output_size,
    built_model,
    path,
    model_type,
    epochs=10,
    batch_size=32,
    num_folds=3,
    verbose=2,
    attention=None,
    name=None,
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
        name (str, optional): The name of the file to save the best sequence length.

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
            # if attention_type is not None:
            if attention:
                model = built_model(
                    input_shape=(X_train.shape[1], X_train.shape[2]),
                    output_size=output_size,
                    pred_length=pred_length,
                )

                _ = model.fit(
                    X_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val, y_val),
                    verbose=verbose,
                )
                # loss = model.evaluate(X_val, y_val, verbose=verbose)

            else:
                model = built_model(
                    model_type=model_type,
                    input_shape=(X_train.shape[1], X_train.shape[2]),
                    output_size=output_size,
                    pred_length=pred_length,
                )
                # fit the model
                _ = model.fit(
                    X_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(X_val, y_val),
                    verbose=verbose,
                )

            loss = model.evaluate(X_val, y_val, verbose=verbose)

            #     # evaluate the model on the validation data and calculate the loss
            #     loss = model.evaluate(
            #         X_val, y_val.reshape(y_val.shape[0], -1), verbose=verbose
            #     )

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

    # save the losses and calculated sequence lengths
    with open(f"{path}/{name}_losses.json", "w") as f:
        json.dump({"losses": losses, "seq_lengths": calculated_seq_lengths}, f)

    return best_seq_length, losses, calculated_seq_lengths


def hypertune_model(
    build_model,
    X_train,
    y_train,
    X_val,
    y_val,
    model_type,
    input_shape,
    output_size,
    pred_length,
    path,
    name,
    max_trials=5,
    epochs=100,
    seed=None,
    attention=False,
    regularization=False,
):
    if attention:
        tuner = BayesianOptimization(
            partial(
                build_model,
                input_shape=input_shape,
                output_size=output_size,
                pred_length=pred_length,
                regularization=regularization,
            ),
            objective="val_loss",
            max_trials=max_trials,
            executions_per_trial=3,
            directory="hypertuning",
            project_name=f"energy_prediction_{name}",
            seed=seed,
        )

    else:
        tuner = BayesianOptimization(
            partial(
                build_model,
                model_type=model_type,
                input_shape=input_shape,
                output_size=output_size,
                pred_length=pred_length,
            ),
            objective="val_loss",
            max_trials=max_trials,
            executions_per_trial=3,
            directory="hypertuning",
            project_name=f"energy_prediction_{name}",
            seed=seed,
        )

    # summary of the search space
    tuner.search_space_summary()

    # save the best sequence length with its loss
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")

    # save the search space in a json file
    # with open(f'{path}/{name}_search_space.json', 'w') as f:
    #     json.dumps(tuner.search_space_summary())

    # Callback to log training and validation loss
    csv_logger = CSVLogger(f"training_log_{name}.csv", append=True)
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    # create a log directory
    if not os.path.exists("logs"):
        os.makedirs("logs")

    tensorboard = TensorBoard(log_dir=f"logs/{name}")

    # if attention:
    tuner.search(
        X_train,
        y_train,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, csv_logger, tensorboard],
    )

    # else:
    #     tuner.search(
    #         X_train,
    #         y_train.reshape(y_train.shape[0], -1),
    #         epochs=epochs,
    #         validation_data=(X_val, y_val.reshape(y_val.shape[0], -1)),
    #         callbacks=[early_stopping, csv_logger, tensorboard],
    #     )

    # retrieve the best model and hyperparameters
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]
    print(f"Best hyperparameters: {best_hp.values}")
    print(f"Best model summary: {best_model.summary()}")
    # print(f"Best model configuration: {best_model.get_config()}")
    # print(f"Best model hyperparameters: {best_model.get_hyperparameters()}")

    # evaluate the best model
    # if attention:
    train_loss = best_model.evaluate(X_train, y_train, verbose=0)
    val_loss = best_model.evaluate(X_val, y_val, verbose=0)
    # else:
    #     train_loss = best_model.evaluate(
    #         X_train, y_train.reshape(y_train.shape[0], -1), verbose=0
    #     )
    #     val_loss = best_model.evaluate(X_val, y_val.reshape(y_val.shape[0], -1), verbose=0)

    # save the best model
    # model_path = os.path.join(path, f'best_energy_prediction_model_{name}.h5')
    best_model.save(f"{path}/best_energy_prediction_model_{name}.keras")
    # best_model.save(model_path)

    # save the best hyperparameters
    hp_path = os.path.join(path, f"best_hyperparameters_{name}.json")
    with open(hp_path, "w") as f:
        json.dump(best_hp.values, f)
    print(f"Best hyperparameters saved at: {hp_path}")

    # save the results summary
    results_summary = {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_hp": best_hp.values,
    }
    results_path = os.path.join(path, f"results_summary_{name}.json")
    with open(results_path, "w") as f:
        json.dump(results_summary, f)
    # save the tuner
    # tuner_path = os.path.join(path, f'tuner_{name}.json')
    tuner.save()

    # Visualization of hyperparameter tuning
    tuner.results_summary()

    return best_model, tuner, train_loss, val_loss, best_hp.values
