import numpy as np
import pandas as pd


def rolling_prediction(
    model,
    initial_sequence,
    test_data,
    seq_length,
    pred_length,
    scaler,
    features,
    output_size,
    output_features,
    path,
    name,
):
    current_sequence = initial_sequence.copy()
    predictions = []
    actuals = []

    for i in range(0, len(test_data) - seq_length - pred_length + 1, pred_length):
        # 0, len(test_data) - seq_length - pred_length + 1, pred_length)
        # predict the next sequence
        next_pred = model.predict(
            current_sequence.reshape(
                1, current_sequence.shape[0], current_sequence.shape[1]
            )
        )[0]
        predictions.append(next_pred)

        # get the actual data with the real labels
        actual_data = test_data[i + seq_length : i + seq_length + pred_length]
        actuals.append(actual_data)

        # update the sequence with the actual test data for the next prediction
        current_sequence = np.vstack([current_sequence[pred_length:], actual_data])

    # _ = np.array(predictions).reshape(-1, output_size)
    # predictions has (x, pred_length, output_size dimensions) -> (x * pred_length, output_size)
    # actuals has (x, pred_length, all_features) dimensions -> (x * pred_length, all_features)
    actuals_data = np.array(actuals).reshape(-1, len(features) + output_size)
    # wrong one
    # actuals_data = np.array(actuals).reshape(-1, len(features))

    # (actual - 2 label columns) and put instead two prediction columns in the same shape
    # predictions: 546, 24, 2, actuals: 546, 24, 22 -> the first 2 columns are the labels and should be replaced by the predictions
    # np.array(predictions.shape[0], predictions.shape[1], actuals.shape[2])
    # actual_predictions = np.zeros((np.array(predictions).shape[0], np.array(predictions).shape[1], actuals_data.shape[1]))
    # create another actual predictions that the first two columns are the predictions and the rest are the actuals
    actuals_predictions = np.array(actuals).copy()
    # replace the first two columns with the predictions
    actuals_predictions[:, :, :output_size] = np.array(predictions).copy()
    actuals_predictions = actuals_predictions.reshape(-1, len(features) + output_size)
    # wrong one
    # actuals_predictions = actuals_predictions.reshape(-1, len(features))

    # Inverse transform predictions and actuals
    actuals_predictions_inverse = scaler.inverse_transform(actuals_predictions)
    actuals_inverse = scaler.inverse_transform(actuals_data)

    # make a dataframe with the actuals_predictions_inverse and assign the columns names using output_features and features
    # create the columns names for the dataframe
    actuals_columns = output_features + features
    # predictions columns with having "pred_" before the name of the output features
    predictions_features = ["pred_" + feature for feature in output_features]
    predicitons_columns = predictions_features + features
    # create the dataframe
    predictions_inverse = pd.DataFrame(
        actuals_predictions_inverse, columns=predicitons_columns
    )
    actuals_inverse = pd.DataFrame(actuals_inverse, columns=actuals_columns)

    # import the main trasformed data
    main_data = pd.read_csv(f"data/transformed/transformed_{name}.csv")
    if seq_length == 24:
        filter_data = main_data[
            (main_data["start_time"] >= "2023-01-02")
            & (main_data["start_time"] < "2024-06-30 01:00:00")
        ]
        filter_data = filter_data.reset_index()
    elif seq_length == 168:
        filter_data = main_data[
            (main_data["start_time"] >= "2023-01-08")
            & (main_data["start_time"] < "2024-06-30 01:00:00")
        ]
        filter_data = filter_data.reset_index()
    elif seq_length == 48:
        filter_data = main_data[
            (main_data["start_time"] >= "2023-01-03")
            & (main_data["start_time"] < "2024-06-30 01:00:00")
        ]
        filter_data = filter_data.reset_index()

    all_data = pd.concat([actuals_inverse, predictions_inverse, filter_data], axis=1)
    all_data = all_data.loc[:, ~all_data.columns.duplicated()].copy()
    # merge the predictions and actuals dataframes on the features columns
    # FIXME
    # merged_df = pd.merge(predictions_inverse, actuals_inverse, on=features, how="inner")
    # merged_columns = predictions_features + output_features + features
    # merged_df = merged_df[merged_columns]

    # save the merged dataframe to a csv file
    all_data.to_csv(f"{path}/predictions_smard_real_{name}.csv", index=False)

    return predictions_inverse, actuals_inverse, all_data
