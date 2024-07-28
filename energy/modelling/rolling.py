import numpy as np


def rolling_prediction(
    model,
    initial_sequence,
    test_data,
    seq_length,
    pred_length,
    scaler,
    features,
    output_size,
):
    current_sequence = initial_sequence.copy()
    predictions = []
    actuals = []

    for i in range(0, len(test_data) - seq_length - pred_length + 1, pred_length):
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

    # Flatten predictions to 2D array
    predictions = np.array(predictions).reshape(-1, output_size)
    actuals = np.array(actuals).reshape(-1, len(features) + output_size)
    # FIXME: after finishing trainign this should be checked
    ####################################################################################################################
    # Prepare for inverse transformation, it stack zeros for the size of the features to match the scaler's shape
    combined_predictions = np.hstack(
        (predictions, np.zeros((predictions.shape[0], len(features))))
    )
    # Inverse transform predictions and actuals
    predictions_inverse = scaler.inverse_transform(combined_predictions)[
        :, :output_size
    ]
    actuals_inverse = scaler.inverse_transform(actuals)[:, :output_size]

    return predictions_inverse, actuals_inverse
