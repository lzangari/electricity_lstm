import os
import json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def huber_loss(y_true, y_pred, delta=1.0):
    """
    Calculate the Huber loss between two arrays.
    Parameters:
    y_true (array-like): True values
    y_pred (array-like): Predicted values
    delta (float): The threshold parameter

    Returns:
    float: The Huber loss
    """
    # Convert input to numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Calculate the difference between true and predicted values
    diff = y_true - y_pred
    abs_diff = np.abs(diff)

    # Calculate the quadratic and linear terms
    quadratic = np.minimum(abs_diff, delta)
    linear = abs_diff - quadratic

    # Calculate the Huber loss
    loss = 0.5 * quadratic**2 + delta * linear

    # Return the mean Huber loss
    return np.sum(loss) / len(y_true)


def calculate_error_metrics(df, prediction_features, output_features, path, name):
    metrics = {}

    for pred_feature, output_feature in zip(prediction_features, output_features):
        # Calculate the errors
        mse = mean_squared_error(df[output_feature], df[pred_feature])
        mae = mean_absolute_error(df[output_feature], df[pred_feature])
        mape = (
            np.mean(
                np.abs((df[output_feature] - df[pred_feature]) / df[output_feature])
            )
            * 100
        )
        rmse = np.sqrt(mse)
        # huber loss
        huber = huber_loss(df[output_feature], df[pred_feature])
        # Store the results in a dictionary
        metrics[output_feature] = {
            "MSE": mse,
            "MAE": mae,
            "MAPE": mape,
            "RMSE": rmse,
            "Huber Loss": huber,
        }

    print(f"Metrics: {metrics}")
    # check if the path exists
    if not os.path.exists(path):
        os.makedirs(path)
    # save the metrics to a json file
    with open(f"{path}/metrics_{name}.json", "w") as file:
        json.dump(metrics, file)

    return metrics
