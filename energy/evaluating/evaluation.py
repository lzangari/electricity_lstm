import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_error_metrics(df, prediction_features, output_features):
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

        # Store the results in a dictionary
        metrics[output_feature] = {"MSE": mse, "MAE": mae, "MAPE": mape, "RMSE": rmse}

    print(f"Metrics: {metrics}")

    return metrics
