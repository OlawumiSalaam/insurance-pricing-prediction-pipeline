from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


# define a function that evaluate model performance
def evaluate_model(y_true, y_predicted):
    mae = mean_absolute_error(y_true, y_predicted)
    mse = mean_squared_error(y_true, y_predicted)
    rmse = np.sqrt(mean_squared_error(y_true, y_predicted))
    r2_square = r2_score(y_true, y_predicted)
    return mae, rmse, r2_square

def print_evaluated_results(model,X_train,y_train,X_test,y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)


    # Evaluate Train and Test dataset
    model_train_mae , model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
    model_test_mae , model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

    # Printing results
    print('Model performance for Training set')
    print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
    print("- R2 Score: {:.4f}".format(model_train_r2))

    print('----------------------------------')

    print('Model performance for Test set')
    print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
    print("- R2 Score: {:.4f}".format(model_test_r2))


def calc_model_performance(y_true, y_pred):
    results = {}
    results['Root Mean Squared Error'] = mean_squared_error(
        y_true, y_pred, squared=False
    )
    results['Mean Squared Error'] = mean_squared_error(y_true, y_pred)
    results['Mean Absolute Error'] = mean_absolute_error(y_true, y_pred)
    results['Mean Absolute Percentage Error'] = mean_absolute_percentage_error(
        y_true, y_pred
    )
    results['R Squared'] = r2_score(y_true, y_pred)
    return results


def compare_model_performance(base_perf, new_perf):
    results = pd.DataFrame(
        columns=['base', 'new', 'abs_improvement', 'perc_improvement']
    )
    for metric, base_value in base_perf.items():
        base_value = round(base_value, 2)
        new_value = round(new_perf[metric], 2)
        results.loc[metric] = [
            base_value, new_value, new_value -
            base_value, round(100 * (new_value-base_value)/base_value, 2)
        ]
    return results


# def compare_homoscedasticity(y_true, y_pred_base, y_pred_new):
#     res_base = y_true-y_pred_base
#     res_new = y_true-y_pred_new
#     fig = make_subplots(rows=1, cols=1)
#     fig.add_trace(
#         go.Scatter(
#             x=y_true,
#             y=res_base,
#             name='base',
#             mode='markers'
#         )
#     )
#     fig.add_trace(
#         go.Scatter(
#             x=y_true,
#             y=res_new,
#             name='new',
#             mode='markers'
#         )
#     )
#     fig.show()

def calc_preds_in_residual_range(y_true, y_pred, range_):
    residuals = abs(y_true - y_pred)
    return 100 * (residuals <= range_).mean()


def calc_preds_in_residual_perc_range(y_true, y_pred, perc_range):
    perc_residuals = 100 * (abs(y_true - y_pred) / y_true)
    return 100 * (perc_residuals <= perc_range).mean()
