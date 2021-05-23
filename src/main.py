from evaluate import evaluate_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

if __name__ == '__main__':
    res = evaluate_model(model_folder='../People-Flows/models/fdst.pth.tar',
                         data_folder='../dataset/FDST/test_data',
                         bs=8,
                         n_workers=1,
                         losses={'mse': mean_squared_error, 'mae': mean_absolute_error},
                         )
    print(res)
