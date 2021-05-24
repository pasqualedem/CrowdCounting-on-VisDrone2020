from evaluate import evaluate_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models.CC import CrowdCounter
from dataset.visdrone import load_test


def load_CC():
    cc = CrowdCounter([0], 'MobileCount')
    cc.load('../MobileCount/exp/05-24_12-41_SHHA_MobileCount_0.0001/all_ep_455_mae_95.8_mse_148.7.pth')
    return cc


if __name__ == '__main__':
    res = evaluate_model(model_function=load_CC,
                         data_function=load_test,
                         bs=8,
                         n_workers=2,
                         losses={'mse': mean_squared_error, 'mae': mean_absolute_error},
                         )
    print(res)
