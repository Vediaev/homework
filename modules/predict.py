import dill
import os
import pandas as pd


def predict():
    path = os.environ.get('PROJECT_PATH', 'C:/Users/vedia/airflow_hw2')
    with open(path + '/data/models/' + os.listdir(path=path + '/data/models')[-1], 'rb') as file:
        model = dill.load(file)
    files = os.listdir(path=path + "/data/test")
    pred_list = []
    for i in files:
        df = pd.read_json(path + "/data/test/" + i, orient='index')
        df = df.transpose()
        y = model['model'].predict(df)
        pred_list.append([i.split('.')[0], *y])
    df_pred = pd.DataFrame(pred_list, columns=['car_id', 'pred'])
    df_pred.to_csv(path + '/data/predictions/predict.csv')


if __name__ == '__main__':
    predict()
