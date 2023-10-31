#ad, mci
#mci 442-1335, ad 1336-1651
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.impute import SimpleImputer

data = pd.read_excel('newdata.xlsx', na_values='--')

def train_admci(data):
    # 選取特定的資料列索引
    selected_rows = [idx for idx in range(len(data)) if idx in range(442, 1335) or idx in range(1336, 1651)]
    X = data.iloc[selected_rows, 14:] 

    # 使用SimpleImputer處理缺失值
    imputer = SimpleImputer(strategy='mean')  # 使用平均值填補缺失值
    X = imputer.fit_transform(X)

    # 資料標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 創建SVM分類器並使用最佳參數
    best_params = {'C': 10, 'gamma': 'scale', 'kernel': 'poly'}
    classifier = svm.SVC(**best_params)

    # 訓練模型
    classifier.fit(X_scaled, data.iloc[selected_rows]['Group'])

    return classifier, scaler, imputer

def admci(data1):
    classifier, scaler, imputer = train_admci(data)

    # 預測未知資料
    data1_scaled = scaler.transform(imputer.transform(data1))
    data1_predictions = classifier.predict(data1_scaled)
    return data1_predictions

data1 = pd.read_excel('predicted_data_2.xlsx', na_values='--')
unknown_data = data1.iloc[:, 14:]
unknown_predictions = admci(unknown_data)

data1['Group'] = unknown_predictions

predicted_data_1 = data1[data1['Group'] == 1]
predicted_data_22 = data1[data1['Group'] == 2]

predicted_data_1.loc[:, 'Group'] = 1
predicted_data_22.loc[:, 'Group'] = 2

predicted_data_1.to_excel('predicted_data_1.xlsx', index=False)
predicted_data_22.to_excel('predicted_data_22.xlsx', index=False)
