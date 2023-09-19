import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, f1_score

# 讀取Excel檔案，將'--'視為缺失值
data = pd.read_excel('ExampleDATA_FDG_SUVR.xlsx', na_values='--')

# 选择存在于数据框中的行索引
selected_rows = [idx for idx in range(len(data)) if idx in range(1, 259) or idx in range(457, 709)]

# 提取特徵和目標
X = data.iloc[selected_rows, 13:]  # 從第6行開始作為特徵變數
y = data.iloc[selected_rows]['Group']
# 使用SimpleImputer處理缺失值
imputer = SimpleImputer(strategy='mean')  # 使用均值填充缺失值
X = imputer.fit_transform(X)

# 將數據集分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#特徵            標籤

# 創建隨機森林分類器
classifier = RandomForestClassifier(random_state=42)

# 定義參數的範圍
param_grid = {
    'n_estimators': np.arange(70, 120, 10), #決策樹數量
    'max_depth': np.arange(15, 20, 1), #每棵樹的深度
    'min_samples_split': np.arange(5, 10, 1), #樣本>=5
    'min_samples_leaf': np.arange(5, 10, 1), #節點>=5
    'max_features': ['sqrt'] #再查詢
}

# 建立 GridSearchCV 物件
grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, scoring='accuracy')

# 在訓練集上進行參數搜索
grid_search.fit(X_train, y_train)

# 最佳參數組合
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# 使用最佳參數建立新的隨機森林模型
best_rf_model = RandomForestClassifier(**best_params, random_state=42)

# 訓練最佳模型
best_rf_model.fit(X_train, y_train)

# 使用模型進行特徵選擇
model = SelectFromModel(best_rf_model, prefit=True)

# 獲取被選擇的特徵索引
selected_feature_indices = model.get_support(indices=True)

# 根據索引選擇對應的特徵名稱
selected_feature_names = data.columns[11:][selected_feature_indices]

# 輸出被選擇的特徵
print("Selected features:", selected_feature_names)

# 使用隨機森林分類器進行預測
y_pred = best_rf_model.predict(X_test)

# 計算準確率和F1分數
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print('Accuracy: ', "%.2f" % (accuracy*100))
print('F1 Score: ', "%.2f" % (f1*100))
