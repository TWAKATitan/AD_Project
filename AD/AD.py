import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import SelectFromModel

# 讀取資料
data = pd.read_excel("ExampleDATA_FDG_SUVR.xlsx")

# 將 "--" 替換為 NaN
data.replace("--", np.nan, inplace=True)

# 使用均值填充缺失值
data.fillna(data.mean(numeric_only=True), inplace=True)  # 使用 numeric_only=True

# 拆分特徵和標籤
X = data.iloc[:, 6:]  # 特徵從第6列開始
y = data["Group"]     

# 將資料集分為訓練和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 標準化特徵
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 使用多類別SVM模型進行訓練
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train_scaled, y_train)

# 從模型中取得特徵重要性
feature_importances = svm_model.coef_

# 選擇對類別判別影響較大的特徵
selector = SelectFromModel(estimator=svm_model, prefit=True)
selected_features = selector.transform(X_train_scaled)

# 取得所選特徵的索引
selected_feature_indices = selector.get_support(indices=True)

# 列印所選特徵的索引
print("Selected feature indices:", selected_feature_indices)

# 列印所選特徵的名稱
selected_feature_names = X.columns[selected_feature_indices]
print("Selected feature names:", selected_feature_names)
