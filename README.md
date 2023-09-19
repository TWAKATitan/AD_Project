# AD_Project
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, f1_score

# 讀取Excel檔案，將'--'視為缺失值
data = pd.read_excel('ExampleDATA_FDG_SUVR.xlsx', na_values='--')

# Extract features and target
X = data.iloc[:, 13:]  # Starting from the 6th column as feature variables
y = data['Group']

# 使用SimpleImputer處理缺失值
imputer = SimpleImputer(strategy='mean')  # 使用均值填充缺失值
X = imputer.fit_transform(X)

# 將數據集分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#特徵            標籤

# 標準化特徵
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) 

# 創建SVM分類器
classifier = SVC(kernel='linear',decision_function_shape='ovr')  # ovr:One-to-Rest、ovo:One-to-One

# 訓練分類器
classifier.fit(X_train, y_train)

# 使用模型進行特徵選擇
model = SelectFromModel(classifier, prefit=True)

# 獲取被選擇的特徵索引
selected_feature_indices = model.get_support(indices=True)#指示哪些特徵被選擇

# 根據索引選擇對應的特徵名稱
selected_feature_names = data.columns[13:][selected_feature_indices]

# 輸出被選擇的特徵
print("Selected features:", selected_feature_names)

# 使用SVM分類器進行預測
y_pred = classifier.predict(X_test)

# 計算準確率和F1分數
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print('Accuracy: ', "%.2f" % (accuracy*100))
print('F1 Score: ', "%.2f" % (f1*100))
