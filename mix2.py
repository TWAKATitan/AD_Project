#65.05
#邏輯回歸+svm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression  # 导入Logistic回归模型

# 读取Excel文件，将'--'视为缺失值
data = pd.read_excel('newdata.xlsx', na_values='--')

# 提取特徵和目標
X = data.iloc[:, 14:]  # Starting from the 6th column as feature variables
y = data['Group']

# 处理缺失值使用SimpleImputer
imputer = SimpleImputer(strategy='mean')  # 使用均值填充缺失值
X = imputer.fit_transform(X)

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建SVM分类器并使用最佳参数
best_params_svm = {'C': 10, 'gamma': 'scale', 'kernel': 'poly'}
svm_classifier = svm.SVC(**best_params_svm)

# 训练SVM模型
svm_classifier.fit(X_train_scaled, y_train)

# 使用SVM模型进行预测
svm_y_pred = svm_classifier.predict(X_test_scaled)

# 计算SVM模型的准确率和F1分数
svm_accuracy = accuracy_score(y_test, svm_y_pred)
svm_f1 = f1_score(y_test, svm_y_pred, average='weighted')

print('SVM Accuracy: ', "%.2f" % (svm_accuracy * 100))
print('SVM F1 Score: ', "%.2f" % (svm_f1 * 100))

# 训练Logistic回归模型并生成新特征
logistic_regression_classifier = LogisticRegression(max_iter=1000, random_state=42)
logistic_regression_classifier.fit(X_train_scaled, y_train)
logistic_regression_features = logistic_regression_classifier.predict_proba(X_train_scaled)

# 合并Logistic回归特征和原始特征
X_train_combined = np.hstack((X_train_scaled, logistic_regression_features))

# 训练一个新的SVM模型，使用合并后的特征
new_svm_classifier = svm.SVC(**best_params_svm)
new_svm_classifier.fit(X_train_combined, y_train)

# 使用合并后的特征进行测试集预测
logistic_regression_test_features = logistic_regression_classifier.predict_proba(X_test_scaled)
X_test_combined = np.hstack((X_test_scaled, logistic_regression_test_features))
new_svm_y_pred = new_svm_classifier.predict(X_test_combined)

# 计算混合模型的准确率和F1分数
new_svm_accuracy = accuracy_score(y_test, new_svm_y_pred)
new_svm_f1 = f1_score(y_test, new_svm_y_pred, average='weighted')

print('Mixed Model Accuracy: ', "%.2f" % (new_svm_accuracy * 100))
print('Mixed Model F1 Score: ', "%.2f" % (new_svm_f1 * 100))
