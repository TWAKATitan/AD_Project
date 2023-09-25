#邏輯回歸混隨機森林 66.06

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression

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
best_params = {'C': 10, 'gamma':'scale', 'kernel':'poly'}
svm_classifier = svm.SVC(**best_params)

# 训练SVM模型
svm_classifier.fit(X_train_scaled, y_train)

# 使用SVM模型进行预测
svm_y_pred = svm_classifier.predict(X_test_scaled)

# 计算SVM模型的准确率和F1分数
svm_accuracy = accuracy_score(y_test, svm_y_pred)
svm_f1 = f1_score(y_test, svm_y_pred, average='weighted')

print('SVM Accuracy: ', "%.2f" % (svm_accuracy * 100))
print('SVM F1 Score: ', "%.2f" % (svm_f1 * 100))

# 训练随机森林模型
random_forest_classifier = RandomForestClassifier(n_estimators=300, random_state=42, max_depth=10, min_samples_split=15)
random_forest_classifier.fit(X_train_scaled, y_train)

# 使用随机森林模型的输出作为新特征
random_forest_features = random_forest_classifier.predict_proba(X_train_scaled)

# 合并随机森林特征和原始特征
X_train_combined = np.hstack((X_train_scaled, random_forest_features))

# 训练一个新的Logistic回归模型，使用合并后的特征
logistic_regression_classifier = LogisticRegression(solver='lbfgs', max_iter=1000)  # 增加max_iter
logistic_regression_classifier.fit(X_train_combined, y_train)

# 使用合并后的特征进行测试集预测
random_forest_test_features = random_forest_classifier.predict_proba(X_test_scaled)
X_test_combined = np.hstack((X_test_scaled, random_forest_test_features))
logistic_regression_y_pred = logistic_regression_classifier.predict(X_test_combined)

# 计算Logistic回归模型的准确率和F1分数
logistic_regression_accuracy = accuracy_score(y_test, logistic_regression_y_pred)
logistic_regression_f1 = f1_score(y_test, logistic_regression_y_pred, average='weighted')

print('Logistic Regression Accuracy: ', "%.2f" % (logistic_regression_accuracy * 100))
print('Logistic Regression F1 Score: ', "%.2f" % (logistic_regression_f1 * 100))
