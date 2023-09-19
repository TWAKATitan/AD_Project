# AD_Project

#57.28
#直接使用最佳參數
#ANOVA分析
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score
import shap  # 导入SHAP库

# 读取Excel文件，将'--'视为缺失值
data = pd.read_excel('ExampleDATA_FDG_SUVR.xlsx', na_values='--')

# 提取特徵和目標
X = data.iloc[259:708, 13:]  
y = data['Group'].iloc[259:708]

# 处理缺失值使用SimpleImputer
imputer = SimpleImputer(strategy='mean')  # 使用均值填充缺失值
X = imputer.fit_transform(X)

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 特征选择 - 使用SelectKBest选择前k个最重要的特征
selector = SelectKBest(score_func=f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# 创建SVM分类器并使用最佳参数
best_params = {'C': 1, 'gamma': 'scale', 'kernel': 'linear'}
classifier = svm.SVC(**best_params)

# 训练模型
classifier.fit(X_train_scaled, y_train)

# 使用最佳参数的模型进行预测
y_pred = classifier.predict(X_test_scaled)

# 计算准确率和F1分数
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print('Accuracy: ', "%.2f" % (accuracy * 100))
print('F1 Score: ', "%.2f" % (f1 * 100))

# 获取由最佳模型选择的特征的索引
selected_feature_indices = selector.get_support(indices=True)

# 获取所选特征的列名
selected_feature_names = data.columns[13:][selected_feature_indices]

# 输出所选特征的列名
print("Selected Features:")
for feature in selected_feature_names:
    print(feature)

'''
#57.28
#递归特征消除
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, f1_score
import shap

# 读取Excel文件，将'--'视为缺失值
data = pd.read_excel('ExampleDATA_FDG_SUVR.xlsx', na_values='--')

# 提取特征和目标
X = data.iloc[:, 13:]  # 从第13列开始作为特征变量
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
best_params = {'C': 1, 'gamma': 'scale', 'kernel': 'linear'}
classifier = SVC(**best_params)

# 特征选择 - 使用递归特征消除（RFE）
estimator = SVC(**best_params)  # 选择您的SVM分类器
selector = RFE(estimator, n_features_to_select=10)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# 训练模型
classifier.fit(X_train_selected, y_train)

# 使用最佳参数的模型进行预测
y_pred = classifier.predict(X_test_selected)

# 计算准确率和F1分数
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print('Accuracy: ', "%.2f" % (accuracy * 100))
print('F1 Score: ', "%.2f" % (f1 * 100))

# 获取由最佳模型选择的特征的索引
selected_feature_indices = selector.get_support(indices=True)

# 获取所选特征的列名
selected_feature_names = data.columns[13:][selected_feature_indices]

# 输出所选特征的列名
print("Selected Features:")
for feature in selected_feature_names:
    print(feature)
'''
'''
#57.75
#L1正則化
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
import shap

# 读取Excel文件，将'--'视为缺失值
data = pd.read_excel('ExampleDATA_FDG_SUVR.xlsx', na_values='--')

# 提取特征和目标
X = data.iloc[:, 13:]  # 从第13列开始作为特征变量
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

# 创建SVM分类器并使用L1正则化
best_params = {'C': 1, 'gamma': 'scale'}
classifier = SVC(**best_params, kernel='linear')

# 训练模型
classifier.fit(X_train_scaled, y_train)

# 获取特征权重
feature_weights = classifier.coef_

# 根据特征权重进行特征选择
selected_feature_indices = []
non_zero_count = 0
for i in range(feature_weights.shape[1]):
    if any(feature_weights[:, i] != 0):
        selected_feature_indices.append(i)
        non_zero_count += 1
        if non_zero_count >= 10:  # 选择前10个非零权重的特征
            break

X_train_selected = X_train_scaled[:, selected_feature_indices]
X_test_selected = X_test_scaled[:, selected_feature_indices]

# 使用最佳参数的模型进行预测
y_pred = classifier.predict(X_test_scaled)

# 计算准确率和F1分数
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print('Accuracy: ', "%.2f" % (accuracy * 100))
print('F1 Score: ', "%.2f" % (f1 * 100))

# 输出所选特征的列名
selected_feature_names = data.columns[13:][selected_feature_indices]
print("Selected Features:")
for feature in selected_feature_names:
    print(feature)
'''
