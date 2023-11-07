
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel, RFE, VarianceThreshold, mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# 读取Excel文件，将'--'视为缺失值
data = pd.read_excel('AD2.xlsx', na_values='--')

# 提取特徵和目標
X = data.iloc[:, 13:]  # Starting from the 6th column as feature variables
y = data['Group']

# 处理缺失值使用SimpleImputer
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 创建SVM分类器并使用最佳参数
best_params = {'C': 10, 'gamma': 'scale', 'kernel': 'poly'}
classifier = svm.SVC(**best_params)

# 训练模型
classifier.fit(X_train_scaled, y_train)

# 定義多個特徵選擇方法
selectors = [
    SelectKBest(score_func=f_classif, k='all'),
    SelectFromModel(svm.SVC(kernel='linear', C=1.0)),
    RFE(estimator=svm.SVC(kernel='linear', C=1.0), n_features_to_select=10),
    VarianceThreshold(threshold=0.1),
    SelectKBest(score_func=mutual_info_classif, k='all'),
    # 添加更多的特徵選擇方法
]

weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]  # 根據性能調整權重
selected_features_train = []
selected_features_test = []

for i, selector in enumerate(selectors):
    if isinstance(selector, SelectFromModel):
        # 使用 SelectFromModel
        selector.fit(X_train_scaled, y_train)
        feature_support = selector.get_support()  # 获取所选特徵的布尔遮罩
        selected_features_train.append(X_train_scaled[:, feature_support])
        selected_features_test.append(X_test_scaled[:, feature_support])
    elif isinstance(selector, RFE):
        # 使用 RFE
        selector.fit(X_train_scaled, y_train)
        feature_ranking = selector.ranking_
        # 選擇前k個最重要的特徵（這裡選擇前10個特徵）
        k = 10
        feature_ranking = np.argsort(feature_ranking)
        selected_features_train.append(X_train_scaled[:, feature_ranking[:k]])
        selected_features_test.append(X_test_scaled[:, feature_ranking[:k]])
    else:
        # 使用其他特徵選擇方法
        selector.fit(X_train_scaled, y_train)
        if hasattr(selector, 'scores_'):
            feature_scores = selector.scores_
        else:
            feature_scores = selector.variances_  # 對於VarianceThreshold，使用方差
        weighted_scores = feature_scores * weights[i]

        # 基於權重和重要性得分對特徵進行排序
        feature_ranking = weighted_scores.argsort()[::-1]

        # 選擇前k個最重要的特徵（這裡選擇前10個特徵）
        k = 10
        selected_features_train.append(X_train_scaled[:, feature_ranking[:k]])
        selected_features_test.append(X_test_scaled[:, feature_ranking[:k]])

# 將選定的特徵合併
X_train_selected = np.concatenate(selected_features_train, axis=1)
X_test_selected = np.concatenate(selected_features_test, axis=1)

# 使用選定的特徵子集來訓練模型並進行預測
classifier.fit(X_train_selected, y_train)
y_pred = classifier.predict(X_test_selected)

# 計算模型性能
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print('Accuracy:', "%.2f" % (accuracy * 100))
print('F1 Score:', "%.2f" % (f1 * 100))
print('Precision Score:', "%.2f" % (precision * 100))
print('Recall Score:', "%.2f" % (recall * 100))