#new data:
LG: Accuracy: 60.81
    F1 Score: 59.67
    Precision Score: 59.60
    Recall Score: 60.81
Svm: Accuracy： 65.25
     F1 Score： 62.46
     Precision Score： 65.11
     Recall Score： 65.25
     
#voting:
svm。rf: Accuracy:  65.45 F1 Score:  63.78 Precision Score： 64.90 Recall Score： 65.45
svm。rf、lr: Accuracy： 66.26 F1 Score： 63.23 Precision Score： 66.85 Recall Score： 66.26
svm。rf、lr、knn: Accuracy： 66.26 F1 Score： 64.65 Precision Score： 65.52 Recall Score： 66.26
svm。rf、lr、knn、gb: Accuracy： 66.87 F1 Score： 63.47 Precision Score： 67.37 Recall Score： 66.87
svm。rf、lr、knn、gb、dt:Accuracy： 66.26 F1 Score： 63.99 Precision Score： 65.94 Recall Score： 66.26


#stacking:
svm。rf、lr、knn:
    Accuracy: 68.48
    F1 Score: 66.61
    Precision Score: 68.45
    Recall Score: 68.48

svm。rf、lr、knn、gb:
    Accuracy: 68.08
    F1 Score: 66.23
    Precision Score: 67.99
    Recall Score: 68.08

svm。rf、lr、knn、gb、dt:
    Accuracy: 69.09
    F1 Score: 67.33
    Precision Score: 68.95
    Recall Score: 69.09

#1005


加上 AUC、圖形 
MCI切半(依照年齡) 
搞懂各種模型方法的意義、問題(找文獻)

#1017
#stacking
NC、AD
資料改動之前:
Accuracy: 66.67
F1 Score: 66.67
Precision Score: 66.67
Recall Score: 66.67

改動後選半:
Accuracy: 66.92
F1 Score: 66.89
Precision Score: 67.15
Recall Score: 66.92

改動後選後半(888 開始):
Accuracy: 64.66
F1 Score: 64.43
Precision Score: 65.45
Recall Score: 64.66

改動後選後半(889 開始):
Accuracy: 71.43
F1 Score: 71.43
Precision Score: 71.51
Recall Score: 71.43

改動後MCI全選:
Accuracy: 70.50
F1 Score: 65.75
Precision Score: 69.89
Recall Score: 70.50

ALL
Accuracy: 60.00
F1 Score: 60.05
Precision Score: 60.37
Recall Score: 60.00#



