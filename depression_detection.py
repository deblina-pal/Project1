from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
 
url = r"E:\project_dsa\dataset1.csv"
dataset = pd.read_csv(url) 


X = dataset.iloc[:,1:9].values
t = dataset.iloc[:,9].values
print(type(X),X,"t",t)
X_train, X_test, t_train, t_test = train_test_split(

    X, t, test_size=0.3, random_state=1234)
 
model = tree.DecisionTreeClassifier()

model = model.fit(X_train, t_train)

for i,a in enumerate(model.predict(X)):
    print(dataset["Name"][i],"needs our help. " if(a) else "is doing alright but may need our help in the future. ")

fig = plt.figure(figsize=(30,20))
_ = tree.plot_tree(model)
fig.savefig("decsion_tree.png")
plt.clf()
print("\n")
print("\n")
print("\n")

predicted_value = model.predict(X_test)

print("Accuracy : ", model.score(X_test,t_test))

#---------------------- CFM-------------------------

plt.rcParams.update({'font.size': 30})
cm = confusion_matrix(t_test,predicted_value)
ax = sns.heatmap(cm, annot=True, cmap='Blues')
ax.set_title('Confusion Matrix\n\n',fontsize=30)
plt.savefig('CFM',dpi=300)
plt.clf()

#....................... ROC ........................

y_prob_pred = model.predict_proba(X_test)

fpr = {}
tpr = {}
thresh ={}
fpr, tpr, thresh = roc_curve(t_test, y_prob_pred[:,1])
plt.plot(fpr, tpr,color='green',label='ROC')


plt.plot([0, 1], [0, 1], linestyle='--',color='blue',label='Mean')
plt.title('ROC Curve',fontsize=20)
plt.xlabel('False Positive Rate',fontsize=30)
plt.ylabel('True Positive rate',fontsize=30)
plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.clf()

#--------------------------------------------------

def get_confusion_matrix(y_true, y_pred):
    conf = np.zeros((2, 2))
    for actual, pred in zip(y_true, y_pred):
        conf[int(actual)][int(pred)] += 1
    return conf.astype('int')

conf = get_confusion_matrix(t_test, predicted_value)
print(conf)