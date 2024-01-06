from django.shortcuts import render,redirect
from django.contrib import messages
from django.contrib.auth.models import User,auth

# Create your views here.

def index(request):
    return render(request,"index.html")


def home(request):
    if request.method=="POST":
        nitro=int(request.POST['nitro'])
        phos=int(request.POST['phos'])
        pottas=int(request.POST['pottas'])
        temp=float(request.POST['temp'])
        humid=float(request.POST['humid'])
        ph=float(request.POST['ph'])
        rain=float(request.POST['rain'])
        #Phosphorous=int(request.POST['phos'])
        #from __future__ import print_function
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import classification_report
        from sklearn import metrics
        from sklearn import tree
        import warnings
        df = pd.read_csv('static/Dataset/Crop_recommendation.csv')
        print(df.head())
        print(df.tail())
        print(df.size)
        print(df.columns)
        print(df['label'].unique())
        print(df.dtypes)
        print(df['label'].value_counts())
        #sns.heatmap(df.corr(),annot=True)
        #plt.show()
        features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
        target = df['label']
        labels = df['label']
        acc = []
        model = []
        from sklearn.model_selection import train_test_split
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)
        from sklearn.tree import DecisionTreeClassifier

        DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)

        DecisionTree.fit(Xtrain,Ytrain)

        predicted_values = DecisionTree.predict(Xtest)
        x = metrics.accuracy_score(Ytest, predicted_values)
        acc.append(x)
        model.append('Decision Tree')
        print("DecisionTrees's Accuracy is: ", x*100)

        print(classification_report(Ytest,predicted_values))
        from sklearn.model_selection import cross_val_score
        score = cross_val_score(DecisionTree, features, target,cv=5)
        print(score)
        import pickle
        # Dump the trained Naive Bayes classifier with Pickle
        DT_pkl_filename = 'DecisionTree.pkl'
        # Open the file to save as pkl file
        DT_Model_pkl = open(DT_pkl_filename, 'wb')
        pickle.dump(DecisionTree, DT_Model_pkl)
        # Close the pickle instances
        DT_Model_pkl.close()
        from sklearn.naive_bayes import GaussianNB

        NaiveBayes = GaussianNB()

        NaiveBayes.fit(Xtrain,Ytrain)

        predicted_values = NaiveBayes.predict(Xtest)
        x = metrics.accuracy_score(Ytest, predicted_values)
        acc.append(x)
        model.append('Naive Bayes')
        print("Naive Bayes's Accuracy is: ", x)

        print(classification_report(Ytest,predicted_values))
        score = cross_val_score(NaiveBayes,features,target,cv=5)
        print(score)
        import pickle
        # Dump the trained Naive Bayes classifier with Pickle
        NB_pkl_filename = 'NBClassifier.pkl'
        # Open the file to save as pkl file
        NB_Model_pkl = open(NB_pkl_filename, 'wb')
        pickle.dump(NaiveBayes, NB_Model_pkl)
        # Close the pickle instances
        NB_Model_pkl.close()
        from sklearn.svm import SVC

        SVM = SVC(gamma='auto')

        SVM.fit(Xtrain,Ytrain)

        predicted_values = SVM.predict(Xtest)

        x = metrics.accuracy_score(Ytest, predicted_values)
        acc.append(x)
        model.append('SVM')
        print("SVM's Accuracy is: ", x)

        print(classification_report(Ytest,predicted_values))
        # Cross validation score (SVM)
        score = cross_val_score(SVM,features,target,cv=5)
        print(score)
        from sklearn.linear_model import LogisticRegression

        LogReg = LogisticRegression(random_state=2)

        LogReg.fit(Xtrain,Ytrain)

        predicted_values = LogReg.predict(Xtest)

        x = metrics.accuracy_score(Ytest, predicted_values)
        acc.append(x)
        model.append('Logistic Regression')
        print("Logistic Regression's Accuracy is: ", x)

        print(classification_report(Ytest,predicted_values))
        score = cross_val_score(LogReg,features,target,cv=5)
        print(score)
        import pickle
        # Dump the trained Naive Bayes classifier with Pickle
        LR_pkl_filename = 'LogisticRegression.pkl'
        # Open the file to save as pkl file
        LR_Model_pkl = open(DT_pkl_filename, 'wb')
        pickle.dump(LogReg, LR_Model_pkl)
        # Close the pickle instances
        LR_Model_pkl.close()
        from sklearn.ensemble import RandomForestClassifier

        RF = RandomForestClassifier(n_estimators=20, random_state=0)
        RF.fit(Xtrain,Ytrain)

        predicted_values = RF.predict(Xtest)

        x = metrics.accuracy_score(Ytest, predicted_values)
        acc.append(x)
        model.append('RF')
        print("RF's Accuracy is: ", x)

        print(classification_report(Ytest,predicted_values))
        # Cross validation score (Random Forest)
        score = cross_val_score(RF,features,target,cv=5)
        print(score)
        import pickle
        # Dump the trained Naive Bayes classifier with Pickle
        RF_pkl_filename = 'RandomForest.pkl'
        # Open the file to save as pkl file
        RF_Model_pkl = open(RF_pkl_filename, 'wb')
        pickle.dump(RF, RF_Model_pkl)
        RF_Model_pkl.close()
        # Close the pickle instances
        plt.figure(figsize=[10,5],dpi = 100)
        plt.title('Accuracy Comparison')
        plt.xlabel('Accuracy')
        plt.ylabel('Algorithm')
        sns.barplot(x = acc,y = model,palette='dark') 
        plt.show()
        accuracy_models = dict(zip(model, acc))
        for k, v in accuracy_models.items():
            print (k, '-->', v)
        data = np.array([[nitro,ph,phos,humid,temp,pottas,rain]])            
        prediction = RF.predict(data)
        print(prediction)
        return render(request,"crop_predict.html",{"nitro":nitro,"phos":phos,"ph":ph,"pottas":pottas,"humid":humid,"temp":temp,"rain":rain,
                                                   "pred":prediction})
    else:
        return render(request,"home.html")
    
def crop_predict(request):
    return render(request,"crop_predict.html")

    return render(request,"home.html")