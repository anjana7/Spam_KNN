from __future__ import print_function
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

features = []
output = []
words = []
word_list = []
roc_dict = {}

#creating a bag of words for dictionary
def make_dict(emails):
    size = len(emails)
    for m in range(size):
        mail = emails["data"][m]
        tokens = nltk.word_tokenize(mail)
            
        for t in tokens:
            if t.isalpha() == False:
                del t
            elif len(t) == 1:
                del t
            else:
                words.append(t)

    #remove stopwords
    filtered = [w for w in words if not w in stop_words]
        
    #lemmatize words
    dictn = [lemmatizer.lemmatize(f) for f in filtered]
        
    #find distinct words
    sets = set(dictn) 
    return(sets)


#defining feature vectors  
def trainset(emails, diction):
    size = len(emails)
    classification = []
    for m in range(size):
        clas = emails["classes"][m]
        mail = emails["data"][m]
        featurev = []
         
        if clas =='spam' :
            classification.append('spam')
        else:
            classification.append('ham')
        
        tokens = nltk.word_tokenize(mail)
        #feature vector for each mail
        for t in diction:
            if t in tokens:
                featurev.append(1)
            else:
                featurev.append(0)
        
        features.append(featurev) 
       
    return(features, classification)

def inputv(email1, diction):
    featuretest = [] 
    tokens = nltk.word_tokenize(email1)
    filtered = [w for w in tokens if not w in stop_words]
    words = [lemmatizer.lemmatize(f) for f in filtered]
    wordl=set(words)
    for t in diction:
        if t in wordl:
            featuretest.append(1)
        else:
            featuretest.append(0)
    
    return(featuretest) 
    
def get_train_data(lines):
    val = lines.split(',')
    clas = val[0]
    line = ' '.join(val[1:])
    return(line, clas)
   
def training_function(features,classification,inputf):
    a = [-1,-1,-1,-1,-1]
    result = ['null','null','null','null','null']
    for k in range(len(features)):
        dot_product = np.dot(features[k],inputf)
        norm_v1 = np.linalg.norm(features[k])
        norm_v2 = np.linalg.norm(inputf)
        res = (dot_product/(norm_v1 * norm_v2))
        
        if(min(a)<res):
            index_min = np.argmin(a)
            a[index_min]=res
            result[index_min]=classification[k]
        
    if (result.count('spam')>result.count('ham')):
        result_test = 'spam'
    else:
        result_test = 'ham'
    return(result_test)
  

              
def algo_performance(confusion_matrix):
    SP = confusion_matrix['TP']/(confusion_matrix['TP']+confusion_matrix['FP'])
    SR = confusion_matrix['TP']/(confusion_matrix['TP']+confusion_matrix['FN'])
    A = (confusion_matrix['TP']+confusion_matrix['TN'])/(confusion_matrix['TP']+confusion_matrix['FP']+confusion_matrix['TN']+confusion_matrix['FN'])
    True_Positives = confusion_matrix['TP']/(confusion_matrix['TP']+confusion_matrix['FN'])
    False_Positives = confusion_matrix['FP']/(confusion_matrix['FP']+confusion_matrix['TN'])
    print('Precision = ' ,SP)
    print('Recall = ',SR )
    print('Accuracy = ',A )
    print('True_Positives = ',True_Positives)
    print('False_Positives = ',False_Positives)
    total = confusion_matrix['FN']+confusion_matrix['FP']+confusion_matrix['TP']+confusion_matrix['TN']
    print('testset = ', total)
    return(True_Positives, False_Positives)
    
#ROC Curve
    
def roc_curve(fpr,tpr):
    print('ROC Curve for KNN')

    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


email = pd.read_csv('/home/anjana/MLprograms/testspam.csv', encoding='latin-1')
emails = email.loc[:, ~email.columns.str.contains('^Unnamed')]   
diction = make_dict(emails)
features, classification = trainset(emails, diction) 
confusion_matrix = {"TP":0, "FP":0, "FN":0, "TN":0} 

addr = '/home/anjana/MLprograms/'
emails = [addr+'test1.csv', addr+'test2.csv', addr+'test3.csv', addr+'test4.csv', addr+'test5.csv', addr+'test6.csv']

for email in emails:
    with open(email) as f:
        line  = f.readlines()
        for lines in line:
            SPAM = False
            HAM = False
            lines, clas = get_train_data(lines)
            if(clas == "ham"):
                HAM = True
            else:
                SPAM = True
            
            inputf = inputv(lines,diction)
            result_test = training_function(features,classification,inputf)
            
            if(result_test=="spam"):
                if (SPAM == True):
                    confusion_matrix['TP']+=1
                else:
                    confusion_matrix['FP']+=1
            else:
                if(SPAM == True):
                    confusion_matrix['FN']+=1
                else:
                    confusion_matrix['TN']+=1             
        print()
        print()
        print("confusion matrix for KNN")
        print(confusion_matrix)
        tpr1,fpr1 = algo_performance(confusion_matrix)
        
        roc_dict[fpr1] = tpr1
        
fpr = roc_dict.keys()
fpr = sorted(fpr)
tpr = []
for key in fpr:
    tpr.append(roc_dict[key])

roc_curve(fpr,tpr)

