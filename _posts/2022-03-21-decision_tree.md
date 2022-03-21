---
layout: single
title:  "decision_tree"
categories: coding
tag: [python, sklearn]
toc: true
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


-딥러닝 텐서플로 교과서-

#결정 트리

#### 영역의 순도는 증가, 불순도와 불확실성은 감소하는 방향으로 학습 진행

###정보획득: 

#####순도가 증가, 불확실성이 감소하는 것(정보 이론에서).

###엔트로피:

#####확률 변수의 불확실성을 수치로 나타낸것, 높을수록 불확실성이 높다는 의미

###지니 계수:

#####분산 정도를 정량화해서 표현한 값, 수가 높을수록 더 분산되에 있음을 의미



```python
import pandas as pd
df = pd.read_csv('./train.csv', index_col='PassengerId')
print(df.head())
```

<pre>
             Survived  Pclass  \
PassengerId                     
1                   0       3   
2                   1       1   
3                   1       3   
4                   1       1   
5                   0       3   

                                                          Name     Sex   Age  \
PassengerId                                                                    
1                                      Braund, Mr. Owen Harris    male  22.0   
2            Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0   
3                                       Heikkinen, Miss. Laina  female  26.0   
4                 Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0   
5                                     Allen, Mr. William Henry    male  35.0   

             SibSp  Parch            Ticket     Fare Cabin Embarked  
PassengerId                                                          
1                1      0         A/5 21171   7.2500   NaN        S  
2                1      0          PC 17599  71.2833   C85        C  
3                0      0  STON/O2. 3101282   7.9250   NaN        S  
4                1      0            113803  53.1000  C123        S  
5                0      0            373450   8.0500   NaN        S  
</pre>

```python
df = df[['Pclass','Sex','Age','SibSp','Parch','Fare','Survived']]
df['Sex'] = df['Sex'].map({'male':0,'female':1})
df = df.dropna()
x = df.drop('Survived',axis=1)
y = df['Survived']
```

<pre>
/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:2: SettingWithCopyWarning: 
A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
  
</pre>

```python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 1)
```


```python
from sklearn import tree
model = tree.DecisionTreeClassifier()
```


```python
model.fit(x_train, y_train)
```

<pre>
DecisionTreeClassifier()
</pre>

```python
y_predict = model.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predict)
```

<pre>
0.8181818181818182
</pre>

```python
from sklearn.metrics import confusion_matrix
pd.DataFrame(
    confusion_matrix(y_test,y_predict),
    columns=['Predicted Not Survival','Predicted Survival'],
    index=['True Not Survival','True Survival']
)
```

<pre>
                   Predicted Not Survival  Predicted Survival
True Not Survival                      72                  14
True Survival                          12                  45
</pre>

```python

```
