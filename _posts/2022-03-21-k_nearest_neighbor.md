---
layout: single
title:  "k_nearest_neighbor"
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

# K-최근접 이웃법



```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

names = [
         'sepal-length',
         'sepal-width',
         'petal-length',
         'petal-width',
         'Class'
]

dataset = pd.read_csv('./iris.data', names=names)
```


```python
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
s = StandardScaler()
x_train = s.fit_transform(x_train)
x_test = s.fit_transform(x_test)
```


```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(x_train, y_train)
```

<pre>
KNeighborsClassifier(n_neighbors=50)
</pre>

```python
from sklearn.metrics import accuracy_score
y_pred = knn.predict(x_test)
print(f'정확도{accuracy_score(y_test,y_pred)}')
```

<pre>
정확도0.8666666666666667
</pre>

```python
k = 10
acc_array = np.zeros(k)
for k in np.arange(1,k+1, 1):
    classifier = KNeighborsClassifier(n_neighbors=k).fit(x_train,y_train)
    y_pred = classifier.predict(x_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    acc_array[k-1] = acc
max_acc = np.amax(acc_array)
acc_list = list(acc_array)
k = acc_list.index(max_acc)
print("정확도", max_acc,"으로 최적의 k는",k+1,"입니다.")
```

<pre>
정확도 0.8666666666666667 으로 최적의 k는 8 입니다.
</pre>

```python

```
