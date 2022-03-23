---
layout: single
title:  "Principal_Component_Analysis"
categories: ML
tag: [python, ml, PCA]
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


#주성분 분석(Principal_Component_Analysis)

#####고차원 데이터를 저차원 데이터로 축소시키는 알고리즘

######출처: 딥러닝with텐서플로우 교과서



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
```


```python
x = pd.read_csv('./credit card.csv')
x = x.drop('CUST_ID', axis=1)
x.fillna(method='ffill',inplace=True) ## 같은 열 앞의 값으로 결측치를 대체
print(x.head())
x.shape ##축 확인
```

<pre>
       BALANCE  BALANCE_FREQUENCY  PURCHASES  ONEOFF_PURCHASES  \
0    40.900749           0.818182      95.40              0.00   
1  3202.467416           0.909091       0.00              0.00   
2  2495.148862           1.000000     773.17            773.17   
3  1666.670542           0.636364    1499.00           1499.00   
4   817.714335           1.000000      16.00             16.00   

   INSTALLMENTS_PURCHASES  CASH_ADVANCE  PURCHASES_FREQUENCY  \
0                    95.4      0.000000             0.166667   
1                     0.0   6442.945483             0.000000   
2                     0.0      0.000000             1.000000   
3                     0.0    205.788017             0.083333   
4                     0.0      0.000000             0.083333   

   ONEOFF_PURCHASES_FREQUENCY  PURCHASES_INSTALLMENTS_FREQUENCY  \
0                    0.000000                          0.083333   
1                    0.000000                          0.000000   
2                    1.000000                          0.000000   
3                    0.083333                          0.000000   
4                    0.083333                          0.000000   

   CASH_ADVANCE_FREQUENCY  CASH_ADVANCE_TRX  PURCHASES_TRX  CREDIT_LIMIT  \
0                0.000000                 0              2        1000.0   
1                0.250000                 4              0        7000.0   
2                0.000000                 0             12        7500.0   
3                0.083333                 1              1        7500.0   
4                0.000000                 0              1        1200.0   

      PAYMENTS  MINIMUM_PAYMENTS  PRC_FULL_PAYMENT  TENURE  
0   201.802084        139.509787          0.000000      12  
1  4103.032597       1072.340217          0.222222      12  
2   622.066742        627.284787          0.000000      12  
3     0.000000        627.284787          0.000000      12  
4   678.334763        244.791237          0.000000      12  
</pre>
<pre>
(8950, 17)
</pre>

```python
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x) ##정규화

x_normalized = normalize(x_scaled) ## 가우스 분포를 따르도록 정규화
x_normalized = pd.DataFrame(x_normalized)

pca = PCA(n_components=2) ##2차원으로 차원 축소 모델
x_principal = pca.fit_transform(x_normalized) ## x_normalized 차원 축소 
x_principal = pd.DataFrame(x_principal)
x_principal.columns = ['P1','P2']
print(x_principal.head())
x_principal.shape ##축 확인
```

<pre>
         P1        P2
0 -0.489949 -0.679976
1 -0.519099  0.544829
2  0.330633  0.268878
3 -0.481656 -0.097607
4 -0.563512 -0.482506
</pre>
<pre>
(8950, 2)
</pre>

```python
db_default = DBSCAN(eps=0.0375, min_samples=3).fit(x_principal)
labels = db_default.labels_

colours = {}
colours[0]='y'
colours[1]='g'
colours[2]='b'
colours[-1]='k'

cvec = [colours[label] for label in labels]

## 플롯의 범례 구성
r = plt.scatter(x_principal['P1'],x_principal['P2'], color='y')
g = plt.scatter(x_principal['P1'],x_principal['P2'],color='g')
b = plt.scatter(x_principal['P1'],x_principal['P2'],color='b')
k = plt.scatter(x_principal['P1'],x_principal['P2'], color='k') 

plt.figure(figsize=(9,9))
plt.scatter(x_principal['P1'],x_principal['P2'],c=cvec) ## c는 색상

plt.legend((r,g,b,k),('Label 0','Label 1','Label 2','Label -1')) ## 범례 지정
plt.show()
```

<pre>
<Figure size 432x288 with 1 Axes>
</pre>
<pre>
<Figure size 648x648 with 1 Axes>
</pre>

```python
db_default = DBSCAN(eps=0.0375, min_samples=50).fit(x_principal)
labels = db_default.labels_

colours1 = {}
colours1[0] = 'r'
colours1[1] = 'g'
colours1[2] = 'b'
colours1[3] = 'c'
colours1[4] = 'y'
colours1[5] = 'm'
colours1[-1] = 'k'

cvec = [colours1[label] for label in labels]
colors1 = ['r', 'g', 'b', 'c', 'y', 'm', 'k']

## 플롯의 범례 구성
r = plt.scatter(x_principal['P1'], x_principal['P2'], marker='o', color=colors1[0])
g = plt.scatter(x_principal['P1'], x_principal['P2'], marker='o', color=colors1[1])
b = plt.scatter(x_principal['P1'], x_principal['P2'], marker='o', color=colors1[2])
c = plt.scatter(x_principal['P1'], x_principal['P2'], marker='o', color=colors1[3])
y = plt.scatter(x_principal['P1'], x_principal['P2'], marker='o', color=colors1[4])
m = plt.scatter(x_principal['P1'], x_principal['P2'], marker='o', color=colors1[5])
k = plt.scatter(x_principal['P1'], x_principal['P2'], marker='o', color=colors1[6])

plt.figure(figsize=(9,9))
plt.scatter(x_principal['P1'],x_principal['P2'],c=cvec) ## c는 색상

plt.legend((r, g, b, c, y, m, k),
          ('Label 0', 'Label 1', 'Label 2', 'Label 3', 'Label 4', 'Label 5', 'Label -1'),
          scatterpoints=1,
          loc='upper right',
          ncol=3,
          fontsize=8) ## 범례 지정
plt.show()
```

<pre>
<Figure size 432x288 with 1 Axes>
</pre>
<pre>
<Figure size 648x648 with 1 Axes>
</pre>