import numpy as np
import pandas as pd
 
fp="Entropy_Hpatches.xlsx"

data=pd.read_excel(fp,index_col=None,header=None)
data1 = data.iloc[:,:].values
# Normalization

data = (data.max() - data)/(data.max() - data.min())
m,n=data.shape

 
data=data.iloc[:,:].values
 
k=1/np.log(m)
print(m)
yij=data.sum(axis=0)
pij=data/yij

#Calculate pij
test=pij*np.log(pij)
test=np.nan_to_num(test)
ej=-k*(test.sum(axis=0))

#Calculate Entropy
 
wi=(1-ej)/(4-np.sum(ej))
 
print('wi:',wi)
 
result=[]

for i in range(m):
    a = 0
    for j in range(n):
        a = a+data1[i][j]*wi[j]
    result.append(a)

print(result)
 
