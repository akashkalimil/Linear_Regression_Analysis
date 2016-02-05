import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import stats


loansData=pd.read_csv("https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv")



cleanInterestrate=loansData["Interest.Rate"].map(lambda x: (float(x.rstrip("%")))/100)
loansData["Interest.Rate"]=cleanInterestrate
print loansData['Interest.Rate'][0:5]

print ""


cleanloanlength=loansData["Loan.Length"].map(lambda x: x.rstrip(" months"))
loansData["Loan.Length"]=cleanloanlength                                                              
print loansData['Loan.Length'][0:5]
  
print ""

cleanFICORange=loansData["FICO.Range"].map(lambda x: (((str(x)).replace("-"," ")).split()))
cleanFICORange2=cleanFICORange.map(lambda x: [int(n) for n in x])
midFICORange3=cleanFICORange2.map(lambda x: (x[0] + x[1])/2.0)
loansData["FICO.Range"]=midFICORange3
print loansData['FICO.Range'][0:5]



#Create Histogram


plt.figure()
#a = pd.scatter_matrix(loansData, alpha=0.05, figsize=(10,10), diagonal='hist')
#plt.show()

#Linear Regression Analysis
#Formula : InterestRate = b + a1(FICOScore) + a2(LoanAmount)

intrate = loansData["Interest.Rate"]
fico = loansData["FICO.Range"]
loanamt = loansData["Amount.Requested"]



#dependent variable
y=np.matrix(intrate).transpose()


#independent varaiblaes
x1=np.matrix(fico).transpose()
x2=np.matrix(loanamt).transpose()

#combine indendent matrices to 1 matrice
x=np.column_stack([x1,x2])


X=sm.add_constant(x)
model=sm.OLS(y,X)
f=model.fit()
print f.summary()



