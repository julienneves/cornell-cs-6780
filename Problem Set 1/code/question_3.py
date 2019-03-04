import pandas as pd
import numpy as np
import math
from statsmodels.stats.contingency_tables import mcnemar

# Import data
df = pd.read_table("code/hyp_test_pred.txt", header=None)

model = df.loc[:,1:2]
true = df.loc[:,0]

# Check where model is right
ma = (true == model[1])
mb = (true == model[2])

# Print contingency table
table = pd.crosstab(ma,mb)
print(table)

# Part (a)
print("Part (a)")
na = ma.count()
nb = mb.count()

pa = 1 - (ma.sum()/na)
pb = 1 - (mb.sum()/nb)

sa = math.sqrt(na*(1-pa)*pa)
sb = math.sqrt(nb*(1-pb)*pb)

na*pa*(1-pa)
nb*pb*(1-pb)

print('Confidence interval for m_a = %.3f%% ± %.3f%%' % (100*pa,100* 1.96*sa/na))
print('Confidence interval for m_b = %.3f%% ± %.3f%%' % (100*pb, 100*1.96*sb/nb))

# Part (b)
print("Part (b)")
result = mcnemar(table, exact= False, correction =False)
print('statistic = %.3f, p-value = %.3f' % (result.statistic, result.pvalue))

result = mcnemar(table, exact= True)
print('statistic(exact) = %.3f, p-value = %.3f' % (result.statistic, result.pvalue))
