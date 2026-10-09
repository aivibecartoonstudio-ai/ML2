import pandas as pd
import seaborn as sns
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination


df = pd.read-csv(Downloads/titanic.csv.csv)   

df = df[['survived', 'pclass', 'sex', 'age']]


df.dropna(inplace=True)
df['sex'] = df['sex'].astype('category').cat.codes
df['age'] = pd.cut(df['age'], bins=5, labels=False)
df['age'] = df['age'].astype('category')
model = DiscreteBayesianNetwork([
    ('pclass', 'survived'),
    ('sex', 'survived'),
    ('age', 'survived')
])

#model.fit(df, estimator=BayesianEstimator, prior_type='BDeu')
model.fit(df)

infer = VariableElimination(model)
result = infer.query(
    variables=['survived'],
    evidence={'sex': 0, 'pclass': 1, 'age': 1}
)

print(result)
