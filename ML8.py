import pandas as pd
import seaborn as sns
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

import pandas as pd
import seaborn as sns

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination

df = sns.load_dataset('titanic')
df = df[['survived', 'pclass', 'sex', 'age']]
df.dropna(inplace=True)

df['age'] = pd.cut(
    df['age'],
    bins=[0, 12, 30, 50, 100],
    labels=['child', 'young', 'adult', 'senior']
)

df['sex'] = df['sex'].astype('category').cat.codes
df['age'] = df['age'].astype('category').cat.codes

model = DiscreteBayesianNetwork([
    ('pclass', 'survived'),
    ('sex', 'survived'),
    ('age', 'survived')
])

model.fit(df)

inference = VariableElimination(model)

result1 = inference.query(
    variables=['survived'],
    evidence={'pclass': 1, 'sex': 0, 'age': 1}
)
print("Query 1 - 1st class, female, young:")
print(result1)

result2 = inference.query(
    variables=['survived'],
    evidence={'pclass': 3, 'sex': 1, 'age': 2}
)
print("\nQuery 2 - 3rd class, male, adult:")
print(result2)

result3 = inference.query(
    variables=['survived'],
    evidence={'pclass': 2, 'sex': 1, 'age': 3}
)
print("\nQuery 3 - 2nd class, male, senior:")
print(result3)

result4 = inference.query(
    variables=['survived'],
    evidence={'pclass': 1, 'sex': 0, 'age': 0}
)
print("\nQuery 4 - 1st class, female, child:")
print(result4)
