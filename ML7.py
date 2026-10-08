import pandas as pd
import seaborn as sns
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

# Load Titanic dataset
df = pd.read-csv(Downloads/titanic.csv.csv)   

# Select required columns
df = df[['survived', 'pclass', 'sex', 'age']]

# Remove missing values
df.dropna(inplace=True)

# Convert categorical variables
df['sex'] = df['sex'].astype('category').cat.codes

# Discretize age into bins
df['age'] = pd.cut(df['age'], bins=5, labels=False)
df['age'] = df['age'].astype('category')

# Define Bayesian Network
model = DiscreteBayesianNetwork([
    ('pclass', 'survived'),
    ('sex', 'survived'),
    ('age', 'survived')
])

# Train the model
#model.fit(df, estimator=BayesianEstimator, prior_type='BDeu')
model.fit(df)

# Inference
infer = VariableElimination(model)

result = infer.query(
    variables=['survived'],
    evidence={'sex': 0, 'pclass': 1, 'age': 1}
)

print(result)