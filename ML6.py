import random
# Function to generate uniform random numbers
def generate_uniform_samples(n):
 return [random.uniform(0, 1) for _ in range(n)]
# Define function f(x) = x^2
def f(x):
 return x**2
# Generate 10 samples
samples = generate_uniform_samples(10)
# Evaluate f(x) for each sample
function_values = [f(x) for x in samples]
# Estimate the integral using Monte Carlo
integral_estimate = sum(function_values) / len(function_values)
# Print results
print("Random Samples (x):")
print([round(x, 4) for x in samples])
print("\nFunction Values f(x) = x^2:")
print([round(val, 4) for val in function_values])
print(f"\nEstimated Integral I ≈ {round(integral_estimate, 4)}")