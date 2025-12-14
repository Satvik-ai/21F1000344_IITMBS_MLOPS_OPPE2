import pandas as pd
import numpy as np

# Number of rows to generate
N_ROWS = 100

# Random data generation based on realistic heart dataset ranges
data = {
    "age": np.random.randint(29, 77, N_ROWS),
    "gender": np.random.choice([1, 0], N_ROWS),
    "cp": np.random.randint(0, 4, N_ROWS),
    "trestbps": np.random.randint(90, 200, N_ROWS),
    "chol": np.random.randint(120, 570, N_ROWS),
    "fbs": np.random.randint(0, 2, N_ROWS),
    "restecg": np.random.randint(0, 2, N_ROWS),
    "thalach": np.random.randint(70, 210, N_ROWS),
    "exang": np.random.randint(0, 2, N_ROWS),
    "oldpeak": np.round(np.random.uniform(0.0, 6.5, N_ROWS), 1),
    "slope": np.random.randint(0, 3, N_ROWS),
    "ca": np.random.randint(0, 4, N_ROWS),
    "thal": np.random.randint(0, 4, N_ROWS),
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("data/generated_test_data.csv", index=False)

print("generated_test_data.csv created with 100 random rows.")