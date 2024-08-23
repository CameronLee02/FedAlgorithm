import pandas as pd
import matplotlib.pyplot as plt

# Load CSV data into a DataFrame
data = pd.read_csv('mnist_limited.csv')

# Plot each overhead metric over epochs
plt.figure(figsize=(10, 6))

plt.plot(data['Epoch'], data['Total Time (seconds)'], label='Total Time')
plt.plot(data['Epoch'], data['Model Distribution Time (seconds)'], label='Model Distribution Time')
plt.plot(data['Epoch'], data['Encryption Time (seconds)'], label='Encryption Time')
plt.plot(data['Epoch'], data['Aggregation Time (seconds)'], label='Aggregation Time')
plt.plot(data['Epoch'], data['Decryption Time (seconds)'], label='Decryption Time')

plt.xlabel('Epoch')
plt.ylabel('Time (seconds)')
plt.title('Overhead Evaluation Per Epoch')
plt.legend()
plt.grid(True)
plt.show()
