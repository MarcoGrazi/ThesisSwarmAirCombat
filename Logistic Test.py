import numpy as np
import matplotlib.pyplot as plt

def logistic(x, alpha=10, midpoint=0.1):
    return 1 / (1 + np.exp(-alpha * (x - midpoint)))


# Extended X-axis range
x = np.linspace(0, 1, 500)
y = logistic(x, alpha=10, midpoint=0.25)
plt.figure(figsize=(8, 5))
plt.plot(x, y, label='Sigmoid Curve', color='blue')
plt.axvline(0.25, color='gray', linestyle='--', label='Midpoint (0.25)')
plt.title('Full Logistic (Sigmoid) Function Curve')
plt.xlabel('Standard Deviation')
plt.ylabel('Penalty')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()