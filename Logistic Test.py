import numpy as np
import matplotlib.pyplot as plt

def logistic(x, alpha=10, midpoint=0.1):
    return np.tan(x*(np.pi/3)) / np.tan((np.pi/3))


# Extended X-axis range
x = np.linspace(-1, 1, 700)
y = logistic(x, alpha=8, midpoint=0.45)
plt.figure(figsize=(8, 5))
plt.plot(x, y, label='Sigmoid Curve', color='blue')
plt.axvline(0, color='gray', linestyle='--', label='Midpoint (0)')
plt.title('Full Logistic (Sigmoid) Function Curve')
plt.xlabel('Standard Deviation')
plt.ylabel('Penalty')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()