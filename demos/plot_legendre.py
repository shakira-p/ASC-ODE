#!/usr/bin/env python3
"""
Plot Legendre polynomials and their derivatives from CSV data
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read the CSV data
data = pd.read_csv('demos/data/legendre_autodiff.csv')

# Extract x values
x = data['x'].values

# Create two subplots: one for polynomials, one for derivatives
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot Legendre polynomials P_n(x)
for n in range(6):
    col_name = f'P{n}'
    ax1.plot(x, data[col_name], label=f'$P_{n}(x)$', linewidth=2)

ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('$P_n(x)$', fontsize=12)
ax1.set_title('Legendre Polynomials (order 0-5)', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)

# Plot derivatives P_n'(x)
for n in range(6):
    col_name = f'P{n}_prime'
    ax2.plot(x, data[col_name], label=f"$P'_{n}(x)$", linewidth=2)

ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel("$P'_n(x)$", fontsize=12)
ax2.set_title('Derivatives of Legendre Polynomials (order 0-5)', fontsize=14, fontweight='bold')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)

plt.tight_layout()
plt.savefig('demos/plots/legendre_polynomials.png', dpi=300, bbox_inches='tight')
print("Plot saved to demos/plots/legendre_polynomials.png")
plt.show()

