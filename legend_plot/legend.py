import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# # Define the elements used in the legend.
# handles = [
#     Line2D([0], [0], color='orange', lw=2, ls='--', marker='o', markersize=6,
#            label=r'Eq. (14) $\Delta\tilde{T}_{c}^{+}$'),
#     Line2D([0], [0], color='green', lw=2, ls='--', marker='s', markersize=6,
#            label=r'Eq. (14) $\Delta\tilde{T}_{c}^{-}$'),
#     Line2D([0], [0], color='red', lw=2, marker='o', markersize=6,
#            label=r'Eq. (13) $\Delta \tilde{T}_{c}^{+}$'),
#     Line2D([0], [0], color='blue', lw=2, marker='s', markersize=6,
#            label=r'Eq. (13) $\Delta \tilde{T}_{c}^{-}$'),
#     Line2D([0], [0], color='purple', lw=0, marker='x', markersize=10,
#            label=r'Eqs. (6) and (7) $\Delta \tilde{T}_{c}^{+}$'),
#     Line2D([0], [0], color='black', lw=0, marker='^', markersize=8,
#            label=r'Eqs. (6) and (7) $\Delta \tilde{T}_{c}^{-}$'),
# ]

# Define the elements used in the legend.
handles = [
    Line2D([0], [0], color='orange', lw=2, ls='--', marker='o', markersize=6,
           label=r'approx. analytical solution $\Delta\tilde{T}_{c}^{+}$'),
    Line2D([0], [0], color='green', lw=2, ls='--', marker='s', markersize=6,
           label=r'approx. analytical solution $\Delta\tilde{T}_{c}^{-}$'),
    Line2D([0], [0], color='red', lw=2, marker='o', markersize=6,
           label=r'numerical solution $\Delta \tilde{T}_{c}^{+}$'),
    Line2D([0], [0], color='blue', lw=2, marker='s', markersize=6,
           label=r'numerical solution $\Delta \tilde{T}_{c}^{-}$'),
    Line2D([0], [0], color='purple', lw=0, marker='x', markersize=10,
           label=r'numerical estimate $\Delta \tilde{T}_{c}^{+}$'),
    Line2D([0], [0], color='black', lw=0, marker='^', markersize=8,
           label=r'numerical estimate $\Delta \tilde{T}_{c}^{-}$'),
    Line2D([0], [0], color='deepskyblue', lw=0, marker='D', markersize=8,
           markeredgecolor='black', markeredgewidth=0.8,
           label='Experiment [9]'),
]

# Empty figure containing only the legend.
fig, ax = plt.subplots(figsize=(4, 3))
ax.axis('off')

legend = ax.legend(
    handles=handles,
    loc='center',
    frameon=True,      # Show the frame.
    fancybox=False,    # Use square corners.
    framealpha=1.0,
    ncol=1,
    fontsize=14,
    handlelength=2.3,
    handletextpad=0.8,
    borderpad=0.8,
    labelspacing=0.7
)

# Save as PNG with a transparent background.
fig.savefig(
    "common_legend.png",
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.05,
    transparent=True
)

# Also save as PDF for use in papers.
fig.savefig(
    "common_legend.pdf",
    bbox_inches='tight',
    pad_inches=0.05,
    transparent=True
)

plt.show()
