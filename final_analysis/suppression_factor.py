import numpy as np
import matplotlib.pyplot as plt

data = np.load('list.npy')

energies = np.unique(data[:, 0])
cms = np.unique(data[:, 1])
suppression_values = np.linspace(0, 1, 32)

for energy in energies:
    plt.figure(figsize=(12, 10))

    # Create 3 subplots: Entries, Mean, StdDev
    ax_entries = plt.subplot(3, 1, 1)
    ax_mean = plt.subplot(3, 1, 2)
    ax_std = plt.subplot(3, 1, 3)

    for cm in cms:
        mask = (data[:, 0] == energy) & (data[:, 1] == cm)
        subset = data[mask]
        if subset.size == 0:
            continue
        subset = subset[np.argsort(subset[:, 2])]
        suppress_indices = subset[:, 2].astype(int)
        x = suppression_values[suppress_indices]

        r_entries = subset[:, 3]
        r_mean = subset[:, 4]
        r_std = subset[:, 5]

        g_entries = subset[:, 6]
        g_mean = subset[:, 7]
        g_std = subset[:, 8]

        label_r = f'RANSAC CM={cm}'
        label_g = f'GMM CM={cm}'

        # Plot Entries
        ax_entries.plot(x, r_entries, 'o-', label=label_r)
        ax_entries.plot(x, g_entries, 's--', label=label_g)

        # Plot Mean
        ax_mean.plot(x, r_mean, 'o-', label=label_r)
        ax_mean.plot(x, g_mean, 's--', label=label_g)

        # Plot StdDev
        ax_std.plot(x, r_std, 'o-', label=label_r)
        ax_std.plot(x, g_std, 's--', label=label_g)

    ax_entries.set_ylabel('Entries')
    ax_entries.set_title(f'Entries vs Suppression Factor (Energy={energy})')
    ax_entries.grid(True)
    ax_entries.legend(fontsize='small', ncol=2)

    ax_mean.set_ylabel('Mean')
    ax_mean.set_title(f'Mean vs Suppression Factor (Energy={energy})')
    ax_mean.grid(True)
    ax_mean.legend(fontsize='small', ncol=2)

    ax_std.set_xlabel('Suppression Factor')
    ax_std.set_ylabel('StdDev')
    ax_std.set_title(f'StdDev vs Suppression Factor (Energy={energy})')
    ax_std.grid(True)
    ax_std.legend(fontsize='small', ncol=2)

    plt.tight_layout()
    plt.show()
