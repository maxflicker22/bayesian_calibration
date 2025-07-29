import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np
from scipy.stats import norm

def plot_relative_difference_over_runs(data_dir="mcmc_output/model_analysis/experiment_1/model_values", fname_prefix="model_values"):
    # Alle passenden CSV-Dateien finden
    pattern = os.path.join(data_dir, f"{fname_prefix}__*.csv")
    csv_files = sorted(glob.glob(pattern))
    if not csv_files:
        print("Keine CSV-Dateien gefunden.")
        return

    run_indices = []
    rel_diffs = []

    for file in csv_files:
        # Extrahiere die Run-Nummer aus dem Dateinamen
        match = re.search(rf"{fname_prefix}__(\d+)\.csv$", os.path.basename(file))
        if not match:
            print(f"Dateiname {file} passt nicht zum erwarteten Muster.")
            continue
        run_idx = int(match.group(1))

        df = pd.read_csv(file)
        # Erwartet: Spalten 'True_Value' und 'Found_Value'
        if 'True_Value' in df.columns and 'Found_Value' in df.columns:
            rel_diff = abs(100 * (df['Found_Value'].iloc[0] - df['True_Value'].iloc[0]) / df['True_Value'].iloc[0])
            run_indices.append(run_idx)
            rel_diffs.append(rel_diff)
        else:
            print(f"Warnung: Datei {file} enthält nicht die erwarteten Spalten.")

    if not run_indices:
        print("Keine gültigen Daten zum Plotten gefunden.")
        return

     # Nach Run sortieren
    sorted_pairs = sorted(zip(run_indices, rel_diffs))
    runs_sorted, rel_diffs_sorted = zip(*sorted_pairs)

    # Plot mit zwei Achsen: links der Verlauf, rechts die Normalverteilung (gedreht)
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(12, 5), sharey=True, 
        gridspec_kw={'width_ratios': [2.5, 1]}
    )

     # Linker Plot: Verlauf
    ax.plot(runs_sorted, rel_diffs_sorted, marker='o', label=r"$\epsilon_r$ pro Run")
    ax.set_xlabel("Run")
    ax.set_ylabel(r"$\epsilon_r$ [%]")
    ax.set_title(r"Relative Differenz $\epsilon_r$ (Found vs. True) pro Run")
    ax.grid(True)

    yticks = np.linspace(min(rel_diffs_sorted), max(rel_diffs_sorted), 8)
    ax.set_yticks(yticks)

    mu, std = np.mean(rel_diffs_sorted), np.std(rel_diffs_sorted)
    ax.axhline(mu, color='red', linestyle='--', label=fr'Mittelwert: {mu:.2f} %')
    ax.legend(loc="upper left")

    # Rechter Plot: Normalverteilung (gedreht)
    y = np.linspace(min(rel_diffs_sorted), max(rel_diffs_sorted), 200)
    pdf = norm.pdf(y, mu, std)
    ax2.plot(pdf, y, color='orange', label=fr"$\epsilon_r$ ~ $\mathcal{{N}}({mu:.2f}, {std:.2f})$")
    ax2.fill_betweenx(y, 0, pdf, color='orange', alpha=0.3)
    ax2.axhline(mu, color='red', linestyle='--')
    ax2.set_xlabel("Dichte")
    ax2.set_xticks([])
    ax2.set_yticks(yticks)
    ax2.set_title("Normalverteilung", fontsize=10)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    plt.show()
    
    
def plot_config_counts_per_run(data_dir="mcmc_output/model_analysis/experiment_1/posterior_configs", fname_prefix="posterior_configs_within_delta_", delta_label=""):
    """
    Plots the number of configs per run for 'found_impedance' and 'true_impedance' files.
    X-axis: run index (from filename), Y-axis: number of configs (rows in CSV).
    """
    # Suche nach passenden Dateien
    found_pattern = os.path.join(data_dir, f"{fname_prefix}found_impedance_*.csv")
    true_pattern = os.path.join(data_dir, f"{fname_prefix}true_impedance_*.csv")
    found_files = sorted(glob.glob(found_pattern))
    true_files = sorted(glob.glob(true_pattern))

    found_runs = []
    found_counts = []
    for file in found_files:
        match = re.search(rf"found_impedance_(\d+)\.csv$", os.path.basename(file))
        if match:
            run_idx = int(match.group(1))
            df = pd.read_csv(file)
            found_runs.append(run_idx)
            found_counts.append(len(df))
    true_runs = []
    true_counts = []
    for file in true_files:
        match = re.search(rf"true_impedance_(\d+)\.csv$", os.path.basename(file))
        if match:
            run_idx = int(match.group(1))
            df = pd.read_csv(file)
            true_runs.append(run_idx)
            true_counts.append(len(df))

    # Sortieren nach Run-Index
    found_sorted = sorted(zip(found_runs, found_counts))
    true_sorted = sorted(zip(true_runs, true_counts))
    if found_sorted:
        found_runs_sorted, found_counts_sorted = zip(*found_sorted)
    else:
        found_runs_sorted, found_counts_sorted = [], []
    if true_sorted:
        true_runs_sorted, true_counts_sorted = zip(*true_sorted)
    else:
        true_runs_sorted, true_counts_sorted = [], []

    plt.figure(figsize=(10, 5))
    if true_runs_sorted:
        plt.plot(true_runs_sorted, true_counts_sorted, marker='o', label="True Impedance")
    if found_runs_sorted:
        plt.plot(found_runs_sorted, found_counts_sorted, marker='x', label="Found Impedance")
    plt.xlabel("Run")
    plt.ylabel("Anzahl Configs")
    plt.title("Anzahl Configs pro Run im Δ-Intervall")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    plot_relative_difference_over_runs()
    plot_config_counts_per_run()