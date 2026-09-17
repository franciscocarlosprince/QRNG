import numpy as np
import matplotlib.pyplot as plt

# =========================
# Configurações
# =========================
from pathlib import Path

base = Path(r"C:\Users\Francisco C. Prince\OneDrive\Paper QRNG\Data_fromChip")
arquivo1 = base / "ClassicalNoiseDark.csv"
arquivo2 = base / "totalNoise7mW.csv"

print("Arquivo1 existe?", arquivo1.exists(), arquivo1)
print("Arquivo2 existe?", arquivo2.exists(), arquivo2)


delimiter = ','
skiprows = 0
bins = 4096
coluna_amplitude = 1  # segunda coluna

# =========================
# Funções
# =========================
def carregar_amplitude(csv_path):
    return np.loadtxt(
        csv_path,
        delimiter=delimiter,
        usecols=(coluna_amplitude,),
        skiprows=skiprows
    )

def centralizar_por_media(amp):
    offset = np.mean(amp)  # centraliza pelo zero via média
    return amp - offset

def hist_prob(amp, bins, vmin, vmax):
    counts, bin_edges = np.histogram(amp, bins=bins, range=(vmin, vmax))
    total = counts.sum()
    prob = counts / total if total > 0 else counts
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return centers, prob

# =========================
# Carregar e centralizar
# =========================
amp1 = carregar_amplitude(arquivo1)
amp2 = carregar_amplitude(arquivo2)

amp1_c = centralizar_por_media(amp1)
amp2_c = centralizar_por_media(amp2)

# Range comum (após centralizar)
vmin = min(amp1_c.min(), amp2_c.min())
vmax = max(amp1_c.max(), amp2_c.max())

if vmin == vmax:
    raise ValueError("Após centralizar, amplitude mínima e máxima ficaram iguais. Verifique os dados.")

# =========================
# Histograma probabilidade
# =========================
x1, p1 = hist_prob(amp1_c, bins, vmin, vmax)
x2, p2 = hist_prob(amp2_c, bins, vmin, vmax)

# =========================
# Plot comparativo
# =========================

# Máscaras: só pontos com probabilidade > 0
mask1 = p1 > 0
mask2 = p2 > 0

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(x1[mask1], p1[mask1], s=10, alpha=0.6,
           label="Classical Noise", edgecolors='none')
ax.scatter(x2[mask2], p2[mask2], s=10, alpha=0.6,
           label="Total Noise", edgecolors='none')

ax.axvline(0, color='k', linestyle='--', linewidth=1, alpha=0.7)

ax.set_xlabel("Amplitude (V)")
ax.set_ylabel("Probabilidade por bin (%)")
ax.set_title("Histograma de probabilidade (4096 bins) — comparação (pontos)")

ax.grid(True, which='major', alpha=0.2)
ax.legend()
##plt.tight_layout()
plt.show()
