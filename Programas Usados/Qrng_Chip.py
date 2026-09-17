import numpy as np
import pandas as pd
from scipy.stats import entropy as shannon_entropy
from scipy.signal import fftconvolve
import secrets

# =========================================================
# 1️⃣ LEITURA E PRÉ-PROCESSAMENTO
# =========================================================
CAMINHO_CSV = r'C:\Users\Francisco C. Prince\OneDrive\Paper QRNG\Data_fromChip\qrng20260916.csv'
COLUNA = 1

df = pd.read_csv(CAMINHO_CSV)
dados_brutos = df.iloc[:, COLUNA].values.astype(np.float64)
N = len(dados_brutos)
print(f"📥 Amostras lidas: {N}")

dados = dados_brutos - np.mean(dados_brutos)
print(f"   Desvio padrão: {np.std(dados):.6f}")
print(f"   Faixa: [{dados.min():.4f}, {dados.max():.4f}]")

# =========================================================
# 2️⃣ VALIDAÇÃO DA DIGITALIZAÇÃO (detecção do passo do ADC)
# =========================================================
valores_unicos = np.unique(dados_brutos)
diffs = np.diff(valores_unicos)
step = np.min(diffs[diffs > 1e-12])   # menor espaçamento não-nulo ≈ 1 LSB

faixa_total = dados_brutos.max() - dados_brutos.min()
n_niveis_estimado = faixa_total / step
n_bits_estimado = np.log2(n_niveis_estimado + 1)

print(f"\n🔍 Validação da digitalização:")
print(f"   Passo de quantização detectado: {step:.8f}")
print(f"   Nº de níveis estimado: {n_niveis_estimado:.1f}")
print(f"   Nº de bits estimado: {n_bits_estimado:.2f}  (esperado: 12)")

if not (11.5 <= n_bits_estimado <= 12.5):
    print("⚠️  ATENÇÃO: resolução detectada NÃO bate com ADC de 12 bits!")
    print("    Verifique se o CSV contém valores calibrados/filtrados.")
    print("    Prosseguindo com o passo detectado mesmo assim.")

# Reconstrução dos códigos digitais inteiros (0 a ~4095)
codigos = np.round((dados_brutos - dados_brutos.min()) / step).astype(np.int64)
print(f"   Códigos reconstruídos: min={codigos.min()}, max={codigos.max()}, "
      f"únicos={len(np.unique(codigos))}")

# =========================================================
# 3️⃣ EXTRAÇÃO DE BITS BRUTOS — MÉTODO B (LSB)
# =========================================================
NUM_LSB = 6   # <<< AJUSTE AQUI: quantos bits menos significativos usar

print(f"\n📊 Extraindo {NUM_LSB} LSB(s) por amostra...")

lsb_matrix = np.zeros((N, NUM_LSB), dtype=np.uint8)
for k in range(NUM_LSB):
    lsb_matrix[:, k] = (codigos >> k) & 1

# Intercala bits por amostra: [b0_amostra0, b1_amostra0, b0_amostra1, ...]
bits_brutos = lsb_matrix.reshape(-1)
N_total_bruto = len(bits_brutos)

print(f"📦 Total de bits brutos: {N_total_bruto} (N={N} × {NUM_LSB} LSB)")
print(f"   P(1) geral = {np.mean(bits_brutos):.6f}  (ideal: 0.5)")

# Checagem rápida por posição de bit (bit 0 tende a ser mais "puro" que bits acima)
print("\n   🔎 Qualidade por posição de bit:")
for k in range(NUM_LSB):
    p1_k = np.mean(lsb_matrix[:, k])
    desvio = abs(p1_k - 0.5)
    alerta = "⚠️" if desvio > 0.02 else "✅"
    print(f"      Bit {k}: P(1) = {p1_k:.6f}  {alerta}")

# =========================================================
# 4️⃣ ESTIMATIVA DE MIN-ENTROPIA (scipy.stats.entropy)
# =========================================================
def estimar_entropias(bits):
    contagem = np.bincount(bits, minlength=2)
    p = contagem / len(bits)
    p_nz = p[p > 0]
    h_shannon = shannon_entropy(p_nz, base=2)   # entropia de Shannon (cota superior)
    h_min = -np.log2(np.max(p))                  # min-entropia (conservadora)
    return h_min, h_shannon

h_min, h_shannon = estimar_entropias(bits_brutos)
print(f"\nEntropia de Shannon: {h_shannon:.6f} bits/bit")
print(f"Min-entropia (usada no dimensionamento): {h_min:.6f} bits/bit")

# =========================================================
# 5️⃣ EXTRATOR TOEPLITZ (via FFT)
# =========================================================
EPSILON = 1e-10
margem_seguranca = 2 * np.log2(1 / EPSILON)

n_out_seguro = int(np.floor(N_total_bruto * h_min - margem_seguranca))
print(f"\nBits brutos disponíveis: {N_total_bruto}")
print(f"Margem de segurança: {margem_seguranca:.1f} bits")
print(f"Tamanho de saída seguro (n_out): {n_out_seguro}")

if n_out_seguro < 1_000_000:
    raise ValueError(
        f"n_out ({n_out_seguro}) < 10^6. "
        f"Aumente NUM_LSB (atual={NUM_LSB}) para gerar mais bits brutos, "
        f"ou revise a min-entropia estimada."
    )

n_out = n_out_seguro

def toeplitz_extractor_fft(x, n_out):
    n_in = len(x)
    tamanho_seed = n_in + n_out - 1
    seed_bytes = secrets.token_bytes((tamanho_seed // 8) + 1)
    seed_bits = np.unpackbits(
        np.frombuffer(seed_bytes, dtype=np.uint8)
    )[:tamanho_seed].astype(np.float64)

    x_float = x.astype(np.float64)
    seed_invertido = seed_bits[::-1]
    conv = fftconvolve(seed_invertido, x_float, mode='valid')
    y = np.mod(np.round(conv), 2).astype(np.uint8)
    return y, seed_bits

print(f"\nExecutando extrator Toeplitz (n_in={N_total_bruto} → n_out={n_out})...")
bits_finais, seed_usado = toeplitz_extractor_fft(bits_brutos, n_out)

print(f"Bits finais extraídos: {len(bits_finais)}")
print(f"   P(1) pós-extração: {np.mean(bits_finais):.6f}")

# =========================================================
# 6️⃣ SALVANDO ARQUIVO .TXT
# =========================================================
NOME_SAIDA = 'bits_aleatorios_qrng20260916.txt'
with open(NOME_SAIDA, 'w') as f:
    f.write(''.join(bits_finais.astype(str)))

print(f"\nArquivo salvo: {NOME_SAIDA}")
print(f"   Total de bits gravados: {len(bits_finais)}")

np.save('toeplitz_seed.npy', seed_usado)
print(f"Semente Toeplitz salva em: toeplitz_seed.npy")