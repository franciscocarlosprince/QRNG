import numpy as np
import pandas as pd
from scipy.stats import entropy, pearsonr
from scipy.linalg import toeplitz
import secrets
import os

# ============================================================
# 🔧 CONFIGURAÇÕES
# ============================================================
CSV_PATH = r"C:\Users\Francisco C. Prince\OneDrive\Paper QRNG\Data_fromChip\qrng20260916.csv"
OUTPUT_BITS_PATH = "bits_aleatorios20260916_6BIT_SAMPLE_2.txt"
SEED_PATH = "toeplitz_seed.npy"

NUM_LSB = 6                   # 🔢 Número de bits menos significativos a testar
LIMIAR_DESVIO_MAX = 0.02       # Tolerância: |P(1) - 0.5| <= 2%
LIMIAR_CORRELACAO_MAX = 0.05   # Tolerância de correlação entre bits
RESOLUCAO_ADC_BITS = 12        # Resolução nominal do ADC

# ============================================================
# 1️⃣ VALIDAÇÃO DO CAMINHO E LEITURA DO CSV
# ============================================================
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Arquivo não encontrado: {CSV_PATH}")

print(f"Lendo arquivo: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
quadraturas = df.iloc[:, 0].values.astype(np.float64)
print(f"{len(quadraturas)} amostras carregadas.\n")

# ============================================================
# 2️⃣ PRÉ-PROCESSAMENTO: REMOÇÃO DE OFFSET DC
# ============================================================
offset_dc = np.mean(quadraturas)
quadraturas_centralizadas = quadraturas - offset_dc
print(f"Offset DC removido: {offset_dc:.6f}\n")

# ============================================================
# 3️⃣ VALIDAÇÃO DO PASSO DO ADC (RESOLUÇÃO EFETIVA)
# ============================================================
valores_unicos = np.unique(quadraturas)
diffs = np.diff(np.sort(valores_unicos))
diffs_positivos = diffs[diffs > 1e-12]
passo_adc_estimado = np.min(diffs_positivos) if len(diffs_positivos) > 0 else None

if passo_adc_estimado is None:
    raise ValueError("Não foi possível estimar o passo do ADC.")

faixa_dados = quadraturas.max() - quadraturas.min()
n_niveis_estimado = faixa_dados / passo_adc_estimado
resolucao_efetiva_bits = np.log2(n_niveis_estimado + 1)

print(f"📏 Passo do ADC estimado: {passo_adc_estimado:.8f}")
print(f"📐 Resolução efetiva estimada: {resolucao_efetiva_bits:.2f} bits")
if resolucao_efetiva_bits < RESOLUCAO_ADC_BITS - 0.5:
    print(f"⚠️  Aviso: resolução efetiva menor que a nominal ({RESOLUCAO_ADC_BITS} bits).")
    print(f"⚠️  Bits mais significativos do código reconstruído podem ser inválidos/redundantes.\n")
else:
    print("✅ Resolução efetiva compatível com a nominal.\n")

# ============================================================
# 4️⃣ RECONSTRUÇÃO DOS CÓDIGOS ADC (INTEIROS)
# ============================================================
codigos_adc = np.round(
    (quadraturas - quadraturas.min()) / passo_adc_estimado
).astype(np.int64)

n_bits_necessarios = int(np.ceil(np.log2(codigos_adc.max() + 1)))
print(f"🔢 Códigos ADC reconstruídos como inteiros de {n_bits_necessarios} bits.\n")

# ============================================================
# 5️⃣ EXTRAÇÃO DE BITS POR POSIÇÃO (LSB) + VALIDAÇÃO AUTOMÁTICA
# ============================================================
def extrair_bit_posicao(codigos, posicao):
    """Extrai o bit na posição `posicao` (0 = LSB) de cada código ADC."""
    return (codigos >> posicao) & 1

print(f"🔎 Testando {NUM_LSB} posições de bit (LSB → mais significativo)...")
print(f"   Critério de aceitação: |P(1) - 0.5| <= {LIMIAR_DESVIO_MAX}\n")

bits_por_posicao = {}
bits_validos = []
bits_rejeitados = []

for i in range(NUM_LSB):
    bit_i = extrair_bit_posicao(codigos_adc, i)
    p1 = np.mean(bit_i)
    desvio = abs(p1 - 0.5)

    if desvio <= LIMIAR_DESVIO_MAX:
        status = "✅ ACEITO"
        bits_validos.append(i)
    else:
        status = "❌ REJEITADO"
        bits_rejeitados.append(i)

    bits_por_posicao[i] = bit_i
    print(f"   Bit {i}: P(1) = {p1:.6f} | Desvio = {desvio:.6f}  {status}")

print(f"\n📌 Bits aceitos: {bits_validos}")
if bits_rejeitados:
    print(f"🚫 Bits rejeitados (excluídos da extração): {bits_rejeitados}")

if len(bits_validos) == 0:
    raise ValueError("❌ Nenhum bit passou a validação. Abortando.")

print()

# ============================================================
# 6️⃣ TESTE DE CORRELAÇÃO DE PEARSON ENTRE BITS VÁLIDOS
# ============================================================
print("🔗 Teste de correlação de Pearson entre pares de bits válidos:")

correlacoes_problematicas = []

if len(bits_validos) < 2:
    print("   ⚠️  Apenas 1 bit válido — teste de correlação não aplicável.\n")
else:
    for idx_a in range(len(bits_validos)):
        for idx_b in range(idx_a + 1, len(bits_validos)):
            pos_a = bits_validos[idx_a]
            pos_b = bits_validos[idx_b]
            bi = bits_por_posicao[pos_a]
            bj = bits_por_posicao[pos_b]

            corr, p_value = pearsonr(bi, bj)
            abs_corr = abs(corr)

            status = "✅" if abs_corr <= LIMIAR_CORRELACAO_MAX else "⚠️  ALTA CORRELAÇÃO"
            if abs_corr > LIMIAR_CORRELACAO_MAX:
                correlacoes_problematicas.append((pos_a, pos_b, corr))

            print(f"   Bit {pos_a} vs Bit {pos_b}: r = {corr:+.6f} (p={p_value:.4f})  {status}")

    print()
    if correlacoes_problematicas:
        print("⚠️  ATENÇÃO: pares de bits com correlação acima do limiar:")
        for pos_a, pos_b, corr in correlacoes_problematicas:
            print(f"     - Bit {pos_a} & Bit {pos_b}: r = {corr:+.6f}")
        print("   → Esses bits ainda serão incluídos, mas considere excluí-los")
        print("     manualmente se a correlação persistir em análises futuras.\n")
    else:
        print("✅ Nenhuma correlação problemática detectada entre os bits.\n")

# ============================================================
# 7️⃣ MONTAGEM DO BITSTREAM BRUTO (APENAS BITS VÁLIDOS)
# ============================================================
# Concatena os bits válidos por amostra, na ordem: bit0, bit1, bit2, ...
bitstream_bruto = np.zeros(len(codigos_adc) * len(bits_validos), dtype=np.uint8)

for idx, pos in enumerate(bits_validos):
    bitstream_bruto[idx::len(bits_validos)] = bits_por_posicao[pos]

n_bits_brutos = len(bitstream_bruto)
print(f"📦 Bitstream bruto montado: {n_bits_brutos} bits "
      f"({len(codigos_adc)} amostras × {len(bits_validos)} bits válidos/amostra)\n")

# ============================================================
# 8️⃣ ESTIMATIVA DE MIN-ENTROPIA (SOBRE O BITSTREAM COMBINADO)
# ============================================================
# Estimativa por blocos de tamanho len(bits_validos), para capturar dependência conjunta
tamanho_bloco = len(bits_validos)
n_blocos = n_bits_brutos // tamanho_bloco
blocos = bitstream_bruto[:n_blocos * tamanho_bloco].reshape(n_blocos, tamanho_bloco)

# Converte cada bloco em um valor inteiro (0 a 2^tamanho_bloco - 1)
valores_blocos = np.zeros(n_blocos, dtype=np.int64)
for i in range(tamanho_bloco):
    valores_blocos = (valores_blocos << 1) | blocos[:, i]

contagens = np.bincount(valores_blocos, minlength=2**tamanho_bloco)
probabilidades = contagens / n_blocos
probabilidades_nao_zero = probabilidades[probabilidades > 0]

# Min-entropia empírica: -log2(max(p))
p_max = np.max(probabilidades_nao_zero)
min_entropia_por_bloco = -np.log2(p_max)
min_entropia_por_bit = min_entropia_por_bloco / tamanho_bloco

# Entropia de Shannon (referência, não usada para dimensionar saída)
shannon_entropia_por_bloco = entropy(probabilidades_nao_zero, base=2)

print(f"📊 Estimativa de entropia (blocos de {tamanho_bloco} bits):")
print(f"   Min-entropia por bloco:  {min_entropia_por_bloco:.6f} bits")
print(f"   Min-entropia por bit:    {min_entropia_por_bit:.6f} bits/bit")
print(f"   Shannon (referência):    {shannon_entropia_por_bloco:.6f} bits/bloco\n")

# ============================================================
# 9️⃣ DIMENSIONAMENTO DA SAÍDA DO EXTRATOR TOEPLITZ
# ============================================================
MARGEM_SEGURANCA = 0.90  # fator de segurança adicional
n_out_estimado = int(n_bits_brutos * min_entropia_por_bit * MARGEM_SEGURANCA)

MIN_BITS_SAIDA = 1_000_000
if n_out_estimado < MIN_BITS_SAIDA:
    print(f"⚠️  Aviso: saída estimada ({n_out_estimado} bits) é menor que o mínimo exigido.")
    print(f"   Ajustando para o máximo possível dado os dados disponíveis.\n")
    n_out = min(n_out_estimado, n_bits_brutos - 1)
else:
    n_out = n_out_estimado

n_out = max(n_out, MIN_BITS_SAIDA) if n_bits_brutos >= MIN_BITS_SAIDA else n_out

print(f"🎯 Dimensão de saída do extrator Toeplitz: {n_out} bits\n")

# ============================================================
# 🔟 EXTRATOR DE TOEPLITZ VIA FFT
# ============================================================
def gerar_seed_toeplitz(n_in, n_out):
    """Gera semente aleatória segura para a matriz de Toeplitz."""
    total_bits_seed = n_in + n_out - 1
    seed_bits = np.array(
        [secrets.randbits(1) for _ in range(total_bits_seed)], dtype=np.uint8
    )
    return seed_bits

def extrator_toeplitz_fft(bitstream, n_out, seed_bits):
    """
    Aplica extrator de Toeplitz usando multiplicação via FFT
    para eficiência com entradas grandes.
    """
    n_in = len(bitstream)
    primeira_coluna = seed_bits[:n_in]
    primeira_linha = seed_bits[n_in - 1:n_in - 1 + n_out]

    # Multiplicação matriz-vetor via convolução FFT (mod 2)
    x = bitstream.astype(np.float64)
    linha_completa = np.concatenate([primeira_coluna[::-1], primeira_linha[1:]])

    convolucao = np.fft.ifft(
        np.fft.fft(linha_completa, n=n_in + n_out - 1) *
        np.fft.fft(x, n=n_in + n_out - 1)
    ).real

    resultado = np.round(convolucao[n_in - 1:n_in - 1 + n_out]).astype(np.int64)
    saida_bits = (resultado % 2).astype(np.uint8)
    return saida_bits

print("🔐 Gerando seed do extrator de Toeplitz...")
seed_bits = gerar_seed_toeplitz(n_bits_brutos, n_out)
np.save(SEED_PATH, seed_bits)
print(f"✅ Seed salva em: {SEED_PATH}\n")

print("⚙️  Aplicando extrator de Toeplitz (via FFT)...")
bits_finais = extrator_toeplitz_fft(bitstream_bruto, n_out, seed_bits)
print(f"✅ Extração concluída: {len(bits_finais)} bits finais gerados.\n")

# ============================================================
# 1️⃣1️⃣ GRAVAÇÃO DO ARQUIVO DE BITS
# ============================================================
bits_str = ''.join(bits_finais.astype(str))

with open(OUTPUT_BITS_PATH, 'w') as f:
    f.write(bits_str)

print(f"💾 Arquivo salvo: {OUTPUT_BITS_PATH}")
print(f"📏 Total de bits gravados: {len(bits_str)}")

if len(bits_str) >= MIN_BITS_SAIDA:
    print(f"✅ Requisito de ≥ {MIN_BITS_SAIDA:,} bits ATENDIDO.\n")
else:
    print(f"❌ Requisito de ≥ {MIN_BITS_SAIDA:,} bits NÃO atendido!")
    print(f"   Considere aumentar NUM_LSB ou verificar a qualidade dos dados.\n")

# ============================================================
# 📋 RESUMO FINAL
# ============================================================
print("=" * 60)
print("📋 RESUMO DA EXECUÇÃO")
print("=" * 60)
print(f"Amostras processadas:        {len(quadraturas):,}")
print(f"Bits testados (NUM_LSB):     {NUM_LSB}")
print(f"Bits válidos utilizados:     {bits_validos}")
print(f"Bits rejeitados:             {bits_rejeitados if bits_rejeitados else 'nenhum'}")
print(f"Pares c/ alta correlação:    {len(correlacoes_problematicas)}")
print(f"Bitstream bruto:             {n_bits_brutos:,} bits")
print(f"Min-entropia estimada:       {min_entropia_por_bit:.4f} bits/bit")
print(f"Bits finais (pós-Toeplitz):  {len(bits_finais):,}")
print(f"Arquivo de saída:            {OUTPUT_BITS_PATH}")
print(f"Arquivo de seed:             {SEED_PATH}")
print("=" * 60)