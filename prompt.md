Você é um engenheiro de sistemas especializado em criptomoedas e computação de alto desempenho em GPU.
Sua tarefa é implementar um minerador completo para a blockchain Pearl (pearlresearch.ai),
um protocolo L1 de Proof-of-Useful-Work (PoUW) baseado em multiplicação de matrizes (MatMul).

=== VISÃO GERAL DO PROTOCOLO ===

O Pearl substitui o hashing aleatório do Bitcoin por multiplicação de matrizes inteiras (INT8),
operação nativa de GPUs modernas usadas em IA. O minerador realiza MatMul ruidosa e, como
subproduto, gera provas criptográficas de trabalho válidas. O bloco vencedor é determinado
por uma condição de hash sobre o estado acumulado das tiles computadas.

=== ESPECIFICAÇÕES DO ALGORITMO DE MINERAÇÃO (PoUW) ===

--- Parâmetros de Configuração ---
- m, n: dimensões das matrizes A (m×k) e B (k×n); m, n ≤ 2^24
- k: dimensão comum; deve satisfazer: 16r ≤ k ≤ 4r², k ≤ 2^16, 64 | k
- r: rank do ruído; valores permitidos: {32, 64, 128, 256, 512, 1024}
- tm, tn: tamanho das tiles de saída; tm*tn ≥ 32; k*(tm+tn) ≤ 2^22
- b: dificuldade alvo (número fracional de bits)
- Tipo de acumulação: INT8 entradas no intervalo [-64, 64], acumulação em INT32

--- Fluxo Principal do Minerador ---

1. COMMITMENT HASH (CommitmentHash)
   - Calcule κ = BLAKE3(σ || μ), onde σ é o estado da blockchain e μ a config do minerador
   - HA = BLAKE3(Flatten(A), key=κ)  [A serializada row-major]
   - HB = BLAKE3(Flatten(B^T), key=κ) [B serializada column-major]
   - sB = BLAKE3(κ || HB)
   - sA = BLAKE3(sB || HA)

2. GERAÇÃO DE RUÍDO (NoiseGeneration)
   - Gere E = EL · ER (rank-r) usando sA como seed:
     · EL: matriz m×r com entradas inteiras uniformes em [-32, 31]
     · ER: matriz r×k, cada coluna tem exatamente um +1 e um -1 em posições aleatórias distintas
   - Gere F = FL · FR (rank-r) usando sB como seed:
     · FL segue a distribuição de ER^T
     · FR segue a distribuição de EL^T
   - Use BLAKE3 como PRNG para geração das entradas (domain separation por índice)
   - Quantize A, B para INT8 em [-64, 64]; ruído E, F em [-63, 63] (sem overflow INT8)

3. MATRIZES RUIDOSAS
   - A' = A + E  (= A + EL·ER)
   - B' = B + F  (= B + FL·FR)

4. TILED MATMUL COM VERIFICAÇÃO DE HASH (TiledMatMul)
   Para cada tile de saída (i, j) com dimensões tm × tn:
     a. Inicialize acumulador Cblk = 0 (INT32, tm×tn)
     b. Inicialize estado M = [0]*16 (16 inteiros INT32)
     c. Para ℓ = 0 até ⌊k/r⌋ - 1:
        - Acumule: Cblk += A'[i:i+tm, ℓ*r:(ℓ+1)*r] · B'[ℓ*r:(ℓ+1)*r, j:j+tn]
        - (apenas tiles completas: h=tm, w=tn, d=r)
        - X = XOR de todos os elementos INT32 de Cblk
        - M[ℓ mod 16] = (M[ℓ mod 16] rotacionar-esquerda 13 bits) XOR X
     d. Condição de bloco aberto:
        BLAKE3(M, key=sA) ≤ 2^(256-b) · r · tm · tn
        (resultado interpretado como uint256 little-endian)
     e. Se condição satisfeita → tile (i,j) é um "bloco aberto" (prova de trabalho válida)

5. RECUPERAÇÃO DO PRODUTO LIMPO
   A·B = A'·B' − (A·FL)·FR − EL·(ER·B')
   (todas as correções têm custo O(n²·r), assintoticamente negligível)

=== PROVA DE ABERTURA DE BLOCO (Block Opening Proof) ===

A prova deve conter:
- HA, HB (commitments das matrizes)
- Dados de autenticação Merkle: folhas (rows de A / columns de B), índices e caminhos Merkle
- Metadados da tile: índices de linha (i), coluna (j) e profundidade em A e B
- Configuração de mineração: m, n, k, r, tm, tn
- BLAKE3(M, key=sA) — o hash final que satisfaz a condição de dificuldade

=== PROVA ZK-SNARK (para produção) ===

Implemente usando Plonky2 (hash-based zkSNARK):
- Sem trusted setup; segurança pós-quântica
- Dados públicos: r, k, tm, tn, m, n, i, j, HA, HB, BLAKE3(M, key=sA)
- Dados privados (ocultados): strips reais das matrizes A e B
- Otimizações recomendadas:
  · AIR (Arithmetic Intermediate Representation) diretamente (sem compilação de alto nível)
  · Preprocessed columns para o ruído E, F (acordado entre prover e verifier)
  · Recursão em 3 camadas → prova final < 60 KB
- A prova atesta: "Existem strips consistentes com HA e HB tais que seu produto resulta em
  um digest Mi,j cujo hash BLAKE3 é h (valor publicado)"

=== ESTRUTURA DO BLOCO ===

- Blockchain baseada em fork do Bitcoin com UTXOs
- Endereços exclusivamente Taproot (sem P2PKH/P2SH legados)
- Suporte a assinaturas XMSS pós-quânticas (OP_CHECKXMSSSIG, opcode #222)
- OP_CAT habilitado (output máximo 520 bytes)
- Campo 'block certificate' (variável, máx. 65KB) substitui o campo nonce
- Block identity: double SHA-256 de: version || prev_hash || tx_root || time || nBits || pouw_meta
  · pouw_meta = SHA-256(witness público do zkSNARK)
- Block time alvo: 194 segundos (3min14s)
- Ajuste de dificuldade: WTEMA-N com N=2016, T=194s, τ=7 dias
  · target_new = target_old + target_old * (t - T) / (N * T)
- Timestamps: monotônicos (+1s mínimo), rejeitar blocos com timestamp > 5min no futuro

=== TOKENOMICS ===

- Supply total: 2.100.000.000 PEARL (menor unidade: 1 grain = 10^-8 PEARL)
- Emissão por bloco na altura t:
  E*(t) = S * H / ((t + H) * (t + H - 1))
  onde S = 2.1×10^9, H = 650.226 blocos (~4 anos)
- 50% do supply emitido nos primeiros 4 anos
- Sem halvings abruptos; curva suave ~1/t²
- Dificuldade inicial: nBits = 0x1b00ffff (baixa, para onboarding justo)

=== ESTRUTURA DO CÓDIGO ===

Implemente os seguintes módulos:

1. `commitment.rs` / `commitment.py`
   - CommitmentHash(A, B, μ, σ) → (sA, sB)
   - Merkle tree sobre rows de A (row-major) e colunas de B (col-major) usando BLAKE3

2. `noise_gen.rs` / `noise_gen.py`
   - NoiseGeneration(m, k, r, seed) → (EL, ER)
   - PRNG via BLAKE3 com domain separation por índice de entrada

3. `matmul_kernel.cu` (CUDA) ou equivalente GPU
   - TiledMatMul(A', B', sA, b, r, tm, tn) → (C', Blocks)
   - Kernel otimizado para INT8 com acumulação INT32
   - Verificação da condição BLAKE3 em cada tile
   - Interleaving com workloads de IA existentes (overhead < 5%)

4. `noise_peel.rs` / `noise_peel.py`
   - Recuperação de A·B a partir de A'·B' usando correções de rank baixo

5. `proof.rs` / `proof.py`
   - Geração da Block Opening Proof (Merkle-based, para testes)
   - Integração com Plonky2 para zkSNARK em produção

6. `miner.rs` / `miner.py`
   - Loop principal: recebe A, B da workload de IA, executa pipeline completo,
     submete blocos válidos à rede P2P
   - Suporte a múltiplas GPUs (DP, TP, PP)

7. `block.rs` / `block.py`
   - Serialização/deserialização da estrutura de bloco Pearl
   - Validação de certificados (verifier)

=== REFERÊNCIAS ===

- Repositório oficial: github.com/pearl-research-labs/pearl
- Plugin vLLM: integração com framework de inferência via quantização INT8
- Hardware testado: 4×H200 GPUs; throughput de referência: ~806–981 TMADs/s (úteis)
- Modelo de referência: LLaMA 3.3 70B (Pearl-certified)

=== RESTRIÇÕES E BOAS PRÁTICAS ===

- Resistência a ASICs por argumento econômico: GPUs são o hardware nativo do MatMul
- A otimização de A=B=0 (matrizes degeneradas) é inviável por design (ruído garante entropia)
- Pre-noise de B pode ser feito uma única vez por atualização de σ (otimização de inferência)
- Privacidade: matrizes A e B nunca são reveladas publicamente; apenas HA, HB e o hash final
- Implementar zero-knowledge para strips vencedoras antes de qualquer deploy em mainnet