# -*- coding: utf-8 -*-
"""
Comparador de modelos (OPTIMA) — versão alinhada ao runner do PLAMUS
- Estima: TgCg, TgCe, TeCg, TeCe, TgCR(BMA)e, TeCR(BMA)e
- Reporta: LL, K, AIC, BIC, HitRate (TA), VoT
- Testes LR para pares aninhados naturais
"""
from pathlib import Path
import math
import pandas as pd

import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
from biogeme.expressions import Beta

# =========================
# CONFIG
# =========================
DATA_PATH   = Path(__file__).parent / "Input"
FILE_IN     = DATA_PATH / "optima.dat"
MODEL_PREFIX= "modelo_optima"
CHOICE_VAR  = "Choice"  # 0 = PT (coletivo), 1 = CAR (individual), 2 = "soft modes" (excluídos)

# Modelos (nomes simples, como combinamos)
SPECS = [
    'TgCg',          # Tempo genérico, Custo genérico
    'TgCe',          # Tempo genérico, Custo específico por modo
    'TeCg',          # Tempo específico por modo, Custo genérico
    'TeCe',          # Tempo específico por modo, Custo específico por modo
    'TgCR(BMA)e',    # Tempo genérico, Custo x Renda (B/M/A) específico por modo
    'TeCR(BMA)e',    # Tempo específico por modo, Custo x Renda (B/M/A) específico por modo
]

# Pares aninhados (restrito -> completo) para LR
NESTED_PAIRS = [
    ('TgCg'        , 'TgCe'),
    ('TgCg'        , 'TeCg'),
    ('TgCe'        , 'TeCe'),
    ('TeCg'        , 'TeCe'),
    ('TgCR(BMA)e'  , 'TeCR(BMA)e'),
]

# =========================
# DADOS + FILTROS
# =========================
df = pd.read_csv(FILE_IN, sep="\t")

# Filtros básicos (remover soft modes e missings usuais)
mask = (
    (df[CHOICE_VAR] != -1) &
    (df[CHOICE_VAR] != 2) &
    (df['Income'] != -1) &
    (df['CarAvail'] != -1) &
    (df['CarAvail'] != 3)
)
df = df.loc[mask].copy()

# Faixas B/M/A (OPTIMA): 1..6 (até 4k, 4-8k, acima de 8k CHF, conforme prática do curso)
df['B'] = df['Income'].astype(int).isin([1, 2]).astype(int)
df['M'] = df['Income'].astype(int).isin([3, 4]).astype(int)
df['A'] = df['Income'].astype(int).isin([5, 6]).astype(int)

database = db.Database('optima', df)
globals().update(database.variables)  # expõe Choice, TimePT, TimeCar, MarginalCostPT, CostCarCHF, B/M/A

# =========================
# HELPERS
# =========================
def BETA(name, start=0, fixed=False):
    return Beta(name, start, None, None, 1 if fixed else 0)

def build_betas(spec: str):
    """Cria apenas os Betas necessários para cada especificação."""
    betas = {'ASC_PT': BETA('ASC_PT', 0, fixed=True),
             'ASC_CAR': BETA('ASC_CAR', 0)}

    if spec == 'TgCg':
        betas['B_TIME'] = BETA('B_TIME'); betas['B_COST'] = BETA('B_COST')

    elif spec == 'TgCe':
        betas['B_TIME'] = BETA('B_TIME')
        betas['B_COST_PT'] = BETA('B_COST_PT'); betas['B_COST_CAR'] = BETA('B_COST_CAR')

    elif spec == 'TeCg':
        betas['B_TIME_PT'] = BETA('B_TIME_PT'); betas['B_TIME_CAR'] = BETA('B_TIME_CAR')
        betas['B_COST'] = BETA('B_COST')

    elif spec == 'TeCe':
        betas['B_TIME_PT'] = BETA('B_TIME_PT'); betas['B_TIME_CAR'] = BETA('B_TIME_CAR')
        betas['B_COST_PT'] = BETA('B_COST_PT'); betas['B_COST_CAR'] = BETA('B_COST_CAR')

    elif spec == 'TgCR(BMA)e':
        betas['B_TIME'] = BETA('B_TIME')
        for alt in ['PT','CAR']:
            for grp in ['B','M','A']:
                betas[f'B_COST_{alt}_{grp}'] = BETA(f'B_COST_{alt}_{grp}')

    elif spec == 'TeCR(BMA)e':
        betas['B_TIME_PT'] = BETA('B_TIME_PT'); betas['B_TIME_CAR'] = BETA('B_TIME_CAR')
        for alt in ['PT','CAR']:
            for grp in ['B','M','A']:
                betas[f'B_COST_{alt}_{grp}'] = BETA(f'B_COST_{alt}_{grp}')

    else:
        raise ValueError(f"SPEC desconhecida: {spec}")
    return betas

def build_utilities(spec: str, betas: dict):
    """Retorna V_PT, V_CAR para cada especificação."""
    ASC_PT, ASC_CAR = betas['ASC_PT'], betas['ASC_CAR']

    if spec == 'TgCg':
        V_PT  = ASC_PT  + betas['B_TIME']*TimePT + betas['B_COST']*MarginalCostPT
        V_CAR = ASC_CAR + betas['B_TIME']*TimeCar + betas['B_COST']*CostCarCHF

    elif spec == 'TgCe':
        V_PT  = ASC_PT  + betas['B_TIME']*TimePT + betas['B_COST_PT']*MarginalCostPT
        V_CAR = ASC_CAR + betas['B_TIME']*TimeCar + betas['B_COST_CAR']*CostCarCHF

    elif spec == 'TeCg':
        V_PT  = ASC_PT  + betas['B_TIME_PT']*TimePT + betas['B_COST']*MarginalCostPT
        V_CAR = ASC_CAR + betas['B_TIME_CAR']*TimeCar + betas['B_COST']*CostCarCHF

    elif spec == 'TeCe':
        V_PT  = ASC_PT  + betas['B_TIME_PT']*TimePT + betas['B_COST_PT']*MarginalCostPT
        V_CAR = ASC_CAR + betas['B_TIME_CAR']*TimeCar + betas['B_COST_CAR']*CostCarCHF

    elif spec == 'TgCR(BMA)e':
        V_PT  = ASC_PT  + betas['B_TIME']*TimePT \
              + (betas['B_COST_PT_B']*B + betas['B_COST_PT_M']*M + betas['B_COST_PT_A']*A) * MarginalCostPT
        V_CAR = ASC_CAR + betas['B_TIME']*TimeCar \
              + (betas['B_COST_CAR_B']*B + betas['B_COST_CAR_M']*M + betas['B_COST_CAR_A']*A) * CostCarCHF

    elif spec == 'TeCR(BMA)e':
        V_PT  = ASC_PT  + betas['B_TIME_PT']*TimePT \
              + (betas['B_COST_PT_B']*B + betas['B_COST_PT_M']*M + betas['B_COST_PT_A']*A) * MarginalCostPT
        V_CAR = ASC_CAR + betas['B_TIME_CAR']*TimeCar \
              + (betas['B_COST_CAR_B']*B + betas['B_COST_CAR_M']*M + betas['B_COST_CAR_A']*A) * CostCarCHF

    else:
        raise ValueError(f"SPEC desconhecida: {spec}")

    return V_PT, V_CAR

def extract_stats(results):
    """LL, K, N, AIC, BIC, rho2, rho2bar."""
    gs = results.getGeneralStatistics()
    def _get(*names):
        for n in names:
            if n in gs:
                v = gs[n]
                return v[0] if isinstance(v, tuple) else v
        return None
    LL0  = _get('Init log likelihood', 'Initial log likelihood')
    LL   = _get('Final log likelihood', 'Final loglikelihood')
    K    = _get('Number of estimated parameters', 'Nbr. parameters')
    N    = _get('Sample size', 'Number of observations')
    rho2 = _get('Rho-square')
    rho2b= _get('Rho-square-bar', 'Rho-square-bar for the final model')
    AIC = (2*K - 2*LL) if (K is not None and LL is not None) else None
    BIC = ((math.log(N)*K - 2*LL) if (N and K is not None and LL is not None) else None)
    return {'LL0': LL0, 'LL': LL, 'K': K, 'N': N, 'AIC': AIC, 'BIC': BIC, 'rho2': rho2, 'rho2bar': rho2b}

def simulate_probs_and_hit(betas_values, V_PT, V_CAR):
    """Probabilidades previstas e taxa de acerto (argmax)"""
    P_PT  = models.logit({0: V_PT, 1: V_CAR}, None, 0)
    P_CAR = models.logit({0: V_PT, 1: V_CAR}, None, 1)
    sim = {'P_PT': P_PT, 'P_CAR': P_CAR}
    bg_sim = bio.BIOGEME(database, sim)
    probs = bg_sim.simulate(betas_values)
    pred_car = (probs['P_CAR'] > probs['P_PT']).astype(int)
    hit = (pred_car.values == df[CHOICE_VAR].values).mean()
    return hit

def compute_vot(results, spec: str):
    """VoT conforme a especificação: genérico, por modo, ou por modo×faixa (B/M/A)."""
    b = results.getBetaValues()
    out = {}
    def r(num, den):
        if num is None or den in (None, 0): return None
        return -num/den

    if spec == 'TgCg':
        out['VoT'] = r(b.get('B_TIME'), b.get('B_COST'))

    elif spec == 'TgCe':
        out['VoT_PT']  = r(b.get('B_TIME'), b.get('B_COST_PT'))
        out['VoT_CAR'] = r(b.get('B_TIME'), b.get('B_COST_CAR'))

    elif spec == 'TeCg':
        out['VoT_PT']  = r(b.get('B_TIME_PT'),  b.get('B_COST'))
        out['VoT_CAR'] = r(b.get('B_TIME_CAR'), b.get('B_COST'))

    elif spec == 'TeCe':
        out['VoT_PT']  = r(b.get('B_TIME_PT'),  b.get('B_COST_PT'))
        out['VoT_CAR'] = r(b.get('B_TIME_CAR'), b.get('B_COST_CAR'))

    elif spec == 'TgCR(BMA)e':
        for alt in ['PT','CAR']:
            for grp in ['B','M','A']:
                out[f'VoT_{alt}_{grp}'] = r(b.get('B_TIME'), b.get(f'B_COST_{alt}_{grp}'))

    elif spec == 'TeCR(BMA)e':
        for alt, tpar in [('PT','B_TIME_PT'), ('CAR','B_TIME_CAR')]:
            for grp in ['B','M','A']:
                out[f'VoT_{alt}_{grp}'] = r(b.get(tpar), b.get(f'B_COST_{alt}_{grp}'))

    return out

def lr_test(spec_restricted, spec_full, results_map):
    """LR = 2(LL_full - LL_restricted), df = K_full - K_restricted."""
    if spec_restricted not in results_map or spec_full not in results_map:
        return None
    R = extract_stats(results_map[spec_restricted])
    F = extract_stats(results_map[spec_full])
    if None in (R['LL'], F['LL'], R['K'], F['K']): return None
    LR = 2*(F['LL'] - R['LL'])
    df_ = F['K'] - R['K']
    try:
        from scipy.stats import chi2
        p = chi2.sf(LR, df_) if df_ > 0 else None
    except Exception:
        p = None
    return {'restricted': spec_restricted, 'full': spec_full, 'LR': LR, 'df': df_, 'pvalue': p}

# =========================
# ESTIMAÇÃO + RELATÓRIO
# =========================
rows = []
results_map = {}
errors = {}

print(f"Amostra após filtros: N = {len(df)}")

for spec in SPECS:
    try:
        betas = build_betas(spec)
        V_PT, V_CAR = build_utilities(spec, betas)

        logprob = models.loglogit({0: V_PT, 1: V_CAR}, None, Choice)
        bg = bio.BIOGEME(database, logprob)
        bg.modelName = f"{MODEL_PREFIX}__{spec}"
        results = bg.estimate()

        stats = extract_stats(results)
        vot   = compute_vot(results, spec)
        hit   = simulate_probs_and_hit(results.getBetaValues(), V_PT, V_CAR)

        results_map[spec] = results
        rows.append({'SPEC': spec, **stats, **vot, 'HitRate': hit})
        print(f"[OK] {spec}: LL={stats['LL']:.3f}, K={stats['K']}, AIC={stats['AIC']:.1f}, BIC={stats['BIC']:.1f}, Hit={hit:.3f}")

    except Exception as e:
        errors[spec] = str(e)
        print(f"[FAIL] {spec}: {e}")

summary = pd.DataFrame(rows)

if summary.empty:
    print("\nNenhum modelo estimado com sucesso. Verifique 'errors'.")
else:
    # 1) INFO: LL, K, AIC, BIC, HitRate (ordenado por BIC)
    tbl = summary[['SPEC','LL','K','AIC','BIC','HitRate']].copy().sort_values(['BIC','AIC'])
    print("\n=== TABELA GERAL: LL, K, AIC, BIC, HitRate (ordenado por BIC) ===")
    print(tbl.to_string(index=False))

    # 2) LR para pares aninhados
    lr_rows = []
    for sr, sf in NESTED_PAIRS:
        out = lr_test(sr, sf, results_map)
        if out: lr_rows.append(out)
    if lr_rows:
        lr_df = pd.DataFrame(lr_rows).sort_values(by='LR', ascending=False)
        print("\n=== TESTES LR (pares aninhados) ===")
        print(lr_df.to_string(index=False))
    else:
        print("\nSem pares aninhados válidos para LR.")

    # 3) Salva CSVs
    out_dir = Path(__file__).parent
    tbl.to_csv(out_dir / f"{MODEL_PREFIX}__comparativo.csv", index=False)
    if lr_rows:
        lr_df.to_csv(out_dir / f"{MODEL_PREFIX}__comparativo_LR.csv", index=False)

    # 4) Destaques
    best_bic = tbl.iloc[0]['SPEC']
    best_aic = summary.sort_values(['AIC','BIC']).iloc[0]['SPEC']
    print(f"\nMelhor por BIC: {best_bic}")
    print(f"Melhor por AIC: {best_aic}")
