# -*- coding: utf-8 -*-
"""
Runner para comparar especificações no Biogeme usando o dataset OPTIMA (Suíça)
- Filtra amostra conforme dicionário (remove Choice=2, e missing/indisp.)
- Estima múltiplas especificações (inclui LH/LMH/ALLFAIXAS genérico e específico)
- Reporta LL, K, N, AIC, BIC, rho2, rho2bar, VoT (quando aplicável) e hit rate in-sample
- Faz testes LR para pares aninhados naturais
"""
from pathlib import Path
import pandas as pd
import math
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
from biogeme.expressions import Beta

# =========================
# CONFIG
# =========================
DATA_PATH  = Path(__file__).parent / "Input"
FILE_IN    = DATA_PATH / "optima.dat"
MODEL_PREFIX = "modelo_optima"

SPECS = [
    'TIME_COST_GENERIC',
    'TIME_COST_SPECIFIC',
    'TIME_GENERIC_COST_SPECIFIC',
    'TIME_SPECIFIC_COST_GENERIC',
    'TIME_COSTxRENDA_LH_GENERIC',                 # LH genérico (iguais PT/CAR)
    'TIME_GENERIC_COSTxRENDA_LH_SPECIFIC',        # LH específico por alternativa
    'TIME_COSTxRENDA_LMH_GENERIC',                # LMH genérico (iguais PT/CAR)
    'TIME_GENERIC_COSTxRENDA_LMH_SPECIFIC',       # LMH específico por alternativa
    'TIME_GENERIC_COSTxRENDA_ALLFAIXAS_GENERIC',  # Todas as faixas genérico
    'TIME_GENERIC_COSTxRENDA_ALLFAIXAS_SPECIFIC', # Todas as faixas específico
]

# Pares aninhados (restrito, completo) p/ LR
NESTED_PAIRS = [
    ('TIME_COST_GENERIC'                    , 'TIME_COST_SPECIFIC'),
    ('TIME_GENERIC_COST_SPECIFIC'           , 'TIME_COST_SPECIFIC'),
    ('TIME_SPECIFIC_COST_GENERIC'           , 'TIME_COST_SPECIFIC'),
    ('TIME_COSTxRENDA_LH_GENERIC'           , 'TIME_GENERIC_COSTxRENDA_LH_SPECIFIC'),
    ('TIME_COSTxRENDA_LMH_GENERIC'          , 'TIME_GENERIC_COSTxRENDA_LMH_SPECIFIC'),
    ('TIME_GENERIC_COSTxRENDA_ALLFAIXAS_GENERIC', 'TIME_GENERIC_COSTxRENDA_ALLFAIXAS_SPECIFIC'),
]

# =========================
# PREPARO DOS DADOS
# =========================
df = pd.read_csv(FILE_IN, sep="\t")

# Filtros (Choice: 0=TC, 1=TI, 2=soft -> remover; missing=-1)
mask = (
    (df['Choice'] != -1) &
    (df['Choice'] != 2) &
    (df['Income'] != -1) &
    (df['CarAvail'] != -1) &
    (df['CarAvail'] != 3)     # mantém "sempre" e "às vezes" disponível
)
df = df.loc[mask].copy()

# Dummies ALLFAIXAS para renda (1..6)
inc = pd.get_dummies(df['Income'].astype(int), prefix='INC', dtype=int)
df = pd.concat([df, inc], axis=1)

# Grupos LH / LMH conforme categorias do dicionário:
# Income: 1 <2500, 2 2501-4000, 3 4001-6000, 4 6001-8000, 5 8001-10000, 6 >10001
df['LI'] = df['Income'].isin([1, 2]).astype(int)
df['MI'] = df['Income'].isin([3, 4]).astype(int)
df['HI'] = df['Income'].isin([5, 6]).astype(int)

# Cria Database Biogeme
database = db.Database('optima', df)
globals().update(database.variables)  # expõe Choice, TimePT, TimeCar, MarginalCostPT, CostCarCHF, LI/MI/HI, INC_x...

# =========================
# HELPERS
# =========================
def B(name, start=0, fixed=False):
    return Beta(name, start, None, None, 1 if fixed else 0)

def income_levels_all():
    return sorted(int(c.split('_')[1]) for c in df.columns if c.startswith('INC_'))

def inc_expr(k: int):
    return database.variables.get(f'INC_{k}')

def build_betas(spec: str):
    betas = {'ASC_PT': B('ASC_PT', 0, fixed=True),
             'ASC_CAR': B('ASC_CAR', 0)}

    if spec == 'TIME_COST_GENERIC':
        betas['B_TIME'] = B('B_TIME')
        betas['B_COST'] = B('B_COST')

    elif spec == 'TIME_COST_SPECIFIC':
        betas['B_TIME_PT']  = B('B_TIME_PT')
        betas['B_TIME_CAR'] = B('B_TIME_CAR')
        betas['B_COST_PT']  = B('B_COST_PT')
        betas['B_COST_CAR'] = B('B_COST_CAR')

    elif spec == 'TIME_GENERIC_COST_SPECIFIC':
        betas['B_TIME'] = B('B_TIME')
        betas['B_COST_PT']  = B('B_COST_PT')
        betas['B_COST_CAR'] = B('B_COST_CAR')

    elif spec == 'TIME_SPECIFIC_COST_GENERIC':
        betas['B_TIME_PT']  = B('B_TIME_PT')
        betas['B_TIME_CAR'] = B('B_TIME_CAR')
        betas['B_COST']     = B('B_COST')

    elif spec == 'TIME_COSTxRENDA_LH_GENERIC':
        betas['B_TIME']   = B('B_TIME')
        betas['B_COST_LI'] = B('B_COST_LI')
        betas['B_COST_HI'] = B('B_COST_HI')

    elif spec == 'TIME_GENERIC_COSTxRENDA_LH_SPECIFIC':
        betas['B_TIME']       = B('B_TIME')
        betas['B_COST_PT_LI'] = B('B_COST_PT_LI')
        betas['B_COST_PT_HI'] = B('B_COST_PT_HI')
        betas['B_COST_CAR_LI']= B('B_COST_CAR_LI')
        betas['B_COST_CAR_HI']= B('B_COST_CAR_HI')

    elif spec == 'TIME_COSTxRENDA_LMH_GENERIC':
        betas['B_TIME']   = B('B_TIME')
        betas['B_COST_LI'] = B('B_COST_LI')
        betas['B_COST_MI'] = B('B_COST_MI')
        betas['B_COST_HI'] = B('B_COST_HI')

    elif spec == 'TIME_GENERIC_COSTxRENDA_LMH_SPECIFIC':
        betas['B_TIME'] = B('B_TIME')
        for alt in ['PT', 'CAR']:
            for grp in ['LI','MI','HI']:
                betas[f'B_COST_{alt}_{grp}'] = B(f'B_COST_{alt}_{grp}')

    elif spec == 'TIME_GENERIC_COSTxRENDA_ALLFAIXAS_GENERIC':
        betas['B_TIME'] = B('B_TIME')
        for k in income_levels_all():
            betas[f'B_COST_INC_{k}'] = B(f'B_COST_INC_{k}')

    elif spec == 'TIME_GENERIC_COSTxRENDA_ALLFAIXAS_SPECIFIC':
        betas['B_TIME'] = B('B_TIME')
        for alt in ['PT','CAR']:
            for k in income_levels_all():
                betas[f'B_COST_{alt}_INC_{k}'] = B(f'B_COST_{alt}_INC_{k}')
    else:
        raise ValueError(f"SPEC desconhecida: {spec}")
    return betas

def build_utilities(spec: str, betas: dict):
    ASC_PT, ASC_CAR = betas['ASC_PT'], betas['ASC_CAR']

    if spec == 'TIME_COST_GENERIC':
        V_PT  = ASC_PT  + betas['B_TIME']*TimePT + betas['B_COST']*MarginalCostPT
        V_CAR = ASC_CAR + betas['B_TIME']*TimeCar + betas['B_COST']*CostCarCHF

    elif spec == 'TIME_COST_SPECIFIC':
        V_PT  = ASC_PT  + betas['B_TIME_PT']*TimePT + betas['B_COST_PT']*MarginalCostPT
        V_CAR = ASC_CAR + betas['B_TIME_CAR']*TimeCar + betas['B_COST_CAR']*CostCarCHF

    elif spec == 'TIME_GENERIC_COST_SPECIFIC':
        V_PT  = ASC_PT  + betas['B_TIME']*TimePT + betas['B_COST_PT']*MarginalCostPT
        V_CAR = ASC_CAR + betas['B_TIME']*TimeCar + betas['B_COST_CAR']*CostCarCHF

    elif spec == 'TIME_SPECIFIC_COST_GENERIC':
        V_PT  = ASC_PT  + betas['B_TIME_PT']*TimePT + betas['B_COST']*MarginalCostPT
        V_CAR = ASC_CAR + betas['B_TIME_CAR']*TimeCar + betas['B_COST']*CostCarCHF

    elif spec == 'TIME_COSTxRENDA_LH_GENERIC':
        V_PT  = ASC_PT  + betas['B_TIME']*TimePT \
                + betas['B_COST_LI']*MarginalCostPT*LI \
                + betas['B_COST_HI']*MarginalCostPT*HI
        V_CAR = ASC_CAR + betas['B_TIME']*TimeCar \
                + betas['B_COST_LI']*CostCarCHF*LI \
                + betas['B_COST_HI']*CostCarCHF*HI

    elif spec == 'TIME_GENERIC_COSTxRENDA_LH_SPECIFIC':
        V_PT  = ASC_PT  + betas['B_TIME']*TimePT \
                + betas['B_COST_PT_LI']*MarginalCostPT*LI \
                + betas['B_COST_PT_HI']*MarginalCostPT*HI
        V_CAR = ASC_CAR + betas['B_TIME']*TimeCar \
                + betas['B_COST_CAR_LI']*CostCarCHF*LI \
                + betas['B_COST_CAR_HI']*CostCarCHF*HI

    elif spec == 'TIME_COSTxRENDA_LMH_GENERIC':
        V_PT  = ASC_PT  + betas['B_TIME']*TimePT \
                + betas['B_COST_LI']*MarginalCostPT*LI \
                + betas['B_COST_MI']*MarginalCostPT*MI \
                + betas['B_COST_HI']*MarginalCostPT*HI
        V_CAR = ASC_CAR + betas['B_TIME']*TimeCar \
                + betas['B_COST_LI']*CostCarCHF*LI \
                + betas['B_COST_MI']*CostCarCHF*MI \
                + betas['B_COST_HI']*CostCarCHF*HI

    elif spec == 'TIME_GENERIC_COSTxRENDA_LMH_SPECIFIC':
        V_PT  = ASC_PT  + betas['B_TIME']*TimePT \
                + betas['B_COST_PT_LI']*MarginalCostPT*LI \
                + betas['B_COST_PT_MI']*MarginalCostPT*MI \
                + betas['B_COST_PT_HI']*MarginalCostPT*HI
        V_CAR = ASC_CAR + betas['B_TIME']*TimeCar \
                + betas['B_COST_CAR_LI']*CostCarCHF*LI \
                + betas['B_COST_CAR_MI']*CostCarCHF*MI \
                + betas['B_COST_CAR_HI']*CostCarCHF*HI

    elif spec == 'TIME_GENERIC_COSTxRENDA_ALLFAIXAS_GENERIC':
        V_PT  = ASC_PT  + betas['B_TIME']*TimePT
        V_CAR = ASC_CAR + betas['B_TIME']*TimeCar
        for k in income_levels_all():
            Ik = inc_expr(k)
            if Ik is not None:
                V_PT  = V_PT  + betas[f'B_COST_INC_{k}']  * MarginalCostPT * Ik
                V_CAR = V_CAR + betas[f'B_COST_INC_{k}']  * CostCarCHF   * Ik

    elif spec == 'TIME_GENERIC_COSTxRENDA_ALLFAIXAS_SPECIFIC':
        V_PT  = ASC_PT  + betas['B_TIME']*TimePT
        V_CAR = ASC_CAR + betas['B_TIME']*TimeCar
        for k in income_levels_all():
            Ik = inc_expr(k)
            if Ik is not None:
                V_PT  = V_PT  + betas[f'B_COST_PT_INC_{k}']  * MarginalCostPT * Ik
                V_CAR = V_CAR + betas[f'B_COST_CAR_INC_{k}'] * CostCarCHF   * Ik
    else:
        raise ValueError(f"SPEC desconhecida: {spec}")

    return V_PT, V_CAR

def extract_stats(results):
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

def compute_vot(results, spec: str):
    betas = results.getBetaValues()
    out = {}
    def safe_ratio(num, den):
        if num is None or den in (None, 0): return None
        return -num/den

    if spec == 'TIME_COST_GENERIC' and 'B_TIME' in betas and 'B_COST' in betas:
        out['VoT_generic'] = safe_ratio(betas['B_TIME'], betas['B_COST'])

    if spec == 'TIME_GENERIC_COST_SPECIFIC' and 'B_TIME' in betas:
        out['VoT_PT']  = safe_ratio(betas['B_TIME'], betas.get('B_COST_PT'))
        out['VoT_CAR'] = safe_ratio(betas['B_TIME'], betas.get('B_COST_CAR'))

    if spec == 'TIME_SPECIFIC_COST_GENERIC' and 'B_COST' in betas:
        out['VoT_PT']  = safe_ratio(betas.get('B_TIME_PT'),  betas['B_COST'])
        out['VoT_CAR'] = safe_ratio(betas.get('B_TIME_CAR'), betas['B_COST'])

    if spec == 'TIME_COSTxRENDA_LH_GENERIC' and 'B_TIME' in betas:
        out['VoT_LI'] = safe_ratio(betas['B_TIME'], betas.get('B_COST_LI'))
        out['VoT_HI'] = safe_ratio(betas['B_TIME'], betas.get('B_COST_HI'))

    if spec == 'TIME_GENERIC_COSTxRENDA_LH_SPECIFIC' and 'B_TIME' in betas:
        out['VoT_PT_LI']  = safe_ratio(betas['B_TIME'], betas.get('B_COST_PT_LI'))
        out['VoT_PT_HI']  = safe_ratio(betas['B_TIME'], betas.get('B_COST_PT_HI'))
        out['VoT_CAR_LI'] = safe_ratio(betas['B_TIME'], betas.get('B_COST_CAR_LI'))
        out['VoT_CAR_HI'] = safe_ratio(betas['B_TIME'], betas.get('B_COST_CAR_HI'))

    if spec == 'TIME_COSTxRENDA_LMH_GENERIC' and 'B_TIME' in betas:
        out['VoT_LI'] = safe_ratio(betas['B_TIME'], betas.get('B_COST_LI'))
        out['VoT_MI'] = safe_ratio(betas['B_TIME'], betas.get('B_COST_MI'))
        out['VoT_HI'] = safe_ratio(betas['B_TIME'], betas.get('B_COST_HI'))

    if spec == 'TIME_GENERIC_COSTxRENDA_LMH_SPECIFIC' and 'B_TIME' in betas:
        for grp in ['LI','MI','HI']:
            out[f'VoT_PT_{grp}']  = safe_ratio(betas['B_TIME'], betas.get(f'B_COST_PT_{grp}'))
            out[f'VoT_CAR_{grp}'] = safe_ratio(betas['B_TIME'], betas.get(f'B_COST_CAR_{grp}'))

    if spec == 'TIME_GENERIC_COSTxRENDA_ALLFAIXAS_GENERIC' and 'B_TIME' in betas:
        for k in income_levels_all():
            out[f'VoT_INC_{k}'] = safe_ratio(betas['B_TIME'], betas.get(f'B_COST_INC_{k}'))

    if spec == 'TIME_GENERIC_COSTxRENDA_ALLFAIXAS_SPECIFIC' and 'B_TIME' in betas:
        for k in income_levels_all():
            out[f'VoT_PT_INC_{k}']  = safe_ratio(betas['B_TIME'], betas.get(f'B_COST_PT_INC_{k}'))
            out[f'VoT_CAR_INC_{k}'] = safe_ratio(betas['B_TIME'], betas.get(f'B_COST_CAR_INC_{k}'))
    return out

def simulate_hit_rate(spec: str, results, V_PT, V_CAR):
    """Calcula taxa de acerto in-sample via probabilidades logit."""
    P_PT  = models.logit({0: V_PT, 1: V_CAR}, None, 0)
    P_CAR = models.logit({0: V_PT, 1: V_CAR}, None, 1)
    sim = {'P_PT': P_PT, 'P_CAR': P_CAR}
    bg_sim = bio.BIOGEME(database, sim)
    bg_sim.modelName = f"{MODEL_PREFIX}__SIM__{spec}"
    probs = bg_sim.simulate(results.getBetaValues())
    # previsão = argmax
    pred_car = (probs['P_CAR'] > probs['P_PT']).astype(int)
    hit = (pred_car.values == df['Choice'].values).mean()
    return hit

def run_one(spec: str):
    betas = build_betas(spec)
    V_PT, V_CAR = build_utilities(spec, betas)
    V = {0: V_PT, 1: V_CAR}
    logprob = models.loglogit(V, None, Choice)
    bg = bio.BIOGEME(database, logprob)
    bg.modelName = f"{MODEL_PREFIX}__{spec}"
    results = bg.estimate()
    stats = extract_stats(results)
    vot   = compute_vot(results, spec)
    hit   = simulate_hit_rate(spec, results, V_PT, V_CAR)
    return results, stats, vot, hit

# =========================
# EXECUÇÃO
# =========================
rows, results_map, errors = [], {}, {}

print(f"Amostra após filtros: N = {len(df)}")
for spec in SPECS:
    try:
        res, st, vot, hit = run_one(spec)
        results_map[spec] = res
        row = {'SPEC': spec, **st, **vot, 'HitRate': hit}
        rows.append(row)
        print(f"[OK] {spec}: LL={st['LL']:.3f}, K={st['K']}, AIC={st['AIC']:.1f}, BIC={st['BIC']:.1f}, Hit={hit:.3f}")
    except Exception as e:
        errors[spec] = str(e)
        print(f"[FAIL] {spec}: {e}")

summary = pd.DataFrame(rows)
if not summary.empty:
    summary_sorted = summary.sort_values(by=['BIC','AIC']).reset_index(drop=True)
    print("\n=== RESUMO (ordenado por BIC) ===")
    print(summary_sorted.to_string(index=False))

    out_csv = Path(__file__).parent / f"{MODEL_PREFIX}__comparativo.csv"
    summary_sorted.to_csv(out_csv, index=False)
    print(f"\nTabela comparativa salva em: {out_csv}")

    best_bic = summary_sorted.iloc[0]['SPEC']
    best_aic = summary.sort_values(by=['AIC','BIC']).iloc[0]['SPEC']
    print(f"\nMelhor por BIC: {best_bic}")
    print(f"Melhor por AIC: {best_aic}")
else:
    print("Nenhum modelo foi estimado com sucesso. Verifique 'errors'.")

# =========================
# Testes LR (pares aninhados)
# =========================
def lr_test(spec_restricted, spec_full):
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

lr_rows = []
for sr, sf in NESTED_PAIRS:
    out = lr_test(sr, sf)
    if out: lr_rows.append(out)

if lr_rows:
    lr_df = pd.DataFrame(lr_rows).sort_values(by='LR', ascending=False)
    print("\n=== TESTES LR (pares aninhados) ===")
    print(lr_df.to_string(index=False))

