"""
=============================================================================
Passos Mágicos -- Predição de Risco de Defasagem  |  app_streamlit.py
=============================================================================
Modelo preditivo de risco de defasagem escolar.

Execute : streamlit run app_streamlit.py
Deps    : pip install streamlit joblib scikit-learn pandas numpy openpyxl

Arquivos esperados em models/
  pipeline.pkl        pipeline completo (preprocessador + classificador)
  melhor_modelo.pkl   classificador isolado
  feature_names.pkl   metadados: features, métricas, parâmetros
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# -----------------------------------------------------------------------------
# Configuração da pagina
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Predição de Risco de Defasagem",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# CSS
# -----------------------------------------------------------------------------

st.markdown("""
<style>
:root {
    --pm-blue:   #1A3A5C;
    --pm-yellow: #F5A623;
    --pm-green:  #1E8449;
    --pm-red:    #C0392B;
}

.pm-header {
    background: linear-gradient(135deg, #1A3A5C 0%, #2471A3 100%);
    padding: 1.6rem 2rem;
    border-radius: 10px;
    color: white;
    margin-bottom: 1.4rem;
}
.pm-header h1 { font-size: 1.8rem; margin: 0; font-weight: 700; }
.pm-header p  { margin: .35rem 0 0; opacity: .85; font-size: .92rem; }

.result-card {
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin: .5rem 0;
    font-size: 1rem;
}
.card-green  { background: #D5F5E3; border: 2px solid #27AE60; color: #1A5C35; }
.card-yellow { background: #FEF9E7; border: 2px solid #F5A623; color: #7D6608; }
.card-red    { background: #FADBD8; border: 2px solid #E74C3C; color: #7B241C; }
.card-blue   { background: #D6EAF8; border: 2px solid #2E86C1; color: #1A3A5C; }
.card-gray   { background: #F2F3F4; border: 2px solid #AEB6BF; color: #2C3E50; }

.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #1A3A5C;
    border-bottom: 2px solid #F5A623;
    padding-bottom: .3rem;
    margin: 1.2rem 0 .8rem;
}

.prob-value {
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.1;
}

.badge {
    display: inline-block;
    border-radius: 20px;
    padding: .2rem .8rem;
    font-size: .8rem;
    font-weight: 700;
    margin: .2rem .1rem;
}
.badge-green  { background: #27AE60; color: white; }
.badge-yellow { background: #F5A623; color: white; }
.badge-red    { background: #E74C3C; color: white; }
.badge-gray   { background: #95A5A6; color: white; }

table.info-table { width: 100%; border-collapse: collapse; font-size: .9rem; }
table.info-table th {
    background: #1A3A5C; color: white;
    padding: .5rem .8rem; text-align: left;
}
table.info-table td { padding: .45rem .8rem; border-bottom: 1px solid #E8E8E8; }
table.info-table tr:nth-child(even) td { background: #F8F9FA; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Carregamento do modelo
# -----------------------------------------------------------------------------

MODEL_DIR = Path("models")


@st.cache_resource(show_spinner="Carregando modelo...")
def load_model():
    pipeline_path = MODEL_DIR / "pipeline.pkl"
    meta_path     = MODEL_DIR / "feature_names.pkl"

    if not pipeline_path.exists():
        return None, None

    pipeline = joblib.load(pipeline_path)
    meta     = joblib.load(meta_path) if meta_path.exists() else {}
    return pipeline, meta


pipeline, meta = load_model()
model_ok = pipeline is not None

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## Passos Mágicos")
    st.markdown("_Transformando vidas pela educação_")
    st.markdown("---")

    # Status do modelo
    st.markdown("### Status do Modelo")
    if model_ok:
        model_name = meta.get("best_model_name", meta.get("model_name", "N/D"))
        st.markdown(
            f'<div class="result-card card-green" style="padding:.6rem 1rem">'
            f'<b>Modelo carregado</b><br>'
            f'<small>'
            f'Algoritmo: {model_name}<br>'
            f'AUC: {meta.get("test_auc", "N/D")} | '
            f'Recall: {meta.get("test_recall", meta.get("recall", "N/D"))} | '
            f'F1: {meta.get("test_f1", meta.get("f1", "N/D"))}'
            f'</small></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-card card-red" style="padding:.6rem 1rem">'
            '<b>Modelo não encontrado</b><br>'
            '<small>Execute ml_pipeline.ipynb '
            'para gerar os arquivos em models/</small></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Glossario
    st.markdown("### Indicadores")
    for sig, desc in [
        ("INDE", "Desenvolvimento Educacional"),
        ("IAA",  "Auto-Avaliação"),
        ("IEG",  "Engajamento"),
        ("IPS",  "Psicossocial"),
        ("IDA",  "Desempenho Acadêmico"),
        ("IAN",  "Adequação ao Nível"),
    ]:
        st.markdown(f"**{sig}** -- {desc}")

    st.markdown("---")
    st.markdown(
        '<div class="result-card card-gray" style="padding:.6rem 1rem;font-size:.8rem">'
        'Os indicadores IPV e Defasagem não sao usados como entrada do modelo: '
        'IPV mede o Ponto de Virada (conceito distinto); '
        'Defasagem é o próprio target que o modelo aprende a prever.'
        '</div>',
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Header principal
# -----------------------------------------------------------------------------

st.markdown("""
<div class="pm-header">
    <h1>Predicao de Risco de Defasagem Escolar</h1>
    <p>
    Identifica alunos com alta probabilidade de entrar em defasagem escolar
    (fase real atrasada em re;a a fase ideal para a idade),
    permitindo Intervenção pedagógica preventiva.
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------

tab_ind, tab_lote, tab_sobre = st.tabs([
    "Analise Individual",
    "Analise em Lote",
    "Sobre o Modelo",
])

# ============================================================================
# TAB 1 -- ANALISE INDIVIDUAL
# ============================================================================

with tab_ind:

    st.markdown('<div class="section-title">Dados do Aluno</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Perfil**")
        fase = st.slider(
            "Fase atual", 0, 8, 2,
            help="0 = Alfa | 1-5 = Ensino Fundamental | 6-7 = Ensino Medio | 8 = Universitario",
        )
        ano_ingresso = st.number_input(
            "Ano de ingresso", min_value=2016, max_value=2024, value=2021, step=1,
        )
        genero = st.selectbox("Gênero", ["Menina", "Menino"])
        instituicao_de_ensino = st.selectbox(
            "Instituição de ensino",
            ["Escola Publica", "Rede Decisão", "Escola Particular", "Outra"],
        )
        pedra = st.selectbox(
            "Pedra (Classificação INDE)",
            ["Quartzo", "Agata", "Ametista", "Topazio"],
            help="Quartzo < Agata < Ametista < Topazio (melhor desempenho)",
        )
        ano_ref = st.selectbox("Ano de referencia", [2022, 2023, 2024], index=2)

    with col2:
        st.markdown("**Indicadores**")
        inde = st.slider("INDE",  0.0, 10.0, 7.0, 0.1)
        iaa  = st.slider("IAA",   0.0, 10.0, 8.0, 0.1)
        ieg  = st.slider("IEG",   0.0, 10.0, 7.5, 0.1)
        ips  = st.slider("IPS",   0.0, 10.0, 6.5, 0.1)
        ida  = st.slider("IDA",   0.0, 10.0, 6.0, 0.1)

    with col3:
        st.markdown("**Indicadores adicionais**")
        ian  = st.slider("IAN",  0.0, 10.0, 5.0, 0.5)
        defas_ref = st.slider(
            "Defasagem atual (referencia)",
            min_value=-5, max_value=5, value=-1,
            help=(
                "Apenas referencia visual. não e usada como feature de entrada -- "
                "defasagem > 0 e o próprio target que o modelo aprende a prever."
            ),
        )
        st.caption(
            "Defasagem = fase real - fase ideal. "
            "Negativo = adiantado. Positivo = atrasado. "
            "Este campo não influencia a predição."
        )

    st.markdown("---")
    btn = st.button("Gerar Predição", type="primary", width='stretch')

    if btn:
        if not model_ok:
            st.error(
                "Modelo não carregado. "
                "Execute ml_pipeline_passos_Mágicos.py primeiro."
            )
            st.stop()

        # Monta o DataFrame de entrada -- sem IPV, sem defas
        input_data = pd.DataFrame([{
            "fase":                  fase,
            "inde":                  inde,
            "iaa":                   iaa,
            "ieg":                   ieg,
            "ips":                   ips,
            "ida":                   ida,
            "ian":                   ian,
            "ano_ingresso":          ano_ingresso,
            "ano_referencia":        ano_ref,
            "genero":                genero,
            "pedra":                 pedra,
            "instituicao_de_ensino": instituicao_de_ensino,
        }])

        # Filtra para as features usadas no treino
        all_feats = meta.get(
            "all_features",
            meta.get("numeric_features", []) + meta.get("categorical_features", []),
        )
        cols_ok = [c for c in all_feats if c in input_data.columns]
        X = input_data[cols_ok]

        prob = float(pipeline.predict_proba(X)[0, 1])

        print(pipeline.predict_proba(X))

        # -- Resultado principal -----------------------------------------
        st.markdown("## Resultado")

        res_col, detail_col = st.columns([2, 1])

        with res_col:
            if prob >= 0.65:
                cls  = "card-red"
                Nível = "ALTO"
                Descrição = (
                    "O modelo identifica alta probabilidade de defasagem. "
                    "Recomenda-se Intervenção pedagógica imediata."
                )
            elif prob >= 0.40:
                cls  = "card-yellow"
                Nível = "MODERADO"
                Descrição = (
                    "Risco intermediário. Monitorar de perto e acionar suporte "
                    "preventivo caso os indicadores se deteriorem."
                )
            else:
                cls  = "card-green"
                Nível = "BAIXO"
                Descrição = (
                    "Baixa probabilidade de defasagem no ciclo atual. "
                    "Manter acompanhamento regular."
                )

            st.markdown(
                f'<div class="result-card {cls}">'
                f'<div style="font-size:.85rem;font-weight:600;'
                f'text-transform:uppercase;letter-spacing:.05em;opacity:.75">'
                f'Risco de Defasagem</div>'
                f'<div class="prob-value">{prob:.1%}</div>'
                f'<div style="font-size:1.05rem;font-weight:700;margin:.3rem 0">'
                f'Nível: {Nível}</div>'
                f'<div style="font-size:.9rem">{Descrição}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.caption("Probabilidade estimada de entrar em risco de defasagem")
            st.progress(prob)

            # Badge da situacao atual informada (so referencia)
            st.markdown("**Situacao atual informada:**", unsafe_allow_html=False)
            if defas_ref > 0:
                st.markdown(
                    f'<span class="badge badge-red">Defasagem: +{defas_ref} fase(s)</span>',
                    unsafe_allow_html=True,
                )
            elif defas_ref == 0:
                st.markdown(
                    '<span class="badge badge-yellow">No Nível esperado</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<span class="badge badge-green">'
                    f'Adiantado: {abs(defas_ref)} fase(s)</span>',
                    unsafe_allow_html=True,
                )

        with detail_col:
            st.markdown("**Resumo dos indicadores**")
            REFS = {"INDE": 7.0, "IAA": 7.0, "IEG": 7.0,
                    "IPS": 5.5, "IDA": 5.0, "IAN": 5.0}
            for name, val in [
                ("INDE", inde), ("IAA", iaa), ("IEG", ieg),
                ("IPS",  ips),  ("IDA", ida), ("IAN", ian),
            ]:
                ref = REFS[name]
                cor = "#27AE60" if val >= ref else ("#F5A623" if val >= ref * 0.75 else "#E74C3C")
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:.3rem .6rem;border-left:4px solid {cor};'
                    f'margin-bottom:.3rem;background:#FAFAFA;border-radius:0 6px 6px 0">'
                    f'<span style="font-weight:600;color:black">{name}</span>'
                    f'<span style="color:{cor};font-weight:700">{val:.1f}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # -- Recomenda -----------------------------------------------
        st.markdown("---")
        st.markdown('<div class="section-title">Recomendações pedagógicas</div>',
                    unsafe_allow_html=True)

        recs = []

        if prob >= 0.65:
            recs.append(("card-red",
                "Risco alto de defasagem detectado -> acionar suporte pedagógico "
                "imediatamente. Verificar frequência, entregas e fatores externos."))

        if defas_ref > 0:
            recs.append(("card-red",
                f"Defasagem atual de +{defas_ref} fase(s) -> planejar Nívelamento "
                "e monitorar a progressao bimestralmente."))

        if ida < 5.0:
            recs.append(("card-red",
                f"IDA critico ({ida:.1f}/10) -> reforco urgente em Matemática "
                "e/ou Português."))

        if ieg < 5.5:
            recs.append(("card-yellow",
                f"Engajamento baixo (IEG = {ieg:.1f}) -> investigar motivação e "
                "incentivar participacao ativa nas atividades."))

        if ips < 5.0:
            recs.append(("card-yellow",
                f"Indicador psicossocial reduzido (IPS = {ips:.1f}) -> avaliar "
                "bem-estar do aluno e considerar encaminhamento para psicopedagogia."))

        if iaa < 6.0:
            recs.append(("card-blue",
                f"Auto-Avaliação baixa (IAA = {iaa:.1f}) -> trabalhar autoconfianca "
                "e protagonismo do aluno."))

        if prob < 0.40 and not recs:
            st.success(
                "Aluno com indicadores saudaveis e baixo risco de defasagem. "
                "Manter acompanhamento regular."
            )
        else:
            for cls, msg in recs:
                st.markdown(
                    f'<div class="result-card {cls}" style="padding:.65rem 1rem">'
                    f'{msg}</div>',
                    unsafe_allow_html=True,
                )

        # -- Painel detalhado --------------------------------------------
        with st.expander("Ver painel detalhado dos indicadores"):
            cols4 = st.columns(4)
            for i, (name, val, ref) in enumerate([
                ("INDE", inde, 7.0), ("IAA", iaa, 7.0),
                ("IEG",  ieg,  7.0), ("IPS", ips, 5.5),
                ("IDA",  ida,  5.0), ("IAN", ian, 5.0),
            ]):
                with cols4[i % 4]:
                    st.metric(
                        label=name,
                        value=f"{val:.1f}",
                        delta=f"{val - ref:+.1f} vs ref {ref}",
                        delta_color="normal" if val >= ref else "inverse",
                    )


# ============================================================================
# TAB 2 -- ANALISE EM LOTE
# ============================================================================

with tab_lote:
    st.markdown('<div class="section-title">Upload de Arquivo</div>',
                unsafe_allow_html=True)

    st.info(
        "Envie um arquivo CSV ou Excel com os dados dos alunos. "
        "Colunas esperadas: fase, inde, iaa, ieg, ips, ida, ian, "
        "ano_ingresso, ano_referencia, Gênero, pedra, Instituição_de_ensino. "
        "A coluna defas e opcional -- não e usada como feature, "
        "apenas exibida nos resultados para referencia."
    )

    uploaded = st.file_uploader(
        "Selecione o arquivo", type=["csv", "xlsx", "xls"]
    )

    if uploaded:
        try:
            df_up = (
                pd.read_csv(uploaded)
                if uploaded.name.endswith(".csv")
                else pd.read_excel(uploaded)
            )
            st.success(f"{len(df_up)} registros carregados.")
            st.dataframe(df_up.head(5), width='stretch')

            if not model_ok:
                st.error("Modelo não dispoNível. Execute o pipeline primeiro.")
            else:
                if st.button("Processar Lote", type="primary"):
                    # Seleciona features do modelo
                    all_feats = meta.get(
                        "all_features",
                        meta.get("numeric_features", []) + meta.get("categorical_features", []),
                    )
                    cols_ok   = [c for c in all_feats if c in df_up.columns]
                    cols_miss = [c for c in all_feats if c not in df_up.columns]

                    if cols_miss:
                        st.warning(
                            f"Colunas ausentes no arquivo (serao imputadas pelo modelo): "
                            f"{', '.join(cols_miss)}"
                        )

                    probs  = pipeline.predict_proba(df_up[cols_ok])[:, 1]
                    result = df_up.copy()
                    result["prob_risco_defasagem"] = np.round(probs, 4)
                    result["Nível_risco"] = pd.cut(
                        probs,
                        bins=[0.0, 0.40, 0.65, 1.0],
                        labels=["Baixo", "Moderado", "Alto"],
                    )
                    result["acao_recomendada"] = np.where(
                        probs >= 0.65, "Intervenção prioritaria",
                        np.where(probs >= 0.40, "Monitoramento preventivo",
                                 "Acompanhamento regular"),
                    )

                    # Ordena do maior risco para o menor
                    result = result.sort_values(
                        "prob_risco_defasagem", ascending=False
                    ).reset_index(drop=True)

                    # KPIs
                    k1, k2, k3, k4 = st.columns(4)
                    n_alto = (probs >= 0.65).sum()
                    n_mod  = ((probs >= 0.40) & (probs < 0.65)).sum()
                    n_baixo = (probs < 0.40).sum()

                    k1.metric("Total de alunos",       len(result))
                    k2.metric("Risco Alto",  f"{n_alto}  ({n_alto/len(result):.0%})")
                    k3.metric("Risco Moderado", f"{n_mod}  ({n_mod/len(result):.0%})")
                    k4.metric("Risco Baixo",  f"{n_baixo} ({n_baixo/len(result):.0%})")

                    st.dataframe(result, width='stretch')

                    st.download_button(
                        "Baixar resultados (.csv)",
                        data=result.to_csv(index=False).encode("utf-8"),
                        file_name="risco_defasagem_passos_Mágicos.csv",
                        mime="text/csv",
                    )

        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")

    else:
        # Template para download
        st.markdown("### Template de Arquivo")
        st.markdown(
            "Baixe o template abaixo, preencha com os dados dos alunos "
            "e faça o upload acima."
        )
        template = pd.DataFrame({
            "ra":                    ["RA-001", "RA-002"],
            "fase":                  [2, 3],
            "inde":                  [7.2, 5.8],
            "iaa":                   [8.0, 6.5],
            "ieg":                   [7.5, 5.0],
            "ips":                   [6.5, 5.5],
            "ida":                   [6.0, 4.2],
            "ian":                   [5.0, 10.0],
            "defas":                 [-1, 1],
            "ano_ingresso":          [2021, 2020],
            "ano_referencia":        [2024, 2024],
            "Gênero":                ["Menina", "Menino"],
            "pedra":                 ["Ametista", "Quartzo"],
            "Instituição_de_ensino": ["Escola Publica", "Escola Publica"],
        })
        st.dataframe(template, width='stretch')
        st.download_button(
            "Baixar template (.csv)",
            data=template.to_csv(index=False).encode("utf-8"),
            file_name="template_risco_defasagem.csv",
            mime="text/csv",
        )


# ============================================================================
# TAB 3 -- SOBRE O MODELO
# ============================================================================

with tab_sobre:
    st.markdown("## Sobre o Modelo")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
### Objetivo

Prever se um aluno esta em trajetória de defasagem escolar crescente,
ou seja, se a fase real do aluno ficara atrasada em relação a fase
ideal esperada para sua idade.

**Target de treinamento:** `Defas > 0`

**Scoring utilizado:** F1
_(equilibra Recall e Precisão; Recall e monitorado como métrica operacional
pois não identificar um aluno em risco -- falso negativo -- e o erro mais custoso)_

### Features de entrada

| Indicador | Descrição |
|-----------|-----------|
| Fase | Fase atual no programa (0-8) |
| INDE | Índice de Desenvolvimento Educacional |
| IAA | Auto-Avaliação |
| IEG | Engajamento |
| IPS | Psicossocial |
| IDA | Desempenho Acadêmico |
| IAN | Adequação ao Nível |
| Ano ingresso | Ano de entrada no programa |
| Ano referencia | Ano do ciclo avaliado |
| Gênero | Gênero do aluno |
| Pedra | Classificação pelo INDE |
| Instituição | Tipo de escola |
        """)

    with col_b:
        st.markdown("""
### O que não entra como feature e por que

**IPV (Índice do Ponto de Virada)**
Mede um conceito distinto -- o grau de transformação do aluno pelo programa.
não e indicador de defasagem escolar.

**Defasagem (Defas)**
é o próprio target que o modelo aprende a prever (`Defas > 0`).
Incluí-la como feature causaria data leakage direto -- o modelo
aprenderia a resposta em vez do padrão.

### Interpretação do resultado

| Probabilidade | Nível | Ação recomendada |
|---|---|---|
| >= 65% | Alto | Intervenção pedagógica imediata |
| 40% - 64% | Moderado | Monitoramento preventivo |
| < 40% | Baixo | Acompanhamento regular |

### Arquivos do modelo

```
models/
  pipeline.pkl        pipeline completo para inferência
  melhor_modelo.pkl   classificador isolado
  feature_names.pkl   metadados, métricas e features
```

### Uso responsável

- Predições sao probabilísticas, não determinísticas
- O modelo ml_pipeline não substitui o julgamento pedagógico
- Dados de alunos devem ser tratados conforme a LGPD
- Retreinar a cada ciclo com dados novos mantém a acurácia

---
_Datathon FIAP -- © 2026_
        """)