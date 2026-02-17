"""
=============================================================================
Passos Mágicos — Aplicação Preditiva  |  app_streamlit.py
=============================================================================
Execute com:  streamlit run app_streamlit.py
Dependências: pip install streamlit joblib scikit-learn pandas numpy openpyxl
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Configuração da página
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Passos Mágicos — Predição de Ponto de Virada",
    page_icon="https://t3.gstatic.com/faviconV2?client=SOCIAL&type=FAVICON&fallback_opts=TYPE,SIZE,URL&url=http://passosmagicos.org.br&size=64",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS customizado com a identidade visual da Passos Mágicos
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Paleta principal */
    :root {
        --pm-blue:   #1A3A5C;
        --pm-yellow: #F5A623;
        --pm-green:  #27AE60;
        --pm-red:    #E74C3C;
        --pm-light:  #F8F9FA;
    }

    /* Header */
    .pm-header {
        background: linear-gradient(135deg, #1A3A5C 0%, #2980B9 100%);
        padding: 2rem 2rem 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .pm-header h1 { font-size: 2rem; margin: 0; font-weight: 700; }
    .pm-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 1rem; }

    /* Cartões de métricas */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 5px solid var(--pm-blue);
        margin-bottom: 0.8rem;
    }
    .metric-card.success { border-left-color: var(--pm-green); }
    .metric-card.warning { border-left-color: var(--pm-yellow); }
    .metric-card.danger  { border-left-color: var(--pm-red); }

    /* Resultado principal */
    .result-box {
        border-radius: 12px;
        padding: 1.5rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        text-align: center;
        margin: 1.2rem 0;
    }
    .result-sim  { background: #D5F5E3; color: #1E8449; border: 2px solid #27AE60; }
    .result-nao  { background: #FADBD8; color: #922B21; border: 2px solid #E74C3C; }

    /* Barra de probabilidade */
    .prob-label { font-size: 0.85rem; color: #555; margin-bottom: 0.2rem; }

    /* Separadores */
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--pm-blue);
        border-bottom: 2px solid var(--pm-yellow);
        padding-bottom: 0.3rem;
        margin: 1.2rem 0 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Carregamento do modelo
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = Path("models/pipeline_completo.pkl")
META_PATH  = Path("models/feature_names.pkl")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, None
    pipeline = joblib.load(MODEL_PATH)
    meta     = joblib.load(META_PATH) if META_PATH.exists() else {}
    return pipeline, meta


pipeline, meta = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — informações do modelo
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Sobre o Modelo")

    if meta:
        st.markdown(f"**Algoritmo  :** {meta.get('best_model_name', 'N/A')}")
        st.markdown(f"**AUC (teste):** {meta.get('test_auc', 'N/A')}")
        st.markdown(f"**F1-Score   :**    {meta.get('test_f1', 'N/A')}")
        st.markdown(f"**Acurácia   :**    {meta.get('test_accuracy', 'N/A')}")
    else:
        st.info("Modelo não carregado. Execute `ml_pipeline_passos_magicos.py` primeiro.")

    st.markdown("---")
    st.markdown("### Legenda de Indicadores")
    indicadores = {
        "INDE":  "Índice de Desenvolvimento Educacional",
        "IAA":   "Índice de Auto-Avaliação",
        "IEG":   "Índice de Engajamento",
        "IPS":   "Índice Psicossocial",
        "IDA":   "Índice de Desempenho Acadêmico",
        "IPV":   "Índice do Ponto de Virada",
        "IAN":   "Índice de Adequação ao Nível",
        "Defas": "Defasagem escolar (fase atual − fase ideal)",
    }
    for sigla, descricao in indicadores.items():
        st.markdown(f"**{sigla}** — {descricao}")

    st.markdown("---")
    st.markdown("**Passos Mágicos** © 2024  \n"
                "_Transformando vidas por meio da educação_")

# ─────────────────────────────────────────────────────────────────────────────
# Header principal
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="pm-header">
    <h1>Predição de Ponto de Virada</h1>
    <p>Ferramenta preditiva para identificar alunos com potencial de atingir
    o Ponto de Virada e apoiar decisões pedagógicas da equipe Passos Mágicos.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs principais
# ─────────────────────────────────────────────────────────────────────────────

tab_individual, tab_lote, tab_sobre = st.tabs([
    "🎯 Predição Individual",
    "📋 Predição em Lote (CSV/Excel)",
    "ℹ️ Sobre a Ferramenta",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDIÇÃO INDIVIDUAL
# ═══════════════════════════════════════════════════════════════════════════

with tab_individual:
    st.markdown('<div class="section-title">Dados do Aluno</div>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Dados Gerais**")
        fase = st.slider("Fase atual", min_value=0, max_value=8, value=2,
                         help="0 = Alfa, 1–7 = Fases regulares, 8 = Universitários")
        ano_ingresso = st.number_input("Ano de ingresso", min_value=2016,
                                        max_value=2024, value=2021, step=1)
        genero = st.selectbox("Gênero", ["Menina", "Menino"])
        instituicao = st.selectbox(
            "Instituição de ensino",
            ["Escola Pública", "Rede Decisão", "Escola Particular", "Outra"],
        )
        pedra = st.selectbox(
            "Pedra (classificação INDE)",
            ["Quartzo", "Ágata", "Ametista", "Topázio"],
            help="Quartzo < Ágata < Ametista < Topázio (melhor desempenho)",
        )
        defas = st.slider("Defasagem escolar", min_value=-5, max_value=5, value=-1,
                          help="Negativo = aluno adiantado; Positivo = atrasado")

    with col2:
        st.markdown("**Indicadores de Desempenho**")
        inde = st.slider("INDE", 0.0, 10.0, 7.0, 0.1)
        iaa  = st.slider("IAA  (Auto-avaliação)", 0.0, 10.0, 8.0, 0.1)
        ieg  = st.slider("IEG  (Engajamento)", 0.0, 10.0, 7.5, 0.1)
        ips  = st.slider("IPS  (Psicossocial)", 0.0, 10.0, 6.5, 0.1)

    with col3:
        st.markdown("**Indicadores Adicionais**")
        ida  = st.slider("IDA  (Desempenho Acadêmico)", 0.0, 10.0, 6.0, 0.1)
        ipv  = st.slider("IPV  (Ponto de Virada)", 0.0, 10.0, 7.0, 0.1)
        ian  = st.slider("IAN  (Adequação ao Nível)", 0.0, 10.0, 5.0, 0.5)
        ano_ref = st.selectbox("Ano de referência", [2022, 2023, 2024], index=2)

    st.markdown("---")
    btn_predict = st.button("🔮 Gerar Predição", type="primary", use_container_width=True)

    if btn_predict:
        if pipeline is None:
            st.error("⚠️ Modelo não encontrado. Execute `ml_pipeline_passos_magicos.py` primeiro.")
        else:
            # Monta DataFrame de entrada
            input_data = pd.DataFrame([{
                "fase":                  fase,
                "inde":                  inde,
                "iaa":                   iaa,
                "ieg":                   ieg,
                "ips":                   ips,
                "ida":                   ida,
                "ipv":                   ipv,
                "ian":                   ian,
                "defas":                 defas,
                "ano_ingresso":          ano_ingresso,
                "ano_referencia":        ano_ref,
                "genero":                genero,
                "pedra":                 pedra,
                "instituicao_de_ensino": instituicao,
            }])

            # Garante que só features usadas no treino sejam passadas
            if meta:
                all_feats = meta.get("all_features", input_data.columns.tolist())
                input_data = input_data[[c for c in all_feats if c in input_data.columns]]

            prob = pipeline.predict_proba(input_data)[0]
            pred = pipeline.predict(input_data)[0]
            prob_sim = prob[1]
            prob_nao = prob[0]

            # ── Resultado ────────────────────────────────────────────
            st.markdown("### Resultado da Predição")
            res_cols = st.columns([2, 1])

            with res_cols[0]:
                if pred == 1:
                    st.markdown(
                        f'<div class="result-box result-sim">'
                        f'✅ O aluno tem <strong>ALTA probabilidade</strong> de '
                        f'atingir o Ponto de Virada<br>'
                        f'<span style="font-size:1.8rem">{prob_sim:.1%}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="result-box result-nao">'
                        f'⚠️ O aluno tem <strong>BAIXA probabilidade</strong> de '
                        f'atingir o Ponto de Virada<br>'
                        f'<span style="font-size:1.8rem">{prob_sim:.1%}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # Barra de probabilidade
                st.markdown('<p class="prob-label">Probabilidade de Atingir o PV</p>',
                            unsafe_allow_html=True)
                st.progress(float(prob_sim))
                st.caption(f"Sim: **{prob_sim:.1%}** | Não: **{prob_nao:.1%}**")

            with res_cols[1]:
                # Resumo dos indicadores
                st.markdown("**Indicadores informados:**")
                indicators = {
                    "INDE": inde, "IAA": iaa, "IEG": ieg,
                    "IPS": ips,   "IDA": ida, "IPV": ipv, "IAN": ian,
                }
                for k, v in indicators.items():
                    color = "#27AE60" if v >= 7 else ("#F5A623" if v >= 5 else "#E74C3C")
                    st.markdown(
                        f'<div class="metric-card" style="padding:0.5rem 1rem;'
                        f'border-left-color:{color}; color: #000;">'
                        f'<b>{k}</b>: {v:.1f}</div>',
                        unsafe_allow_html=True,
                    )

            # ── Recomendações ────────────────────────────────────────
            st.markdown('<div class="section-title">Recomendações Pedagógicas</div>',
                        unsafe_allow_html=True)

            recs = []
            if iaa < 6:
                recs.append("**Auto-avaliação (IAA)** baixa — promover atividades de autoconhecimento e valorização pessoal.")
            if ieg < 6:
                recs.append("**Engajamento (IEG)** abaixo do esperado — incentivar participação nas aulas e entrega de atividades.")
            if ida < 5:
                recs.append("**Desempenho Acadêmico (IDA)** crítico — considerar reforço em Matemática e Português.")
            if ips < 5.5:
                recs.append("**Indicador Psicossocial (IPS)** reduzido — acionar suporte psicopedagógico.")
            if defas > 0:
                recs.append(f"**Defasagem escolar de {defas} fase(s)** — monitorar progressão e avaliar nivelamento.")
            if ipv < 6:
                recs.append("**IPV** baixo — trabalhar motivação, liderança e protagonismo com o aluno.")

            if not recs:
                st.success("Aluno com indicadores saudáveis! Manter acompanhamento regular e considerar indicação para bolsa/programa avançado.")
            else:
                for r in recs:
                    st.warning(r)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDIÇÃO EM LOTE
# ═══════════════════════════════════════════════════════════════════════════

with tab_lote:
    st.markdown('<div class="section-title">Upload de Arquivo</div>',
                unsafe_allow_html=True)

    st.info(
        "📎 Envie um arquivo CSV ou Excel com os indicadores dos alunos. "
        "As colunas devem seguir os mesmos nomes usados no treinamento: "
        "`fase`, `inde`, `iaa`, `ieg`, `ips`, `ida`, `ipv`, `ian`, `defas`, "
        "`ano_ingresso`, `ano_referencia`, `genero`, `pedra`, `instituicao_de_ensino`."
    )

    uploaded_file = st.file_uploader(
        "Selecione o arquivo", type=["csv", "xlsx", "xls"]
    )

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)

            st.success(f"✅ Arquivo carregado: {len(df_upload)} registros")
            st.dataframe(df_upload.head(5), use_container_width=True)

            if pipeline is None:
                st.error("⚠️ Modelo não encontrado.")
            else:
                btn_lote = st.button("Gerar Predições em Lote", type="primary")
                if btn_lote:
                    if meta:
                        all_feats = meta.get("all_features", [])
                        cols_ok   = [c for c in all_feats if c in df_upload.columns]
                        cols_miss = [c for c in all_feats if c not in df_upload.columns]
                        if cols_miss:
                            st.warning(f"⚠️ Colunas ausentes (serão imputadas): {cols_miss}")
                        df_input = df_upload[cols_ok].copy() if cols_ok else df_upload.copy()
                    else:
                        df_input = df_upload.copy()

                    probs = pipeline.predict_proba(df_input)[:, 1]
                    preds = pipeline.predict(df_input)

                    df_resultado = df_upload.copy()
                    df_resultado["prob_ponto_de_virada"] = np.round(probs, 4)
                    df_resultado["predicao_pv"] = np.where(preds == 1, "Sim", "Não")

                    # Ordenar por probabilidade
                    df_resultado = df_resultado.sort_values(
                        "prob_ponto_de_virada", ascending=False
                    ).reset_index(drop=True)

                    st.markdown("### 📊 Resultados")

                    kpi1, kpi2, kpi3 = st.columns(3)
                    n_sim  = (preds == 1).sum()
                    n_nao  = (preds == 0).sum()
                    pct_pv = n_sim / len(preds)
                    kpi1.metric("Total de alunos", len(preds))
                    kpi2.metric("Atingirão PV (predito)", f"{n_sim} ({pct_pv:.1%})")
                    kpi3.metric("Precisam de atenção", f"{n_nao} ({1-pct_pv:.1%})")

                    st.dataframe(
                        df_resultado.style.background_gradient(
                            subset=["prob_ponto_de_virada"], cmap="RdYlGn"
                        ),
                        use_container_width=True,
                    )

                    csv_out = df_resultado.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Baixar resultados (.csv)",
                        data=csv_out,
                        file_name="predicoes_ponto_de_virada.csv",
                        mime="text/csv",
                    )
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {e}")

    else:
        # Template para download
        st.markdown("### Template de Arquivo")
        template_data = {
            "ra":                    ["RA-001", "RA-002"],
            "fase":                  [2, 3],
            "inde":                  [7.2, 6.5],
            "iaa":                   [8.0, 7.5],
            "ieg":                   [7.5, 6.0],
            "ips":                   [6.5, 5.5],
            "ida":                   [6.0, 5.0],
            "ipv":                   [7.0, 6.5],
            "ian":                   [5.0, 10.0],
            "defas":                 [-1, 0],
            "ano_ingresso":          [2021, 2020],
            "ano_referencia":        [2024, 2024],
            "genero":                ["Menina", "Menino"],
            "pedra":                 ["Ametista", "Quartzo"],
            "instituicao_de_ensino": ["Escola Pública", "Rede Decisão"],
        }
        df_template = pd.DataFrame(template_data)
        st.dataframe(df_template, use_container_width=True)
        csv_template = df_template.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Baixar template (.csv)",
            data=csv_template,
            file_name="template_predicao.csv",
            mime="text/csv",
        )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — SOBRE A FERRAMENTA
# ═══════════════════════════════════════════════════════════════════════════

with tab_sobre:
    st.markdown("""
    ## Sobre a Ferramenta Preditiva

    Esta aplicação foi desenvolvida como parte do **Datathon FIAP — Passos Mágicos**,
    com o objetivo de colocar Inteligência Artificial a serviço da missão social da
    **Associação Passos Mágicos**.

    ---

    ### Objetivo
    Prever se um aluno tem potencial de atingir o **Ponto de Virada** — um marco
    transformador no desenvolvimento educacional que indica que o aluno internalizou
    os valores e princípios do programa e está pronto para uma nova etapa de crescimento.

    ---

    ### Como funciona o modelo?
    O modelo de Machine Learning foi treinado com dados históricos dos ciclos **2022,
    2023 e 2024** do PEDE (Pesquisa Extensiva do Desenvolvimento Educacional).
    Utiliza os seguintes indicadores como entrada:

    | Indicador | Descrição |
    |-----------|-----------|
    | **Fase** | Fase atual do aluno no programa (0–8) |
    | **INDE**  | Índice de Desenvolvimento Educacional (principal KPI) |
    | **IAA**   | Auto-avaliação do aluno |
    | **IEG**   | Engajamento com o programa |
    | **IPS**   | Bem-estar psicossocial |
    | **IDA**   | Desempenho nas avaliações acadêmicas |
    | **IPV**   | Indicador específico do Ponto de Virada |
    | **IAN**   | Adequação à fase/nível esperado |
    | **Defas** | Defasagem escolar em relação à fase ideal |

    ---

    ### Limitações e Uso Responsável
    - As predições são **probabilísticas**, não determinísticas.
    - O modelo **não substitui** o julgamento dos educadores e psicopedagogos.
    - Use os resultados como **apoio à decisão**, nunca como critério único.
    - Dados sensíveis de alunos devem ser tratados conforme a **LGPD**.

    ---

    ### Tecnologias utilizadas
    `Python` · `scikit-learn` · `pandas` · `Streamlit` · `joblib`

    ---

    *Desenvolvido com ❤️ para a Passos Mágicos — transformando vidas pela educação.*
    """)