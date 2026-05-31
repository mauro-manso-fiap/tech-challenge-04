"""
Configurações globais do pipeline de analytics de obesidade.

Todas as constantes, ranges e mapeamentos do projeto ficam aqui.
Nenhum valor hard-coded deve existir fora deste módulo.
"""

# ---------------------------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------------------------

RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Schema do dataset
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS: list[str] = [
    "Gender",
    "Age",
    "Height",
    "Weight",
    "family_history",
    "FAVC",
    "FCVC",
    "NCP",
    "CAEC",
    "SMOKE",
    "CH2O",
    "SCC",
    "FAF",
    "TUE",
    "CALC",
    "MTRANS",
    "Obesity",
]

# ---------------------------------------------------------------------------
# Ranges numéricos válidos  {coluna: (min, max)}
# ---------------------------------------------------------------------------

NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "Age": (10.0, 100.0),
    "Height": (1.20, 2.20),
    "Weight": (20.0, 250.0),
    "FCVC": (0.0, 3.0),
    "NCP": (1.0, 4.0),
    "CH2O": (0.0, 3.0),
    "FAF": (0.0, 3.0),
    "TUE": (0.0, 2.0),
}

# ---------------------------------------------------------------------------
# Valores categóricos válidos  {coluna: [valores aceitos]}
# ---------------------------------------------------------------------------

CATEGORICAL_VALUES: dict[str, list[str]] = {
    "Gender": ["Male", "Female"],
    "family_history": ["yes", "no"],
    "FAVC": ["yes", "no"],
    "CAEC": ["no", "Sometimes", "Frequently", "Always"],
    "SMOKE": ["yes", "no"],
    "SCC": ["yes", "no"],
    "CALC": ["no", "Sometimes", "Frequently", "Always"],
    "MTRANS": [
        "Automobile",
        "Bike",
        "Motorbike",
        "Public_Transportation",
        "Walking",
    ],
    "Obesity": [
        "Insufficient_Weight",
        "Normal_Weight",
        "Overweight_Level_I",
        "Overweight_Level_II",
        "Obesity_Type_I",
        "Obesity_Type_II",
        "Obesity_Type_III",
    ],
}

# ---------------------------------------------------------------------------
# Rótulos PT-BR das classes de obesidade
# ---------------------------------------------------------------------------

OBESITY_LABELS_PT: dict[str, str] = {
    "Insufficient_Weight": "Abaixo do Peso",
    "Normal_Weight": "Peso Normal",
    "Overweight_Level_I": "Sobrepeso Nível I",
    "Overweight_Level_II": "Sobrepeso Nível II",
    "Obesity_Type_I": "Obesidade Tipo I",
    "Obesity_Type_II": "Obesidade Tipo II",
    "Obesity_Type_III": "Obesidade Tipo III",
}

# ---------------------------------------------------------------------------
# Opções PT-BR para inputs do Streamlit  {label_pt: valor_dataset}
# ---------------------------------------------------------------------------

GENDER_OPTIONS: dict[str, str] = {
    "Masculino": "Male",
    "Feminino": "Female",
}

CAEC_OPTIONS: dict[str, str] = {
    "Não": "no",
    "Às vezes": "Sometimes",
    "Frequentemente": "Frequently",
    "Sempre": "Always",
}

CALC_OPTIONS: dict[str, str] = {
    "Não": "no",
    "Às vezes": "Sometimes",
    "Frequentemente": "Frequently",
    "Sempre": "Always",
}

MTRANS_OPTIONS: dict[str, str] = {
    "Automóvel": "Automobile",
    "Bicicleta": "Bike",
    "Motocicleta": "Motorbike",
    "Transporte Público": "Public_Transportation",
    "Caminhada": "Walking",
}

# ---------------------------------------------------------------------------
# Informações clínicas das 7 classes de obesidade
# ---------------------------------------------------------------------------

RESULT_INFO: dict[str, dict] = {
    "Insufficient_Weight": {
        "label_pt": "Abaixo do Peso",
        "icon": "🔵",
        "color": "#3B82F6",
        "description": (
            "O peso corporal está abaixo do considerado saudável para a altura. "
            "Pode indicar desnutrição, distúrbios alimentares ou condições médicas subjacentes."
        ),
        "risks": [
            "Deficiências nutricionais (ferro, vitaminas, cálcio)",
            "Sistema imunológico enfraquecido",
            "Osteoporose e fragilidade óssea",
            "Anemia",
            "Problemas de fertilidade",
        ],
        "recommendations": [
            "Consultar nutricionista para plano alimentar hipercalórico saudável",
            "Realizar exames laboratoriais para identificar deficiências",
            "Praticar exercícios de resistência para ganho de massa muscular",
            "Monitorar peso regularmente com acompanhamento médico",
        ],
    },
    "Normal_Weight": {
        "label_pt": "Peso Normal",
        "icon": "🟢",
        "color": "#22C55E",
        "description": (
            "O peso corporal está dentro da faixa considerada saudável para a altura. "
            "Manter este estado é fundamental para a saúde a longo prazo."
        ),
        "risks": [
            "Risco cardiovascular e metabólico dentro do esperado para a faixa etária",
            "Atenção à composição corporal (percentual de gordura vs. massa magra)",
        ],
        "recommendations": [
            "Manter alimentação equilibrada e variada",
            "Praticar pelo menos 150 minutos de atividade física moderada por semana",
            "Realizar check-ups anuais preventivos",
            "Manter hidratação adequada (≥ 2 litros de água por dia)",
        ],
    },
    "Overweight_Level_I": {
        "label_pt": "Sobrepeso Nível I",
        "icon": "🟡",
        "color": "#EAB308",
        "description": (
            "O peso está levemente acima do ideal. Este estágio é um sinal de alerta "
            "e o momento mais eficaz para intervenção preventiva."
        ),
        "risks": [
            "Aumento do risco de hipertensão arterial",
            "Resistência à insulina incipiente",
            "Dores articulares (joelhos, quadril)",
            "Apneia do sono leve",
        ],
        "recommendations": [
            "Reduzir consumo de alimentos ultraprocessados e açúcares",
            "Aumentar atividade física para 200–300 min/semana",
            "Monitorar pressão arterial e glicemia regularmente",
            "Consultar nutricionista para ajuste alimentar",
        ],
    },
    "Overweight_Level_II": {
        "label_pt": "Sobrepeso Nível II",
        "icon": "🟠",
        "color": "#F97316",
        "description": (
            "O peso está moderadamente acima do ideal. Intervenção nutricional e "
            "aumento da atividade física são recomendados com urgência."
        ),
        "risks": [
            "Risco elevado de diabetes tipo 2",
            "Hipertensão arterial",
            "Dislipidemia (colesterol e triglicerídeos alterados)",
            "Síndrome metabólica",
            "Problemas articulares e musculoesqueléticos",
        ],
        "recommendations": [
            "Iniciar acompanhamento médico e nutricional estruturado",
            "Adotar dieta com déficit calórico moderado (300–500 kcal/dia)",
            "Praticar exercícios aeróbicos e de resistência regularmente",
            "Realizar exames metabólicos completos (glicemia, lipidograma, TSH)",
        ],
    },
    "Obesity_Type_I": {
        "label_pt": "Obesidade Tipo I",
        "icon": "🔴",
        "color": "#EF4444",
        "description": (
            "Obesidade grau I. O excesso de peso já representa risco significativo "
            "para a saúde cardiovascular e metabólica. Tratamento multidisciplinar é indicado."
        ),
        "risks": [
            "Alto risco de diabetes tipo 2",
            "Doença cardiovascular (infarto, AVC)",
            "Hipertensão arterial grave",
            "Apneia obstrutiva do sono",
            "Esteatose hepática (gordura no fígado)",
            "Refluxo gastroesofágico",
        ],
        "recommendations": [
            "Tratamento multidisciplinar: médico, nutricionista e educador físico",
            "Avaliação cardiológica e endocrinológica",
            "Programa estruturado de perda de peso (5–10% do peso corporal)",
            "Considerar terapia comportamental para mudança de hábitos",
        ],
    },
    "Obesity_Type_II": {
        "label_pt": "Obesidade Tipo II",
        "icon": "🔴",
        "color": "#DC2626",
        "description": (
            "Obesidade grau II. Risco muito elevado de comorbidades graves. "
            "Intervenção médica intensiva é necessária, podendo incluir farmacoterapia."
        ),
        "risks": [
            "Risco muito alto de eventos cardiovasculares fatais",
            "Diabetes tipo 2 estabelecido ou pré-diabetes avançado",
            "Insuficiência cardíaca",
            "Trombose venosa profunda",
            "Problemas renais crônicos",
            "Depressão e ansiedade associadas",
        ],
        "recommendations": [
            "Acompanhamento médico intensivo e regular",
            "Avaliação para farmacoterapia da obesidade",
            "Programa intensivo de mudança de estilo de vida",
            "Suporte psicológico para transtornos alimentares e saúde mental",
            "Monitoramento contínuo de comorbidades",
        ],
    },
    "Obesity_Type_III": {
        "label_pt": "Obesidade Tipo III",
        "icon": "🔴",
        "color": "#B91C1C",
        "description": (
            "Obesidade grau III (mórbida). Condição de alto risco que compromete "
            "significativamente a qualidade e expectativa de vida. "
            "Avaliação para cirurgia bariátrica pode ser indicada."
        ),
        "risks": [
            "Risco extremamente elevado de mortalidade cardiovascular",
            "Limitação funcional grave (mobilidade reduzida)",
            "Insuficiência respiratória e apneia grave",
            "Diabetes tipo 2 de difícil controle",
            "Hipertensão refratária",
            "Risco cirúrgico aumentado para qualquer procedimento",
        ],
        "recommendations": [
            "Encaminhamento urgente para equipe especializada em obesidade",
            "Avaliação para cirurgia bariátrica (critérios: IMC ≥ 40 ou ≥ 35 com comorbidades)",
            "Acompanhamento psiquiátrico e psicológico",
            "Programa de reabilitação física adaptada",
            "Monitoramento intensivo de todas as comorbidades",
        ],
    },
}
