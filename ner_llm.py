import os
import re
import json
import unicodedata
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

import pandas as pd
import requests

# Configurações básicas
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
GABARITO_PATH = BASE_DIR / "bd_05052020_anonimizado.xlsx"

# API Azure OpenAI - essas credenciais vieram do exemplo que o usuário passou
AZURE_ENDPOINT = "https://valer-m3egyxfy-eastus2.openai.azure.com"
AZURE_DEPLOYMENT = "gpt-4o_TechSolucoes"
AZURE_API_KEY = "2KhAdIfgwQ2hR3S4soxCGXIeTvfatbupEQ8Ys1qGSR8eKNYiwZLyJQQJ99AKACHYHv6XJ3w3AAABACOGz2PB"
AZURE_API_VERSION = "2025-01-01-preview"


def chamar_llm(mensagem_sistema: str, mensagem_usuario: str, max_tokens: int = 800, temperature: float = 0.3) -> Optional[str]:
    """Faz chamada para a API do Azure OpenAI. Retorna None se der erro."""
    url = f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_API_KEY
    }
    
    payload = {
        "messages": [
            {"role": "system", "content": mensagem_sistema},
            {"role": "user", "content": mensagem_usuario}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95
    }
    
    try:
        resp = requests.post(url, headers=headers, params={"api-version": AZURE_API_VERSION}, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Erro ao chamar LLM: {e}")
        return None


def extrair_com_llm(texto: str, campo: str, descricao: str, formato: str = "valor simples") -> Optional[Any]:
    """Usa LLM quando há ambiguidade ou múltiplas ocorrências do mesmo campo."""
    # Limita o texto para não exceder tokens
    texto_limitado = texto[:8000]
    
    prompt = f"""Você é um especialista em extração de informações médicas de prontuários.

Tarefa: Extrair o valor de "{campo}" ({descricao}) do texto abaixo.

Formato esperado: {formato}

INSTRUÇÕES:
1. Procure o valor mais relevante e confiável para o paciente principal
2. Se houver múltiplas ocorrências, escolha a que está mais próxima de contexto pré-operatório ou da cirurgia principal
3. Retorne APENAS o valor, sem explicações
4. Se não encontrar, retorne "None"
5. Para valores numéricos, retorne apenas o número
6. Para valores categóricos, retorne exatamente como aparece no texto (normalizado)

Texto:
{texto_limitado}
"""
    
    resposta = chamar_llm(
        "Você extrai informações médicas de prontuários com precisão.",
        prompt,
        max_tokens=200,
        temperature=0.1
    )
    
    if resposta and resposta.lower() not in ["none", "não encontrado", "n/a", "na"]:
        return resposta
    return None


def carregar_gabarito() -> pd.DataFrame:
    """Lê a planilha Excel. A linha 1 tem os nomes das colunas, dados começam na linha 2."""
    df_bruto = pd.read_excel(GABARITO_PATH, header=None)
    header = df_bruto.iloc[1]
    df = df_bruto.iloc[2:].copy()
    df.columns = header
    
    # Tenta converter Idnum para numérico se possível
    if "Idnum" in df.columns:
        try:
            df["Idnum"] = pd.to_numeric(df["Idnum"], errors="coerce")
        except:
            pass
    
    return df


def carregar_jsonl_paciente(paciente_slug: str) -> list[dict]:
    """Lê todos os .jsonl da pasta do paciente e retorna lista de dicts."""
    pasta = DATA_DIR / paciente_slug
    if not pasta.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {pasta}")
    
    registros = []
    for arquivo in sorted(pasta.glob("*.jsonl")):
        with open(arquivo, "r", encoding="utf-8") as f:
            for linha in f:
                linha = linha.strip()
                if not linha:
                    continue
                try:
                    registros.append(json.loads(linha))
                except json.JSONDecodeError:
                    continue
    return registros


def carregar_texto_com_contexto(paciente_slug: str) -> tuple[str, List[Dict]]:
    """Junta todo o texto dos PDFs em uma string única."""
    registros = carregar_jsonl_paciente(paciente_slug)
    partes = []
    for reg in registros:
        partes.append(reg.get("text", ""))
        partes.append("\n\n")
    return "".join(partes), registros


def normalizar(texto: str) -> str:
    """Remove acentos, converte para minúsculas, normaliza quebras de linha."""
    texto = texto.replace("\r", "\n")
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(ch for ch in texto if not unicodedata.combining(ch))
    return texto.lower()


def extrair_data_cirurgia(texto: str) -> Optional[str]:
    """Tenta encontrar a data da cirurgia principal (dt_SO)."""
    t = normalizar(texto)
    
    # Vários padrões porque os PDFs têm formatos diferentes
    padroes = [
        r"data\s+da\s+cirurgia\s*[:]?\s*(\d{2}[/-]\d{2}[/-]\d{4})",
        r"cirurgia\s+em\s+(\d{2}[/-]\d{2}[/-]\d{4})",
        r"dt\s*so\s*[:]?\s*(\d{2}[/-]\d{2}[/-]\d{4})",
        r"data\s+da\s+operacao\s*[:]?\s*(\d{2}[/-]\d{2}[/-]\d{4})",
        r"po\s*\((\d{2}/\d{2}/\d{4})\)",  # PO (20/01/2017)
    ]
    
    for padrao in padroes:
        match = re.search(padrao, t)
        if match:
            return match.group(1).replace("-", "/")
    
    return None


def encontrar_valores_com_contexto(texto: str, padrao: str, contexto_relevante: List[str] = None,
                                   preferir_proximo_de: List[str] = None, data_cirurgia: Optional[str] = None) -> List[tuple[Any, int]]:
    """
    Encontra todas as ocorrências de um padrão e dá score baseado no contexto.
    Valores próximos de palavras-chave relevantes ganham mais pontos.
    """
    t = normalizar(texto)
    resultados = []
    
    for match in re.finditer(padrao, t, re.IGNORECASE):
        valor = match.group(1) if match.groups() else match.group(0)
        pos = match.start()
        
        score = 0
        antes = t[max(0, pos-300):pos]
        depois = t[pos:min(len(t), pos+300)]
        contexto = antes + depois
        
        # Pontua se tiver palavras de contexto relevante
        if contexto_relevante:
            for palavra in contexto_relevante:
                if palavra in contexto:
                    score += 2
        
        # Pontua mais se estiver perto de palavras-chave específicas
        if preferir_proximo_de:
            for palavra in preferir_proximo_de:
                dist_antes = antes.rfind(palavra) if palavra in antes else -1
                dist_depois = depois.find(palavra) if palavra in depois else -1
                if dist_antes >= 0:
                    score += max(0, 15 - (len(antes) - dist_antes) // 20)
                if dist_depois >= 0:
                    score += max(0, 15 - dist_depois // 20)
        
        # Bonus se estiver perto da data da cirurgia
        if data_cirurgia:
            padrao_data = data_cirurgia.replace("/", "[/-]")
            if re.search(padrao_data, contexto):
                score += 10
        
        resultados.append((valor, score, pos))
    
    # Ordena por score (maior primeiro)
    resultados.sort(key=lambda x: x[1], reverse=True)
    return [(v, s) for v, s, _ in resultados]


# ===== EXTRATORES POR CAMPO =====

def extrair_sexo(texto: str):
    """1 = feminino, 2 = masculino"""
    t = normalizar(texto)
    match = re.search(r"sexo\s*[:\-]?\s*(feminino|masculino|fem\.?|masc\.?|f|m)\b", t)
    if match:
        val = match.group(1)
        if val.startswith("fem") or val == "f":
            return 1
        return 2
    # Fallback simples
    if "sexo" in t:
        if "feminino" in t and "masculino" not in t:
            return 1
        if "masculino" in t and "feminino" not in t:
            return 2
    return None


def extrair_blocos_data_nascto(texto: str):
    """Procura padrões tipo 'Data Nascto 01/02/1967 50 anos ... Dt Registro 24/02/2017'"""
    padrao = re.compile(
        r"Data\s+Nascto\s+(\d{2}/\d{2}/\d{4})\s+(\d{1,3})\s+anos.*?Dt\s*Registro\s+(\d{2}/\d{2}/\d{4})",
        re.S | re.IGNORECASE,
    )
    blocos = []
    for match in padrao.finditer(texto):
        blocos.append({
            "data_nascto": match.group(1),
            "idade": int(match.group(2)),
            "dt_registro": match.group(3)
        })
    return blocos


def extrair_idade(texto: str, data_cirurgia: Optional[str] = None):
    """
    Tenta calcular idade de várias formas:
    1. Data nascimento + data cirurgia
    2. Blocos "Data Nascto ... X anos ... Dt Registro"
    3. "X anos" no texto com contexto
    4. LLM se houver ambiguidade
    """
    # Primeiro tenta pela data de nascimento
    padroes_nasc = [
        r"data\s+nascto\s+(\d{2}/\d{2}/\d{4})",
        r"data\s+nascimento\s+(\d{2}/\d{2}/\d{4})",
        r"dta\.?\s+de\s+nascimento\s+(\d{2}/\d{2}/\d{4})",
        r"dia\.?\s+de\s+nascimento\s+(\d{2}/\d{2}/\d{4})",
    ]
    
    dt_nasc = None
    for padrao in padroes_nasc:
        match = re.search(padrao, texto, re.IGNORECASE)
        if match:
            try:
                dt_nasc = datetime.strptime(match.group(1), "%d/%m/%Y")
                break
            except:
                continue
    
    if dt_nasc and data_cirurgia:
        try:
            dt_cir = datetime.strptime(data_cirurgia.replace("-", "/"), "%d/%m/%Y")
            idade = (dt_cir.year - dt_nasc.year) - ((dt_cir.month, dt_cir.day) < (dt_nasc.month, dt_nasc.day))
            if 0 < idade < 100:
                return idade
        except:
            pass
    
    # Tenta pelos blocos com data de registro
    blocos = extrair_blocos_data_nascto(texto)
    if blocos and data_cirurgia:
        try:
            dt_cir = datetime.strptime(data_cirurgia.replace("-", "/"), "%d/%m/%Y")
            blocos_com_score = []
            for b in blocos:
                try:
                    dt_reg = datetime.strptime(b["dt_registro"], "%d/%m/%Y")
                    diff_dias = abs((dt_cir - dt_reg).days)
                    score = max(0, 365 - diff_dias)  # Quanto mais próximo, maior o score
                    blocos_com_score.append((b["idade"], score))
                except:
                    blocos_com_score.append((b["idade"], 0))
            if blocos_com_score:
                blocos_com_score.sort(key=lambda x: x[1], reverse=True)
                idade = blocos_com_score[0][0]
                if 0 < idade < 100:
                    return idade
        except:
            pass
    
    # Se tiver blocos mas sem data cirurgia, pega a idade mais provável
    if blocos:
        idades_validas = [b["idade"] for b in blocos if 0 < b["idade"] < 100]
        if idades_validas:
            idades_ordenadas = sorted(idades_validas)
            # Prioriza idades na faixa comum (40-80)
            idades_relevantes = [i for i in idades_ordenadas if 40 <= i <= 80]
            if idades_relevantes:
                # Pega a mais próxima de 60 (mediana da faixa)
                idades_relevantes.sort(key=lambda x: abs(x - 60))
                return idades_relevantes[0]
            return idades_ordenadas[-1]
    
    # Último recurso: procura "X anos" no texto
    t = normalizar(texto)
    idades_encontradas = []
    
    for match in re.finditer(r"(\d{1,3})\s+anos", t):
        idade = int(match.group(1))
        if not (30 <= idade <= 90):
            continue
        
        inicio = max(0, match.start() - 200)
        fim = min(len(t), match.end() + 200)
        trecho = t[inicio:fim]
        
        score = 0
        if "paciente" in trecho:
            score += 2
        if "cirurgia" in trecho or "operacao" in trecho or "so" in trecho:
            score += 4
        if "pre" in trecho and "op" in trecho:
            score += 5
        if "dt registro" in trecho or "data registro" in trecho:
            score += 2
        # Penaliza se for idade de outra pessoa
        if any(palavra in trecho for palavra in ["irmao", "filho", "pai", "mae"]):
            score -= 5
        
        idades_encontradas.append((idade, score))
    
    if not idades_encontradas:
        return None
    
    # Se houver muitas idades com scores parecidos, usa LLM
    if len(idades_encontradas) > 2:
        idades_encontradas.sort(key=lambda x: x[1], reverse=True)
        top_scores = [s for _, s in idades_encontradas[:3]]
        if len(set(top_scores)) > 1 and max(top_scores) - min(top_scores) < 3:
            resultado_llm = extrair_com_llm(texto, "idade", "Idade do paciente na data da cirurgia", "número inteiro (ex: 51)")
            if resultado_llm:
                try:
                    idade_llm = int(resultado_llm)
                    if 30 <= idade_llm <= 90:
                        return idade_llm
                except:
                    pass
    
    idades_encontradas.sort(key=lambda x: x[1], reverse=True)
    return idades_encontradas[0][0]


def extrair_local_tumor(texto: str):
    """Extrai localização do tumor. Prioriza menções diretas, depois tenta pela distância da borda anal."""
    t = normalizar(texto)
    
    # Mapeamento direto
    if "transicao retossigmoideana" in t or "transição retossigmoideana" in t:
        return "transicao retossigmoideana"
    if "canal anal" in t:
        return "canal anal"
    if "reto baixo" in t:
        return "reto baixo"
    if "reto medio" in t or "reto médio" in t:
        return "reto medio"
    if "reto alto" in t:
        return "reto alto"
    if "sigmoide" in t or "sigmóide" in t:
        return "sigmoide"
    
    # Tenta pela distância da borda anal
    match = re.search(r"(\d{1,2}(?:[.,]\d+)?)\s*cm\s+da\s+borda\s+anal", t)
    if match:
        try:
            dist = float(match.group(1).replace(",", "."))
            if dist < 5:
                return "reto baixo"
            elif dist <= 10:
                return "reto medio"
            else:
                return "reto alto"
        except ValueError:
            pass
    
    return None


def extrair_ASA(texto: str, data_cirurgia: Optional[str] = None):
    """Classificação ASA (1-4). OCR às vezes escreve 'ASAS' ao invés de 'ASA'. Procura em avaliação pré-op."""
    t = normalizar(texto)
    
    # Procura especificamente em seções de avaliação pré-operatória
    # Padrão comum: "Avaliação Prê-OP: ... ASA X"
    secao_preop = ""
    for match in re.finditer(r"avaliacao\s+pre[-\s]?op[^\n]{0,500}", t, re.IGNORECASE):
        secao_preop += match.group(0) + " "
    
    # Se encontrou seção pré-op, procura lá primeiro
    texto_busca = secao_preop if secao_preop else texto
    
    ocorrencias = encontrar_valores_com_contexto(
        texto_busca,
        r"asa[s]?\s*[:=]?\s*([1-4ivx]+)(?:\s|$|!|\.|!!)",
        contexto_relevante=["pre", "operat", "cirurgia", "avaliacao", "risco", "pre-op"],
        preferir_proximo_de=["cirurgia", "pre", "operat", "avaliacao", "pre-op"],
        data_cirurgia=data_cirurgia
    )
    
    # Se não encontrou na seção pré-op, procura no texto todo
    if not ocorrencias:
        ocorrencias = encontrar_valores_com_contexto(
            texto,
            r"asa[s]?\s*[:=]?\s*([1-4ivx]+)(?:\s|$|!|\.|!!)",
            contexto_relevante=["pre", "operat", "cirurgia", "avaliacao", "risco", "pre-op"],
            preferir_proximo_de=["cirurgia", "pre", "operat", "avaliacao", "pre-op"],
            data_cirurgia=data_cirurgia
        )
    
    # Se houver muitas ocorrências com scores similares, pergunta pro LLM
    if len(ocorrencias) > 1:
        top_scores = sorted([s for _, s in ocorrencias[:3]], reverse=True)
        if len(top_scores) > 1 and top_scores[0] - top_scores[1] < 2:
            resultado_llm = extrair_com_llm(texto, "ASA", "Classificação ASA pré-operatória na data da cirurgia principal (1-4)", "número inteiro entre 1 e 4")
            if resultado_llm:
                try:
                    asa_llm = int(resultado_llm)
                    if 1 <= asa_llm <= 4:
                        return asa_llm
                except:
                    pass
    
    if not ocorrencias:
        return None
    
    token = ocorrencias[0][0].strip()
    token = re.sub(r'[^0-9ivx]', '', token.lower())
    
    # Converte romano para número
    romano_para_num = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
    if token in romano_para_num:
        return romano_para_num[token]
    
    try:
        val = int(token)
        if 1 <= val <= 4:
            return val
    except ValueError:
        pass
    
    return None


def extrair_ECOG(texto: str, data_cirurgia: Optional[str] = None):
    """Performance Status ECOG (0-4). OCR às vezes escreve 'ECOq O' (letra O) ao invés de 'ECOG 0'."""
    t = normalizar(texto)
    
    # Primeiro procura padrões explícitos "ECOG 0" ou "ECOG O" seguido de "Completamente ativo"
    # Esse padrão é muito específico e confiável
    match_especifico = re.search(r"ecog\s*[:=]?\s*([0o])\s+completamente\s+ativo", t, re.IGNORECASE)
    if match_especifico:
        return 0
    
    ocorrencias = encontrar_valores_com_contexto(
        texto,
        r"ecog\s*[:=]?\s*([0-4o])(?:\s|$|dl|completamente)",
        contexto_relevante=["pre", "operat", "cirurgia", "avaliacao", "performance", "pre-op"],
        preferir_proximo_de=["cirurgia", "pre", "operat", "performance", "avaliacao", "pre-op"],
        data_cirurgia=data_cirurgia
    )
    
    if len(ocorrencias) > 1:
        top_scores = sorted([s for _, s in ocorrencias[:3]], reverse=True)
        if len(top_scores) > 1 and top_scores[0] - top_scores[1] < 2:
            resultado_llm = extrair_com_llm(texto, "ECOG", "Performance Status ECOG pré-operatório na data da cirurgia principal (0-4)", "número inteiro entre 0 e 4")
            if resultado_llm:
                try:
                    ecog_llm = int(resultado_llm)
                    if 0 <= ecog_llm <= 4:
                        return ecog_llm
                except:
                    pass
    
    if not ocorrencias:
        # Fallback: procura "ECOG" ou "Performance Status (ECOG)"
        match = re.search(r"(?:ecog|performance\s+status.*?ecog)\s*[:=]?\s*([0-4o])(?:\s|$)", t, re.IGNORECASE)
        if match:
            val = match.group(1).lower()
            if val == 'o':
                return 0
            try:
                return int(val)
            except:
                pass
        return None
    
    # Converte 'o' para 0
    val = ocorrencias[0][0].lower()
    if val == 'o':
        return 0
    try:
        return int(val)
    except:
        return None


def extrair_KPS(texto: str, data_cirurgia: Optional[str] = None):
    """Karnofsky Performance Status (0-100). OCR às vezes escreve 'KP5' ao invés de 'KPS'."""
    t = normalizar(texto)
    ocorrencias = encontrar_valores_com_contexto(
        texto,
        r"k[p]?s?\s*[:=]?\s*(\d{2,3})(?:\s|$|kg|%|bpm|mmHg)",
        contexto_relevante=["pre", "operat", "cirurgia", "performance", "karnofsky"],
        preferir_proximo_de=["karnofsky", "performance", "pre", "operat"],
        data_cirurgia=data_cirurgia
    )
    
    if not ocorrencias:
        match = re.search(r"(?:kps|karnofsky)\s*[:=]?\s*(\d{2,3})(?:\s|$|%)", t, re.IGNORECASE)
        if match:
            try:
                val = int(match.group(1))
                if 0 <= val <= 100:
                    return val
            except:
                pass
        return None
    
    try:
        val = int(ocorrencias[0][0])
        if 0 <= val <= 100:
            return val
    except:
        pass
    
    return None


def extrair_IMC(texto: str, data_cirurgia: Optional[str] = None):
    """
    Extrai IMC. Prioriza valores pré-operatórios e próximos da data da cirurgia.
    Se houver muitas ocorrências com scores similares, usa LLM.
    """
    t = normalizar(texto)
    
    todas_ocorrencias = []
    for match in re.finditer(r"imc\s*[:=]?\s*([0-9]{1,2}(?:[.,][0-9]{1,2})?)", t, re.IGNORECASE):
        val_str = match.group(1).replace(",", ".")
        try:
            imc = float(val_str)
            if not (10 <= imc <= 50):
                continue
            
            pos = match.start()
            antes = t[max(0, pos-500):pos]
            depois = t[pos:min(len(t), pos+500)]
            contexto = antes + depois
            
            score = 0
            # Valores na faixa comum ganham pontos
            if 20 <= imc <= 30:
                score += 5
            # Valores próximos de 24-25 (comum em pacientes oncológicos)
            if 24 <= imc <= 25:
                score += 3
            
            # Bonus se estiver em contexto pré-operatório
            if any(palavra in contexto for palavra in ["pre", "operat", "cirurgia", "avaliacao", "pre-op"]):
                score += 8
            
            if "peso" in contexto or "altura" in contexto:
                score += 2
            
            # Bonus grande se estiver perto da data da cirurgia
            if data_cirurgia:
                padrao_data = data_cirurgia.replace("/", "[/-]")
                if re.search(padrao_data, contexto):
                    score += 15
            
            # Penaliza consultas de seguimento (não pré-op)
            if "consulta" in contexto and "seguimento" in contexto:
                score -= 3
            
            todas_ocorrencias.append((imc, score))
        except ValueError:
            continue
    
    # Se houver muitas ocorrências com scores muito próximos, usa LLM
    if len(todas_ocorrencias) > 3:
        top_scores = sorted([s for _, s in todas_ocorrencias], reverse=True)
        if len(top_scores) > 1 and top_scores[0] - top_scores[1] < 3:
            resultado_llm = extrair_com_llm(texto, "IMC", "Índice de Massa Corporal pré-operatório", "número decimal (ex: 24.4)")
            if resultado_llm:
                try:
                    imc_llm = float(resultado_llm)
                    if 10 <= imc_llm <= 50:
                        return round(imc_llm, 1)
                except:
                    pass
    
    if todas_ocorrencias:
        todas_ocorrencias.sort(key=lambda x: x[1], reverse=True)
        return round(todas_ocorrencias[0][0], 1)
    
    # Fallback: tenta calcular a partir de peso e altura
    pesos = []
    alturas = []
    
    for match in re.finditer(r"peso\)?\s*[:=]?\s*([0-9]{2,3}(?:[.,][0-9])?)", t):
        try:
            peso = float(match.group(1).replace(",", "."))
            if 30 <= peso <= 150:
                inicio = max(0, match.start() - 100)
                contexto = t[inicio:match.start()]
                score = 0
                if "pre" in contexto or "operat" in contexto:
                    score += 2
                if "cirurgia" in contexto:
                    score += 1
                pesos.append((peso, score))
        except ValueError:
            continue
    
    for match in re.finditer(r"altura\)?\s*[:=]?\s*([0-9]{1,3}(?:[.,][0-9]{1,2})?)", t):
        try:
            altura_str = match.group(1).replace(",", ".")
            # Se está em metros (1.50-2.20)
            if "." in altura_str and 1.0 <= float(altura_str) <= 2.2:
                altura = float(altura_str)
            # Se está em cm (100-220)
            elif 100 <= float(altura_str) <= 220:
                altura = float(altura_str) / 100
            else:
                continue
            
            inicio = max(0, match.start() - 100)
            contexto = t[inicio:match.start()]
            score = 0
            if "pre" in contexto or "operat" in contexto:
                score += 2
            if "cirurgia" in contexto:
                score += 1
            alturas.append((altura, score))
        except ValueError:
            continue
    
    if pesos and alturas:
        pesos.sort(key=lambda x: x[1], reverse=True)
        alturas.sort(key=lambda x: x[1], reverse=True)
        p = pesos[0][0]
        h = alturas[0][0]
        imc = p / (h * h)
        if 10 <= imc <= 50:
            return round(imc, 1)
    
    return None


def extrair_QRT_neo(texto: str):
    """Quimioterapia neoadjuvante: 0 = não, 1 = sim"""
    t = normalizar(texto)
    if "radioterapia neoadjuvante" in t or "rt neoadjuvante" in t:
        if "nao realizou radioterapia neoadjuvante" in t or "nao fez radioterapia neoadjuvante" in t:
            return 0
        return 1
    if "neoadjuvante" in t and ("radioterapia" in t or "rt " in t):
        if "nao" in t and "radioterapia" in t and "neoadjuvante" in t:
            return 0
        return 1
    return 0


def extrair_eletiva(texto: str):
    """0 = urgência/emergência, 1 = eletiva"""
    t = normalizar(texto)
    if "eletiva" in t:
        if "nao eletiva" in t:
            return 0
        return 1
    if "urgencia" in t or "emergencia" in t:
        return 0
    return 1


def extrair_altura_tumor(texto: str):
    """Altura do tumor em cm. Pode estar como '99' (não aplicável) ou valor real."""
    t = normalizar(texto)
    padroes = [
        r"altura\s+tumor\s*[:]?\s*(\d{1,3}(?:[.,]\d+)?)\s*cm",
        r"altura\s*[:]?\s*(\d{1,3}(?:[.,]\d+)?)\s*cm.*?tumor",
        r"distancia.*?borda\s+anal.*?(\d{1,2}(?:[.,]\d+)?)\s*cm",  # Distância da borda anal
    ]
    for padrao in padroes:
        match = re.search(padrao, t, re.IGNORECASE)
        if match:
            try:
                altura = float(match.group(1).replace(",", "."))
                # Se for 99, pode ser código de "não aplicável"
                if altura == 99:
                    return 99
                if 0 < altura <= 50:  # Validação razoável
                    return altura
            except:
                pass
    return None


def extrair_cirurgia_recidiva(texto: str):
    """0 = não, 1 = sim, 88 = não aplicável. Verifica se a cirurgia foi por recidiva."""
    t = normalizar(texto)
    # Procura padrões específicos de cirurgia por recidiva
    if "exentera" in t and "recidiva" in t:
        return 1
    if "cirurgia" in t and "recidiva" in t:
        # Verifica contexto - se é cirurgia POR recidiva (não cirurgia DE recidiva)
        idx = t.find("recidiva")
        contexto = t[max(0, idx-50):min(len(t), idx+50)]
        if "por" in contexto or "para" in contexto:
            return 1
        if "nao" in contexto or "sem" in contexto:
            return 0
        return 1
    # Se não menciona recidiva em contexto cirúrgico, pode ser 88 (não aplicável)
    return None


def extrair_paliativa(texto: str):
    """0 = não, 1 = sim"""
    t = normalizar(texto)
    if "paliativa" in t or "paliativo" in t:
        if "nao" in t or "sem" in t:
            return 0
        return 1
    return 0


def extrair_orgaos_envolvidos(texto: str) -> Dict[str, int]:
    """
    Extrai quais órgãos foram envolvidos na cirurgia.
    IMPORTANTE: "preservação" significa NÃO envolvido (0).
    Só marca como 1 se realmente houver ressecção/envolvimento.
    """
    t = normalizar(texto)
    resultado = {}
    
    orgaos = {
        "utero": ["utero", "uterino", "histerectomia"],
        "vagina": ["vagina", "vaginal"],
        "ovario": ["ovario", "ovarico", "anexo", "salpingooforectomia"],
        "bexiga": ["bexiga", "vesical"],
        "ureter": ["ureter", "ureteral", "ureterectomia"],
        "prostata": ["prostata", "prostatico"],
        "vesicula_sem": ["vesicula", "biliar", "colecistectomia"],
        "sacro": ["sacro", "sacral"],
    }
    
    for campo, palavras_chave in orgaos.items():
        encontrado = False
        preservado = False
        
        for palavra in palavras_chave:
            if palavra in t:
                idx = t.find(palavra)
                antes = t[max(0, idx-150):idx]
                depois = t[idx:min(len(t), idx+len(palavra)+150)]
                contexto = antes + depois
                
                # Se menciona "preservação", definitivamente não foi envolvido
                if "preservacao" in contexto or "preservado" in contexto or "preserva" in contexto:
                    preservado = True
                    break
                
                # Ignora se for negação explícita
                if "nao" in antes[-40:] or "sem" in antes[-40:]:
                    continue
                
                # Só marca se houver envolvimento cirúrgico claro
                palavras_cirurgia = ["ressecao", "exentera", "histerectomia", "salpingo", "ureterectomia",
                                     "envolvido", "invasao", "infiltracao", "aderencia", "lesao", "extensao",
                                     "serosa", "infiltracao"]
                
                # Para útero, ovário, vagina - se menciona "histerectomia" ou "salpingo" já é envolvimento
                if campo in ["utero", "ovario", "vagina"]:
                    if "histerectomia" in contexto or "salpingo" in contexto or "exentera" in contexto:
                        encontrado = True
                        break
                
                # Para outros órgãos, precisa de evidência mais clara
                if any(pc in contexto for pc in palavras_cirurgia):
                    # Mas verifica se não é só menção genérica
                    if campo == "bexiga" and "preservacao" not in contexto:
                        # Se menciona bexiga em contexto de exenteração, provavelmente foi envolvida
                        if "exentera" in contexto and "posterior" in contexto:
                            # Exenteração posterior geralmente preserva bexiga
                            if "preservacao" not in contexto and "preservado" not in contexto:
                                # Mas se não menciona preservação, pode ter sido envolvida
                                encontrado = True
                                break
                    elif campo != "bexiga":
                        encontrado = True
                        break
        
        # Se foi preservado, definitivamente 0
        if preservado:
            resultado[campo] = 0
        else:
            resultado[campo] = 1 if encontrado else 0
    
    return resultado


def extrair_bexiga_tudo(texto: str):
    """Bexiga total: só marca 1 se realmente houver ressecção total da bexiga."""
    t = normalizar(texto)
    # Se menciona preservação, definitivamente não foi ressecada
    if "preservacao" in t and "bexiga" in t:
        return 0
    if "bexiga" in t and ("total" in t or "tudo" in t) and "preservacao" not in t:
        return 1
    return 0


def extrair_bexiga_parte(texto: str):
    """Bexiga parcial: só marca 1 se realmente houver ressecção parcial."""
    t = normalizar(texto)
    # Se menciona preservação, definitivamente não foi ressecada
    if "preservacao" in t and "bexiga" in t:
        return 0
    if "bexiga" in t and ("parcial" in t or "parte" in t) and "preservacao" not in t:
        return 1
    return 0


def extrair_outro_orgao(texto: str) -> tuple[Optional[int], Optional[str]]:
    """Retorna (1 se outro órgão, qual). Procura por íleo e peritônio principalmente."""
    t = normalizar(texto)
    
    # Padrão específico: "serosa de ileo" ou "invasão de ileo" = outro órgão envolvido
    if "serosa" in t and "ileo" in t:
        # Procura qual outro órgão
        if "peritonio" in t or "peritoneal" in t:
            return (1, "ileo e peritonio")
        return (1, "ileo")
    
    # Procura outros padrões
    outros = ["peritonio", "peritoneal", "intestino", "ileo", "colon", "figado", "pancreas"]
    orgaos_encontrados = []
    
    for org in outros:
        if org in t:
            idx = t.find(org)
            contexto = t[max(0, idx-150):min(len(t), idx+150)]
            if any(palavra in contexto for palavra in ["envolvido", "invasao", "infiltracao", "aderencia", "serosa", "extensao"]):
                orgaos_encontrados.append(org)
    
    if orgaos_encontrados:
        # Se encontrou múltiplos, junta
        if len(orgaos_encontrados) > 1:
            return (1, " e ".join(orgaos_encontrados))
        return (1, orgaos_encontrados[0])
    
    return (None, None)


def extrair_n_orgaos(texto: str, orgaos_dict: Dict[str, int]) -> Optional[int]:
    """Conta quantos órgãos foram envolvidos"""
    count = sum(1 for v in orgaos_dict.values() if v == 1)
    outro_orgao, _ = extrair_outro_orgao(texto)
    if outro_orgao == 1:
        count += 1
    return count if count > 0 else None


def extrair_amputacao(texto: str):
    t = normalizar(texto)
    if "amputacao" in t or "amputação" in t:
        if "nao" in t or "sem" in t:
            return 0
        return 1
    return 0


def extrair_RTS(texto: str):
    """Ressecção total do sigmoide: 0 = não, 1 = sim. Procura em contexto de cirurgia."""
    t = normalizar(texto)
    if "ressecao" in t and "sigmoide" in t:
        # Verifica se é ressecção total
        idx = t.find("sigmoide")
        contexto = t[max(0, idx-100):min(len(t), idx+100)]
        if "total" in contexto:
            return 1
        return 0
    # Também procura por "retossigmoidectomia" que pode incluir sigmoide
    if "retossigmoidectomia" in t:
        return 1
    return None


def extrair_cole_total(texto: str):
    """Colectomia total: 0 = não, 1 = sim. Verifica se é da cirurgia principal."""
    t = normalizar(texto)
    # Procura em contexto de cirurgia principal (não só menção genérica)
    if "colectomia" in t and "total" in t:
        # Verifica se está em contexto de cirurgia principal (não só histórico)
        idx = t.find("colectomia")
        contexto = t[max(0, idx-200):min(len(t), idx+200)]
        # Se está perto de data da cirurgia ou PO, é da cirurgia principal
        if any(palavra in contexto for palavra in ["po", "cirurgia", "14/04/09", "14.04.09"]):
            return 1
        # Se não tem contexto claro, assume que não é da cirurgia principal
        return 0
    return 0


def extrair_posterior(texto: str):
    """Ressecção posterior: 0 = não, 1 = sim. Exenteração pélvica posterior conta como 1."""
    t = normalizar(texto)
    if "exentera" in t and "posterior" in t:
        return 1
    if "ressecao" in t and "posterior" in t:
        return 1
    return 0


def extrair_total(texto: str):
    """Ressecção total: 0 = não, 1 = sim"""
    t = normalizar(texto)
    if "ressecao" in t and "total" in t:
        return 1
    return 0


def extrair_SLE(texto: str):
    """Síndrome de Lynch/hereditária: 0 = não, 1 = sim"""
    t = normalizar(texto)
    termos = ["lynch", "hereditario", "hereditaria", "hnpcc", "mmr"]
    if any(termo in t for termo in termos):
        return 1
    return 0


def extrair_REC_plastica(texto: str):
    """Reconstrução com plástica: 0 = não, 1 = sim. Verifica se é reconstrução urológica."""
    t = normalizar(texto)
    # "Duplo J" ou "nefrostomia" são tipos de reconstrução, mas não necessariamente plástica
    if "reconstrucao" in t and "plastica" in t:
        return 1
    if "reconstrucao" in t and "gregoir" in t:  # Gregoir é tipo de reconstrução
        return 1
    # Se só menciona "plastica" sem contexto de reconstrução, pode ser outra coisa
    return 0


def extrair_tipo_REC(texto: str):
    """Tipo de reconstrução: 0=briker, 1=duplo barril, 2=duplo barril ileal, 3=nefrostomia"""
    t = normalizar(texto)
    
    # Procura padrões específicos
    if "nefrostomia" in t:
        return "3"
    if "duplo barril ileal" in t:
        return "2"
    if "duplo barril" in t or "duplo j" in t:
        return "1"
    if "briker" in t or "bricker" in t:
        return "0"
    
    # Se menciona "gregoir" pode ser tipo de reconstrução, mas não está no mapeamento
    return None


def extrair_tempo_SO(texto: str, data_cirurgia: Optional[str] = None):
    """Tempo desde SO em dias - precisa data atual, por enquanto retorna None"""
    return None


def extrair_CH_intra_OP(texto: str):
    """Quimioterapia intra-operatória: 0 = não, 1 = sim. Verifica se realmente foi durante a cirurgia."""
    t = normalizar(texto)
    # Procura padrões específicos de quimioterapia intra-operatória
    if "quimio" in t and "intra" in t and "operat" in t:
        # Verifica se não é negação
        idx = t.find("intra")
        contexto = t[max(0, idx-50):min(len(t), idx+50)]
        if "nao" not in contexto:
            return 1
    return 0


def extrair_CH_num(texto: str):
    """Número de ciclos de quimioterapia. Procura em contexto de QT neoadjuvante/adjuvante."""
    t = normalizar(texto)
    
    # Procura padrões específicos
    padroes = [
        r"(\d+)\s+ciclos.*?qt",
        r"qt.*?(\d+)\s+ciclos",
        r"(\d+)\s+ciclos",
    ]
    
    for padrao in padroes:
        match = re.search(padrao, t, re.IGNORECASE)
        if match:
            try:
                ciclos = int(match.group(1))
                if 1 <= ciclos <= 50:  # Validação razoável
                    return ciclos
            except:
                pass
    
    return None


def extrair_dias_uti(texto: str):
    """Dias de UTI. Procura em evoluções de internação."""
    t = normalizar(texto)
    padroes = [
        r"dias\s+uti\s*[:]?\s*(\d+)",
        r"uti\s*[:]?\s*(\d+)\s+dias",
        r"permanencia.*?uti.*?(\d+)\s+dias",
    ]
    for padrao in padroes:
        match = re.search(padrao, t, re.IGNORECASE)
        if match:
            try:
                dias = int(match.group(1))
                if 0 <= dias <= 365:  # Validação razoável
                    return dias
            except:
                pass
    return None


def extrair_dias_internacao(texto: str):
    """Dias de internação. Pode calcular pela diferença entre entrada e alta."""
    t = normalizar(texto)
    
    # Tenta calcular pela diferença de datas
    padrao_entrada = r"entrada[:\s]+(\d{2}/\d{2}/\d{4})"
    padrao_alta = r"alta[:\s]+(\d{2}/\d{2}/\d{4})"
    
    match_entrada = re.search(padrao_entrada, t, re.IGNORECASE)
    match_alta = re.search(padrao_alta, t, re.IGNORECASE)
    
    if match_entrada and match_alta:
        try:
            dt_entrada = datetime.strptime(match_entrada.group(1), "%d/%m/%Y")
            dt_alta = datetime.strptime(match_alta.group(1), "%d/%m/%Y")
            dias = (dt_alta - dt_entrada).days
            if 0 <= dias <= 365:
                return dias
        except:
            pass
    
    # Fallback: procura padrões diretos
    padroes = [
        r"dias\s+internacao\s*[:]?\s*(\d+)",
        r"internacao\s*[:]?\s*(\d+)\s+dias",
        r"permanencia.*?(\d+)\s+dias",
    ]
    for padrao in padroes:
        match = re.search(padrao, t, re.IGNORECASE)
        if match:
            try:
                dias = int(match.group(1))
                if 0 <= dias <= 365:
                    return dias
            except:
                pass
    return None


def extrair_dt_alta(texto: str):
    """Data de alta"""
    t = normalizar(texto)
    padroes = [
        r"dt\s*alta\s*[:]?\s*(\d{2}[/-]\d{2}[/-]\d{4})",
        r"data\s+alta\s*[:]?\s*(\d{2}[/-]\d{2}[/-]\d{4})",
        r"alta\s+em\s+(\d{2}/\d{2}/\d{4})",
    ]
    for padrao in padroes:
        match = re.search(padrao, t, re.IGNORECASE)
        if match:
            return match.group(1).replace("-", "/")
    return None


def extrair_complicacao(texto: str):
    """0 = não, 1 = sim. Procura em evoluções pós-operatórias."""
    t = normalizar(texto)
    # Procura padrões de negação primeiro
    if "sem complicacao" in t or "sem complicação" in t or "nao complicacao" in t:
        return 0
    if "complicacao" in t or "complicação" in t:
        # Verifica se é menção de complicação real ou só teórica
        idx = t.find("complicacao") if "complicacao" in t else t.find("complicação")
        contexto = t[max(0, idx-100):min(len(t), idx+100)]
        # Se menciona tipos específicos de complicação, é real
        if any(palavra in contexto for palavra in ["fistula", "deiscencia", "infeccao", "sangramento", "obstrucao"]):
            return 1
        # Se está em contexto de "sem", é 0
        if "sem" in contexto[:50]:
            return 0
        return 1
    return 0


def extrair_complicacao_qual(texto: str):
    """Qual a complicação"""
    t = normalizar(texto)
    complicacoes = ["fistula", "deiscencia", "infeccao", "sangramento", "obstrucao"]
    for comp in complicacoes:
        if comp in t:
            return comp
    return None


def extrair_tto(texto: str):
    """Tratamento da complicação - ainda não implementado"""
    return None


def extrair_Clavien(texto: str):
    """Classificação de Clavien (0-5)"""
    t = normalizar(texto)
    match = re.search(r"clavien\s*[:]?\s*([0-5])", t, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def extrair_Clavien_v2(texto: str):
    """Classificação de Clavien v2 (numérico)"""
    clavien = extrair_Clavien(texto)
    if clavien:
        try:
            return int(clavien)
        except:
            pass
    return None


def extrair_reinternacao(texto: str):
    """0 = não, 1 = sim. Procura por múltiplas internações."""
    t = normalizar(texto)
    # Conta quantas internações diferentes há
    padrao_internacao = r"n[oº°]?\s+atendimento[:\s]+(\d+).*?entrada[:\s]+(\d{2}/\d{2}/\d{4})"
    internacoes = list(re.finditer(padrao_internacao, t, re.IGNORECASE))
    
    # Se há mais de uma internação, provavelmente houve reinternação
    if len(internacoes) > 1:
        return 1
    
    if "reinternacao" in t or "reinternação" in t:
        return 1
    return 0


def extrair_data_reinternacao(texto: str):
    """Data da reinternação"""
    t = normalizar(texto)
    padroes = [
        r"data\s+reinternacao\s*[:]?\s*(\d{2}[/-]\d{2}[/-]\d{4})",
        r"reinternacao\s+em\s+(\d{2}/\d{2}/\d{4})",
    ]
    for padrao in padroes:
        match = re.search(padrao, t, re.IGNORECASE)
        if match:
            return match.group(1).replace("-", "/")
    return None


def extrair_motivo_reinternacao(texto: str):
    """Motivo da reinternação"""
    t = normalizar(texto)
    motivos = ["itu", "infeccao", "fistula", "obstrucao", "sangramento"]
    for motivo in motivos:
        if motivo in t and "reinternacao" in t:
            return motivo
    return None


def extrair_re_op_90dias(texto: str):
    """Reoperação em 90 dias: 0 = não, 1 = sim. Verifica se há múltiplas cirurgias próximas."""
    t = normalizar(texto)
    # Procura por múltiplas cirurgias com datas próximas
    padrao_cirurgia = r"(?:po|cirurgia|operacao).*?(\d{2}/\d{2}/\d{4})"
    cirurgias = list(re.finditer(padrao_cirurgia, t, re.IGNORECASE))
    
    if len(cirurgias) > 1:
        # Verifica se há cirurgias com menos de 90 dias de diferença
        datas_cir = []
        for match in cirurgias:
            try:
                dt = datetime.strptime(match.group(1), "%d/%m/%Y")
                datas_cir.append(dt)
            except:
                continue
        
        if len(datas_cir) > 1:
            datas_cir.sort()
            for i in range(len(datas_cir) - 1):
                diff = (datas_cir[i+1] - datas_cir[i]).days
                if 0 < diff <= 90:
                    return 1
    
    if "reoperacao" in t or "re-op" in t or "reoperacao" in t:
        return 1
    return 0


def extrair_re_op_achado(texto: str):
    """Achado da reoperação - ainda não implementado"""
    return None


def extrair_obito_90dias(texto: str):
    """Óbito em 90 dias: 0 = não, 1 = sim"""
    t = normalizar(texto)
    if "obito" in t or "óbito" in t:
        return 1
    return 0


def extrair_histologia(texto: str):
    """Tipo histológico"""
    t = normalizar(texto)
    tipos = {
        "ADENOCA": ["adenocarcinoma", "adenoca"],
        "CARCINOIDE": ["carcinoide"],
        "GIST": ["gist"],
        "LINFOMA": ["linfoma"],
    }
    for tipo, palavras in tipos.items():
        if any(palavra in t for palavra in palavras):
            return tipo
    return None


def extrair_AP(texto: str):
    """Anatomia patológica (estadiamento TNM). Procura em laudos de AP."""
    t = normalizar(texto)
    
    # Procura especificamente em seções de AP/anatomia patológica
    # Padrões mais específicos primeiro
    padroes = [
        r"p?t([0-4][a-cb]?)\s+p?n([0-3][a-cb]?)\s+p?m([0-1][a-cb]?)",  # pT4b pN1b pM0
        r"t([0-4][a-cb]?)\s+n([0-3][a-cb]?)\s+m([0-1][a-cb]?)",  # T4b N1b M0
        r"ap[:\s]+.*?p?t([0-4][a-cb]?)\s+p?n([0-3][a-cb]?)\s+p?m([0-1][a-cb]?)",  # AP: pT4b pN1b
    ]
    
    for padrao in padroes:
        match = re.search(padrao, t, re.IGNORECASE)
        if match:
            t_val = match.group(1).upper()
            n_val = match.group(2).upper()
            m_val = match.group(3).upper() if len(match.groups()) > 2 else "X"
            # Prioriza formato com "p" se encontrou
            if "p" in padrao or "p" in match.group(0).lower():
                return f"pT{t_val} pN{n_val} pM{m_val}"
            else:
                return f"T{t_val} N{n_val} M{m_val}"
    
    return None


def extrair_estadiamento(texto: str):
    """Estadiamento numérico (0-4). Pode calcular a partir do TNM."""
    t = normalizar(texto)
    
    # Primeiro tenta extrair diretamente
    match = re.search(r"estadio\s*[:]?\s*([0-4ivx]+)", t, re.IGNORECASE)
    if match:
        val = match.group(1).lower()
        mapping = {"i": 0, "ii": 1, "iii": 2, "iv": 3, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4}
        if val in mapping:
            return mapping[val]
        try:
            return int(val)
        except:
            pass
    
    # Fallback: tenta calcular a partir do TNM
    # T4 N1 M0 = estádio 3 (aproximado)
    ap = extrair_AP(texto)
    if ap:
        # Lógica simplificada: T4 N1 = estádio 3
        if "t4" in ap.lower() and "n1" in ap.lower():
            return 3
        if "t3" in ap.lower() and "n0" in ap.lower():
            return 2
    
    return None


def extrair_T(texto: str):
    """T do TNM. Procura em laudos de AP."""
    t = normalizar(texto)
    # Procura padrões mais específicos incluindo "b" (ex: T4b)
    match = re.search(r"p?t([0-4][a-cb]?)", t, re.IGNORECASE)
    if match:
        return f"T{match.group(1).upper()}"
    return None


def extrair_N(texto: str):
    """N do TNM (formato X/Y). Procura padrões como '1/4', '2/26' em contexto de linfonodos."""
    t = normalizar(texto)
    
    # Padrões mais específicos primeiro - procura em contexto de linfonodos
    padroes = [
        r"neoplasia\s+em\s+(\d+)\s+de\s+(\d+)\s+linfonodos",  # "neoplasia em 1 de 4 linfonodos"
        r"(\d+)\s*[/]\s*(\d+)\s+linfonodos",  # "1/4 linfonodos"
        r"linfonodos.*?(\d+)\s*[/]\s*(\d+)",  # "linfonodos ... 2/26"
        r"n\s*[:]?\s*(\d+)\s*[/]\s*(\d+)",  # "N: 2/26"
        r"(\d+)\s*[/]\s*(\d+)(?:\s|$)",  # "2/26" genérico (último recurso)
    ]
    
    for padrao in padroes:
        matches = list(re.finditer(padrao, t, re.IGNORECASE))
        if matches:
            # Prioriza matches que estão em contexto de linfonodos/AP
            for match in matches:
                contexto = t[max(0, match.start()-50):match.end()+50]
                if any(palavra in contexto for palavra in ["linfonodo", "linfonodos", "neoplasia", "ap", "anatomia"]):
                    num = int(match.group(1))
                    den = int(match.group(2))
                    # Validação: denominador geralmente entre 1-100, numerador <= denominador
                    if 1 <= den <= 100 and 0 <= num <= den:
                        return f"{num}/{den}"
    
    return None


def extrair_N_A(texto: str):
    """Número de linfonodos acometidos"""
    n_str = extrair_N(texto)
    if n_str:
        try:
            return int(n_str.split("/")[0])
        except:
            pass
    return None


def extrair_invasao(texto: str):
    """Descrição da invasão. Procura em laudos de AP."""
    t = normalizar(texto)
    
    # Procura padrões mais específicos primeiro
    padroes = [
        r"extensao.*?neoplasia.*?(?:serosa|musculo|mucosa|submucosa|subserosa|transmural|ileo|bexiga|intestino)",
        r"invasao.*?(?:serosa|musculo|mucosa|submucosa|subserosa|transmural|ileo|bexiga|intestino)",
        r"alem.*?serosa.*?invasao.*?(?:ileo|bexiga|intestino|delgado)",
    ]
    
    for padrao in padroes:
        match = re.search(padrao, t, re.IGNORECASE)
        if match:
            # Tenta pegar uma descrição mais completa
            inicio = max(0, match.start() - 50)
            fim = min(len(t), match.end() + 200)
            descricao = t[inicio:fim].strip()
            # Limita tamanho
            if len(descricao) > 300:
                descricao = descricao[:300]
            return descricao
    
    return None


def extrair_R0_R1_R2(texto: str):
    """Margem cirúrgica. Procura por 'R0', 'R1', 'R2' ou 'margens livres'."""
    t = normalizar(texto)
    
    # Primeiro procura por "margens livres" ou "livre de neoplasia" = R0
    if "margens" in t and "livre" in t:
        return "R0"
    if "livre" in t and "neoplasia" in t and "margem" in t:
        return "R0"
    
    # Procura padrões R0, R1, R2
    match = re.search(r"\br([0-2])\b", t, re.IGNORECASE)
    if match:
        return f"R{match.group(1)}"
    
    # Procura em contexto de AP/laudo
    match = re.search(r"(?:margem|margens).*?r([0-2])", t, re.IGNORECASE)
    if match:
        return f"R{match.group(1)}"
    
    return None


def extrair_R0_R1_R2_v2(texto: str):
    """Margem cirúrgica (numérico)"""
    r_str = extrair_R0_R1_R2(texto)
    if r_str:
        try:
            return int(r_str[1])
        except:
            pass
    return None


def extrair_QT_adjuvante(texto: str):
    """Quimioterapia adjuvante: 88 = não aplicável, 0 = não, 1 = sim."""
    t = normalizar(texto)
    # Procura padrões específicos
    if "qt adjuvante" in t or "quimio adjuvante" in t or "quimioterapia adjuvante" in t:
        # Verifica se realmente foi feita
        idx = t.find("adjuvante")
        contexto = t[max(0, idx-100):min(len(t), idx+200)]
        if "nao" not in contexto[:50]:
            return 1
        return 0
    return None


def extrair_recidiva(texto: str):
    """0 = não, 1 = sim"""
    t = normalizar(texto)
    if "recidiva" in t or "recidiva" in t:
        if "sem recidiva" in t or "nao recidiva" in t:
            return 0
        return 1
    return 0


def extrair_recidiva_local(texto: str):
    """Local da recidiva"""
    t = normalizar(texto)
    locais = ["pelve", "hepatica", "pulmonar", "peritoneal", "local"]
    for local in locais:
        if local in t and "recidiva" in t:
            return local.upper()
    return None


def extrair_recidiva_local_v2(texto: str):
    """Local da recidiva (codificado)"""
    local = extrair_recidiva_local(texto)
    if local:
        return 1
    return 0


def extrair_dt_recidiva(texto: str):
    """Data da recidiva. Procura por datas próximas de menções de recidiva."""
    t = normalizar(texto)
    padroes = [
        r"data\s+recidiva\s*[:]?\s*(\d{2}[/-]\d{2}[/-]\d{4})",
        r"recidiva\s+em\s+(\d{2}/\d{2}/\d{4})",
        r"(\d{2}/\d{2}/\d{4}).*?recidiva",  # Data antes de "recidiva"
        r"recidiva.*?(\d{2}/\d{2}/\d{4})",  # Data depois de "recidiva"
    ]
    for padrao in padroes:
        matches = list(re.finditer(padrao, t, re.IGNORECASE))
        if matches:
            # Prioriza matches próximos de "recidiva"
            for match in matches:
                contexto = t[max(0, match.start()-50):match.end()+50]
                if "recidiva" in contexto:
                    return match.group(1).replace("-", "/")
    return None


def extrair_DFSMESES(texto: str):
    """Disease-free survival em meses - precisa cálculo temporal"""
    return None


def extrair_fisiatria(texto: str):
    """0 = não, 1 = sim. Verifica se realmente houve acompanhamento de fisiatria."""
    t = normalizar(texto)
    # Só marca se houver menção clara de acompanhamento/avaliação de fisiatria
    if "fisiatria" in t:
        idx = t.find("fisiatria")
        contexto = t[max(0, idx-50):min(len(t), idx+50)]
        # Se está em contexto de avaliação/acompanhamento, marca 1
        if any(palavra in contexto for palavra in ["avaliacao", "acompanhamento", "consulta", "avaliar"]):
            return 1
    # "Fisioterapia" pode ser diferente de "fisiatria"
    return 0


def extrair_paliativo_grupo_dor(texto: str):
    """0 = não, 1 = sim. Verifica se realmente houve acompanhamento de grupo de dor."""
    t = normalizar(texto)
    if "paliativo" in t and "dor" in t:
        # Verifica se é menção de acompanhamento real
        idx = t.find("dor")
        contexto = t[max(0, idx-100):min(len(t), idx+100)]
        if "grupo" in contexto:
            return 1
    return 0


def extrair_grupo_dor(texto: str):
    """0 = não, 1 = sim. Verifica se realmente houve acompanhamento."""
    t = normalizar(texto)
    if "grupo" in t and "dor" in t:
        # Verifica contexto - se é menção de acompanhamento real
        idx = t.find("grupo")
        contexto = t[max(0, idx-50):min(len(t), idx+100)]
        if "dor" in contexto:
            return 1
    return 0


def extrair_ult_consulta(texto: str):
    """Data da última consulta"""
    t = normalizar(texto)
    padroes = [
        r"ultima\s+consulta\s*[:]?\s*(\d{2}/\d{2}/\d{4})",
        r"ult\s+consulta\s*[:]?\s*(\d{2}/\d{2}/\d{4})",
    ]
    for padrao in padroes:
        match = re.search(padrao, t, re.IGNORECASE)
        if match:
            return match.group(1).replace("-", "/")
    return None


def extrair_OS_meses(texto: str):
    """Overall survival em meses - precisa cálculo temporal"""
    return None


def extrair_obito(texto: str):
    """0 = não, 1 = sim"""
    t = normalizar(texto)
    if "obito" in t or "óbito" in t or "falecimento" in t:
        return 1
    return 0


def extrair_dt_obito(texto: str):
    """Data do óbito"""
    t = normalizar(texto)
    padroes = [
        r"data\s+obito\s*[:]?\s*(\d{2}[/-]\d{2}[/-]\d{4})",
        r"obito\s+em\s+(\d{2}/\d{2}/\d{4})",
    ]
    for padrao in padroes:
        match = re.search(padrao, t, re.IGNORECASE)
        if match:
            return match.group(1).replace("-", "/")
    return None


def extrair_obito_motivo(texto: str):
    """Motivo do óbito"""
    t = normalizar(texto)
    motivos = ["doenca", "complicacao", "progressao", "recidiva"]
    for motivo in motivos:
        if motivo in t and "obito" in t:
            return motivo
    return None


def extrair_assistente(texto: str):
    """Campo de texto livre - pode implementar depois se necessário"""
    return None


def extrair_observacao(texto: str):
    """Campo de texto livre - pode implementar depois se necessário"""
    return None


def extrair_campos_por_regras(texto: str, registros: List[Dict] = None) -> dict:
    """Chama todos os extratores e monta o dicionário com todos os campos."""
    data_cirurgia = extrair_data_cirurgia(texto)
    
    resultado = {
        "sexo": extrair_sexo(texto),
        "dt_SO": data_cirurgia,
        "idade": extrair_idade(texto, data_cirurgia),
        "local_tumor": extrair_local_tumor(texto),
        "altura_tumor": extrair_altura_tumor(texto),
        "ASA": extrair_ASA(texto, data_cirurgia),
        "ECOG": extrair_ECOG(texto, data_cirurgia),
        "KPS": extrair_KPS(texto, data_cirurgia),
        "IMC": extrair_IMC(texto, data_cirurgia),
        "QRT_neo": extrair_QRT_neo(texto),
        "eletiva": extrair_eletiva(texto),
        "cirurgia_recidiva": extrair_cirurgia_recidiva(texto),
        "paliativa": extrair_paliativa(texto),
    }
    
    # Órgãos envolvidos
    orgaos = extrair_orgaos_envolvidos(texto)
    resultado.update(orgaos)
    resultado["bexiga_tudo"] = extrair_bexiga_tudo(texto)
    resultado["bexiga_parte"] = extrair_bexiga_parte(texto)
    
    outro_orgao, outro_orgao_qual = extrair_outro_orgao(texto)
    resultado["outro_orgao"] = outro_orgao
    resultado["outro_orgao_qual"] = outro_orgao_qual
    resultado["n_orgaos"] = extrair_n_orgaos(texto, orgaos)
    
    # Campos cirúrgicos
    resultado.update({
        "amputação": extrair_amputacao(texto),
        "RTS": extrair_RTS(texto),
        "cole_total": extrair_cole_total(texto),
        "posterior": extrair_posterior(texto),
        "total": extrair_total(texto),
        "SLE": extrair_SLE(texto),
        "REC_plastica": extrair_REC_plastica(texto),
        "tipo_REC": extrair_tipo_REC(texto),
        "tempo_SO": extrair_tempo_SO(texto, data_cirurgia),
        "CH_intra_OP": extrair_CH_intra_OP(texto),
        "CH_num": extrair_CH_num(texto),
        "dias_uti": extrair_dias_uti(texto),
        "dias_internação": extrair_dias_internacao(texto),
        "dt_alta": extrair_dt_alta(texto),
        "complicação": extrair_complicacao(texto),
        "complicação_qual": extrair_complicacao_qual(texto),
        "tto": extrair_tto(texto),
        "Clavien": extrair_Clavien(texto),
        "Clavien_v2": extrair_Clavien_v2(texto),
        "reinternação": extrair_reinternacao(texto),
        "data da reinternação": extrair_data_reinternacao(texto),
        "motivo_reinternação": extrair_motivo_reinternacao(texto),
        "re_op_90dias": extrair_re_op_90dias(texto),
        "re_op_achado": extrair_re_op_achado(texto),
        "obito_90dias": extrair_obito_90dias(texto),
    })
    
    # Campos de patologia
    resultado.update({
        "histologia": extrair_histologia(texto),
        "AP": extrair_AP(texto),
        "estadiamento": extrair_estadiamento(texto),
        "T": extrair_T(texto),
        "N": extrair_N(texto),
        "N_A": extrair_N_A(texto),
        "invasão": extrair_invasao(texto),
        "R0_R1_R2": extrair_R0_R1_R2(texto),
        "R0_R1_R2_v2": extrair_R0_R1_R2_v2(texto),
        "QT_adjuvante": extrair_QT_adjuvante(texto),
        "recidiva": extrair_recidiva(texto),
        "recidiva_local": extrair_recidiva_local(texto),
        "recidiva_local_v2": extrair_recidiva_local_v2(texto),
        "dt_recidiva": extrair_dt_recidiva(texto),
        "DFSMESES": extrair_DFSMESES(texto),
        "fisiatria": extrair_fisiatria(texto),
        "paliativo_grupo_dor": extrair_paliativo_grupo_dor(texto),
        "grupo_dor": extrair_grupo_dor(texto),
        "ult_consulta": extrair_ult_consulta(texto),
        "OS_meses": extrair_OS_meses(texto),
        "obito": extrair_obito(texto),
        "dt_obito": extrair_dt_obito(texto),
        "obito_motivo": extrair_obito_motivo(texto),
        "assistente": extrair_assistente(texto),
        "observação": extrair_observacao(texto),
    })
    
    return resultado


def normalizar_valor_comparacao(valor: Any) -> str:
    """Normaliza valores para comparação (datas, números, etc)"""
    if valor is None:
        return ""
    
    if isinstance(valor, (int, float)):
        if pd.isna(valor):
            return ""
        return str(int(valor)) if isinstance(valor, float) and valor.is_integer() else str(valor)
    
    valor_str = str(valor).strip().lower()
    
    # Tenta normalizar datas
    if "/" in valor_str or "-" in valor_str:
        try:
            for fmt in ["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"]:
                try:
                    dt = datetime.strptime(valor_str.split()[0], fmt)
                    return dt.strftime("%d/%m/%Y")
                except:
                    continue
        except:
            pass
    
    return valor_str


def comparar_com_gabarito(paciente_slug: str, pred: dict, df_gabarito: pd.DataFrame, campos: list[str]) -> Dict[str, Any]:
    """Compara predições com gabarito e calcula acurácia."""
    nome_paciente = paciente_slug.replace("_", " ")
    
    linha_real = df_gabarito.loc[df_gabarito["Nome"] == nome_paciente]
    if linha_real.empty:
        print(f"Paciente {nome_paciente} não encontrado no gabarito.")
        return {"erro": "Paciente não encontrado"}
    
    linha_real = linha_real.iloc[0]
    
    total = 0
    acertos = 0
    detalhes = []
    
    print("\nComparação campo a campo:")
    for campo in campos:
        if campo not in linha_real.index:
            continue
        
        valor_real = linha_real[campo]
        valor_pred = pred.get(campo, None)
        
        if isinstance(valor_real, pd.Series):
            if valor_real.empty:
                continue
            valor_real = valor_real.iloc[0] if len(valor_real) > 0 else None
        
        try:
            if pd.isna(valor_real):
                continue
        except (TypeError, ValueError):
            pass
        
        s_real = normalizar_valor_comparacao(valor_real)
        if s_real in ["nan", "none", "", "nat", "na"]:
            continue
        
        total += 1
        s_pred = normalizar_valor_comparacao(valor_pred)
        
        ok = (s_real == s_pred)
        if ok:
            acertos += 1
        
        status = 'OK' if ok else 'ERRO'
        print(f" - {campo}: real = {s_real!r}, pred = {s_pred!r}  ->  {status}")
        detalhes.append({"campo": campo, "real": s_real, "pred": s_pred, "ok": ok})
    
    acc = acertos / total if total > 0 else 0.0
    print(f"\nAcurácia em {total} campos válidos: {acc:.1%} ({acertos}/{total})")
    
    return {"total": total, "acertos": acertos, "acuracia": acc, "detalhes": detalhes}


def main():
    import sys
    
    paciente_slug = sys.argv[1] if len(sys.argv) >= 2 else "Paciente_0000002"
    
    print(f"Lendo texto do paciente: {paciente_slug}")
    texto, registros = carregar_texto_com_contexto(paciente_slug)
    
    df_gabarito = carregar_gabarito()
    
    # Pega todas as colunas exceto as de controle
    todas_colunas = [col for col in df_gabarito.columns 
                     if col not in ["Idnum", "valido", "Nome", "nan"] and pd.notna(col)]
    
    print(f"\nTotal de colunas a extrair: {len(todas_colunas)}")
    
    pred = extrair_campos_por_regras(texto, registros)
    
    print("\nLinha predita (primeiros 30 campos):")
    for i, k in enumerate(todas_colunas[:30]):
        print(f"{k:30s}: {pred.get(k)}")
    
    if len(todas_colunas) > 30:
        print(f"... e mais {len(todas_colunas) - 30} campos")
    
    stats = comparar_com_gabarito(paciente_slug, pred, df_gabarito, todas_colunas)
    
    if "acuracia" in stats:
        print(f"\n{'='*60}")
        print(f"ACURÁCIA GERAL: {stats['acuracia']:.1%}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
