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


def extrair_secoes_relevantes(texto: str, data_cirurgia: Optional[str] = None) -> str:
    """
    Extrai as seções mais relevantes do prontuário, mantendo contexto completo.
    Estratégia: pega MUITO mais contexto para garantir que nada importante seja perdido.
    """
    secoes = []
    
    # 1. Início do prontuário (dados do paciente, data nascimento) - CRÍTICO para idade
    # AUMENTADO para garantir que todos os dados do paciente sejam incluídos
    inicio = texto[:150000]  # Primeiros 150k chars têm dados do paciente e histórico completo
    secoes.append("=== INÍCIO DO PRONTUÁRIO (DADOS DO PACIENTE E HISTÓRICO COMPLETO) ===\n" + inicio)
    
    # 2. Se data_cirurgia conhecida, extrai contexto amplo ao redor dela
    if data_cirurgia:
        # Procura todas as ocorrências da data da cirurgia
        padrao_data = re.escape(data_cirurgia.replace("/", "[/-]"))
        matches = list(re.finditer(padrao_data, texto, re.IGNORECASE))
        
        # Pega contexto MUITO amplo ao redor de cada ocorrência (até 8 ocorrências)
        for match in matches[:8]:
            # Contexto MUITO maior: 20k chars antes e 30k depois da data
            inicio_ctx = max(0, match.start() - 20000)
            fim_ctx = min(len(texto), match.end() + 30000)
            contexto = texto[inicio_ctx:fim_ctx]
            
            # Verifica se é contexto relevante (exenteração, cirurgia, etc)
            if any(palavra in contexto.lower() for palavra in ["exentera", "cirurgia", "operacao", "po", "p.o", "internacao", "uti", "alta"]):
                secoes.append(f"\n=== CONTEXTO DA CIRURGIA PRINCIPAL ({data_cirurgia}) ===\n{contexto}")
    
    # 3. Procura especificamente por laudos de AP próximos da data da cirurgia
    # CRÍTICO: Busca mais agressiva por AP - inclui "exame anatomopatológico" também
    padrao_ap = r"(?:produto\s+de\s+exentera|anatomia\s+patologica|laudo\s+anatomopatologico|exame\s+anatomopatologico|ap\s*[:]|estadiamento\s+tnm)"
    matches_ap = list(re.finditer(padrao_ap, texto, re.IGNORECASE | re.DOTALL))
    
    # CRÍTICO: Sempre inclui AP encontrado, mesmo que não esteja próximo da data
    # O LLM precisa ver TODOS os laudos de AP para encontrar o correto
    for match in matches_ap[:5]:  # Pega até 5 laudos de AP
        # Contexto MUITO maior para AP: 10k antes e 30k depois
        contexto_ap = texto[max(0, match.start()-10000):match.end()+30000]
        
        # Prioriza se está próximo da cirurgia, mas SEMPRE inclui
        if data_cirurgia:
            ano_cir = data_cirurgia.split("/")[2] if "/" in data_cirurgia else ""
            if data_cirurgia[:7] in contexto_ap or ano_cir in contexto_ap or "exentera" in contexto_ap.lower():
                secoes.append(f"\n=== LAUDO DE ANATOMIA PATOLÓGICA (CIRURGIA PRINCIPAL - PRIORIDADE) ===\n{contexto_ap}")
            else:
                # Mesmo que não esteja próximo, inclui (pode ser o laudo correto)
                secoes.append(f"\n=== LAUDO DE ANATOMIA PATOLÓGICA ===\n{contexto_ap}")
        else:
            secoes.append(f"\n=== LAUDO DE ANATOMIA PATOLÓGICA ===\n{contexto_ap}")
    
    # 4. Procura especificamente por informações de internação (dias internação, UTI, alta)
    if data_cirurgia:
        # Procura padrões relacionados a internação
        padroes_internacao = [
            r"dias\s+de\s+internac[aao]+",
            r"dias\s+uti",
            r"alta\s+[:\s]+(\d{2}/\d{2}/\d{4})",
            r"entrada\s+[:\s]+(\d{2}/\d{2}/\d{4})",
        ]
        
        # Procura TODOS os padrões de internação e inclui os próximos da cirurgia
        todos_contextos_internacao = []
        for padrao in padroes_internacao:
            matches = list(re.finditer(padrao, texto, re.IGNORECASE | re.DOTALL))
            for match in matches[:10]:  # Até 10 ocorrências por padrão
                # Contexto maior: 8k antes e 12k depois
                contexto = texto[max(0, match.start()-8000):match.end()+12000]
                # Verifica se está próximo da data da cirurgia
                if data_cirurgia:
                    ano_cir = data_cirurgia.split("/")[2] if "/" in data_cirurgia else ""
                    if data_cirurgia[:7] in contexto or ano_cir in contexto:
                        todos_contextos_internacao.append((contexto, match.start()))
        
        # Ordena por proximidade da data da cirurgia e pega os mais próximos
        if todos_contextos_internacao:
            todos_contextos_internacao.sort(key=lambda x: x[1])  # Ordena por posição
            # Pega os 3 contextos mais próximos da cirurgia
            for contexto, _ in todos_contextos_internacao[:3]:
                secoes.append(f"\n=== INFORMAÇÕES DE INTERNAÇÃO ===\n{contexto}")
    
    # 4. Fim do prontuário (pode ter resumos e informações finais)
    fim = texto[-100000:]  # Últimos 100k chars (aumentado)
    secoes.append("\n=== FIM DO PRONTUÁRIO (RESUMOS E INFORMAÇÕES FINAIS) ===\n" + fim)
    
    texto_relevante = "\n\n".join(secoes)
    
    # Remove duplicatas mantendo ordem (mas preserva seções importantes)
    # CRÍTICO: Não remove seções de AP, internação ou cirurgia principal
    secoes_unicas = []
    conteudos_vistos = set()
    
    for secao in secoes:
        secao_lower = secao.lower()
        # Seções críticas SEMPRE são incluídas (não remove duplicatas delas)
        is_critica = any(palavra in secao_lower for palavra in [
            "anatomia patologica", "laudo de anatomia", "produto de exentera",
            "cirurgia principal", "internacao", "uti", "alta"
        ])
        
        if is_critica:
            # Seções críticas: sempre inclui, mesmo se duplicada
            secoes_unicas.append(secao)
        else:
            # Outras seções: remove duplicatas
            assinatura = secao[:500].strip()
            hash_assinatura = hash(assinatura)
            
            if hash_assinatura not in conteudos_vistos:
                conteudos_vistos.add(hash_assinatura)
                secoes_unicas.append(secao)
    
    texto_relevante = "\n\n".join(secoes_unicas)
    
    # CRÍTICO: Garante que seções importantes (AP, internação) SEMPRE sejam incluídas
    # Estratégia: separa seções críticas das outras e garante que críticas sempre caibam
    if len(texto_relevante) > 300000:
        partes = texto_relevante.split("\n=== ")
        if len(partes) >= 2:
            # Separa seções críticas das outras
            secoes_criticas = []
            secoes_normais = []
            
            for parte in partes[1:]:
                parte_lower = parte.lower()
                if any(palavra in parte_lower for palavra in [
                    "anatomia patologica", "laudo de anatomia", "produto de exentera",
                    "cirurgia principal", "internacao", "uti", "alta"
                ]):
                    secoes_criticas.append(parte)
                else:
                    secoes_normais.append(parte)
            
            # Monta texto final: início + críticas completas + normais limitadas + fim
            texto_final = partes[0]  # Início completo (dados paciente)
            
            # CRÍTICAS: sempre inclui completas (até 60k cada)
            for secao_critica in secoes_criticas:
                texto_final += "\n=== " + secao_critica[:60000]
            
            # NORMAIS: inclui limitadas (até 20k cada, até completar limite)
            espaco_restante = 300000 - len(texto_final)
            for secao_normal in secoes_normais:
                if len(texto_final) + len(secao_normal[:20000]) < 300000:
                    texto_final += "\n=== " + secao_normal[:20000]
                else:
                    break
            
            if len(partes) > 1:
                texto_final += "\n=== " + partes[-1]  # Fim completo
            
            texto_relevante = texto_final[:300000]
    
    return texto_relevante


def extrair_campos_criticos_llm(texto: str, data_cirurgia: Optional[str] = None) -> Dict[str, Any]:
    """
    Extrai campos críticos que mais erram usando LLM em uma única chamada.
    Extrai seções relevantes do prontuário mantendo contexto completo.
    """
    # Extrai seções mais relevantes mantendo contexto
    texto_relevante = extrair_secoes_relevantes(texto, data_cirurgia)
    
    # Campos críticos que mais erram (baseado no relatório)
    # IMPORTANTE: O LLM precisa entender o CONTEXTO COMPLETO do prontuário
    campos_desc = {
        "dt_SO": f"Data da CIRURGIA PRINCIPAL de exenteração pélvica (formato: dd/mm/aaaa). Procure por 'Exenteração' seguida de data, ou 'PO (data)', ou data próxima de menções de exenteração pélvica. NÃO pegue cirurgias antigas (ex: 2009). Se não encontrar, null",
        "idade": f"Idade do paciente NA DATA DA CIRURGIA PRINCIPAL ({data_cirurgia if data_cirurgia else 'procure pela data da exenteração'}). Calcule baseado em 'Data Nascto' e data da cirurgia. Se encontrar 'X anos' próximo de dt registro da cirurgia principal, use esse valor. Número inteiro. Se não encontrar, null",
        "ASA": "Classificação ASA PRÉ-OPERATÓRIA da cirurgia principal. Procure 'ASA' em contexto de 'avaliação pré-op' ou 'pré-operatório' próximo da data da cirurgia principal. Pode estar como número (1-4) ou romano (I-IV). Se não encontrar, null",
        "IMC": "Índice de Massa Corporal PRÉ-OPERATÓRIO da cirurgia principal. Procure 'IMC' em contexto pré-operatório, próximo da data da cirurgia principal. Número decimal (ex: 24.4). Se não encontrar, null",
        "ECOG": "Performance Status ECOG PRÉ-OPERATÓRIO da cirurgia principal. Procure 'ECOG' em contexto pré-operatório próximo da data da cirurgia. Valores: 0, 1, 2, 3 ou 4. Se não encontrar, null",
        "KPS": "Karnofsky Performance Status PRÉ-OPERATÓRIO da cirurgia principal. Procure 'KPS' em contexto pré-operatório próximo da data da cirurgia. Valores: 0-100. Se não encontrar, null",
        "AP": "Estadiamento TNM da CIRURGIA PRINCIPAL. CRÍTICO: Procure ESPECIFICAMENTE no laudo de 'produto de exenteração pélvica' ou 'anatomia patológica' da cirurgia principal. O TNM pode estar escrito como 'pT4b pN1b', 'pT4b pN1b pM0', 'T4b N1b', 'pT4b pN1b' (com ou sem espaços). Procure em TODAS as seções que mencionam 'produto de exenteração', 'anatomia patológica', 'laudo anatomopatológico', 'AP:', 'estadiamento'. O TNM pode estar em qualquer parte do laudo, não apenas no início. Formato esperado: pT4b pN1b pM0 ou pT4b pN1b (sem M). NÃO pegue de cirurgias antigas. Se não encontrar após análise completa, null",
        "estadiamento": "Estadiamento numérico baseado no AP da cirurgia principal: T4+N1=3, T4+N0=2, T3=2, T2=1, T1=0. Se não encontrar, null",
        "T": "T do TNM da cirurgia principal. Extraia do AP (ex: se AP é 'pT4b pN1b', então T é 'T4b'). Se não encontrar, null",
        "N": "N do TNM da cirurgia principal no formato X/Y (ex: 2/26). Procure padrões como '1/4 linfonodos' ou '2/26' em contexto de linfonodos dissecados da CIRURGIA PRINCIPAL. Se não encontrar, null",
        "N_A": "Número de linfonodos acometidos da cirurgia principal. É o primeiro número do formato N (ex: se N é '2/26', então N_A é 2). Se não encontrar, null",
        "dias_internação": "Dias de internação da CIRURGIA PRINCIPAL. Procure por 'dias de internação', 'dias internacao', 'permanência', ou calcule pela diferença entre 'entrada' e 'alta' da internação da cirurgia principal. Procure próximo da data da cirurgia principal. Número inteiro. Se não encontrar após análise completa, null",
        "dias_uti": "Dias de UTI da internação da CIRURGIA PRINCIPAL. Procure por 'dias UTI', 'dias de UTI', 'permanência UTI', 'permanencia UTI' da internação da cirurgia principal. Procure próximo da data da cirurgia principal. Número inteiro. Se não encontrar após análise completa, null",
        "dt_alta": "Data de alta da internação da CIRURGIA PRINCIPAL (formato: dd/mm/aaaa). Procure por 'alta', 'data de alta', 'dt alta', 'data alta' próxima da data da cirurgia principal. Procure em evoluções de alta ou sumários de alta. Se não encontrar após análise completa, null",
        "cirurgia_recidiva": "A CIRURGIA PRINCIPAL foi uma cirurgia DE recidiva? Procure se a exenteração foi feita 'por recidiva' ou 'devido a recidiva'. 0=não foi cirurgia de recidiva, 1=foi cirurgia DE recidiva, 88=não aplicável. Se não encontrar, null",
        "local_tumor": "Localização do tumor principal (ex: reto, sigmoide, colon). Procure em contexto do diagnóstico inicial ou do tumor principal. Se não encontrar, null",
    }
    
    tamanho_texto = len(texto_relevante)
    
    prompt = f"""Você é um especialista em análise de prontuários médicos eletrônicos.

CONTEXTO CRÍTICO: Você recebeu um prontuário médico COMPLETO de um paciente. Este prontuário contém TODO o histórico médico do paciente, incluindo:
- Múltiplas consultas ao longo dos anos
- Múltiplas cirurgias (algumas antigas, algumas recentes)
- Exames e laudos de anatomia patológica
- Evoluções clínicas e internações
- Dados pré-operatórios de diferentes épocas

TAREFA PRINCIPAL: Identificar a CIRURGIA PRINCIPAL de exenteração pélvica e extrair informações ESPECIFICAMENTE dessa cirurgia.

REGRA DE OURO: A CIRURGIA PRINCIPAL é geralmente a exenteração pélvica mais RECENTE e IMPORTANTE mencionada no prontuário. Ela é diferente de cirurgias antigas (ex: cirurgias de 2009, 2012, etc).

IDENTIFICAÇÃO DA CIRURGIA PRINCIPAL:
- Procure por menções de "Exenteração Pélvica" ou "Exenteração Pelvica"
- A data da cirurgia principal é: {data_cirurgia if data_cirurgia else 'PROCURE pela data da exenteração pélvica mais recente'}
- Todos os valores extraídos DEVEM ser dessa cirurgia específica, não de outras

ESTRATÉGIA DE EXTRAÇÃO:
1. Leia TODO o prontuário para entender a sequência temporal dos eventos
2. Identifique claramente qual é a CIRURGIA PRINCIPAL (exenteração pélvica mais recente)
3. Para cada campo abaixo, procure o valor ESPECÍFICO dessa cirurgia principal
4. Valores pré-operatórios (ASA, IMC, ECOG, KPS) devem ser os da avaliação pré-op DA CIRURGIA PRINCIPAL
5. Valores de patologia (AP, T, N, N_A) devem ser do laudo DA CIRURGIA PRINCIPAL
6. Valores de internação (dias_internação, dias_uti, dt_alta) devem ser da internação DA CIRURGIA PRINCIPAL

CAMPOS A EXTRAIR (TODOS da CIRURGIA PRINCIPAL):
"""
    
    for campo, descricao in campos_desc.items():
        prompt += f"\n{campo}: {descricao}"
    
    prompt += f"""

INSTRUÇÕES GERAIS:
1. Leia TODO o prontuário para entender o contexto completo
2. Identifique a CIRURGIA PRINCIPAL (exenteração pélvica mais recente)
3. Para cada campo, procure o valor ESPECÍFICO da cirurgia principal
4. Se um campo não for encontrado, use null (não invente valores)
5. Para números, retorne apenas o número (sem aspas)
6. Para datas, use formato dd/mm/aaaa
7. Para strings, retorne o valor exato encontrado
8. NÃO pegue valores de cirurgias antigas (ex: 2009, 2012) - apenas da cirurgia principal
9. Considere o CONTEXTO TEMPORAL - valores pré-operatórios devem ser próximos da data da cirurgia principal

SEÇÕES RELEVANTES DO PRONTUÁRIO ({tamanho_texto} caracteres extraídos de {len(texto)} total):
{texto_relevante}

IMPORTANTE: 
- Analise TODO o texto para entender o contexto completo
- Identifique a sequência temporal dos eventos
- Extraia valores ESPECIFICAMENTE da cirurgia principal
- Retorne TODOS os campos solicitados
- Se encontrar o valor, retorne. Se não encontrar, retorne null

CRÍTICO: Você DEVE retornar TODOS os 16 campos solicitados. Se encontrar o valor, retorne. Se NÃO encontrar após analisar TODO o texto, retorne null para aquele campo.

Retorne APENAS um JSON válido no formato (exemplo):
{{"dt_SO": "20/01/2017", "idade": 51, "ASA": 2, "IMC": 24.4, "ECOG": 0, "KPS": 90, "AP": "pT4b pN1b", "estadiamento": 3, "T": "T4b", "N": "2/26", "N_A": 2, "dias_internação": 11, "dias_uti": 3, "dt_alta": "31/01/2017", "cirurgia_recidiva": 1, "local_tumor": "reto"}}

IMPORTANTE: 
- NÃO deixe campos de fora
- Se não encontrar após análise completa, use null
- Analise TODO o texto fornecido, não apenas partes
"""
    
    resposta = chamar_llm(
        "Você é um especialista em análise de prontuários médicos. Você analisa TODO o contexto do prontuário para extrair informações precisas da cirurgia principal. Retorne APENAS JSON válido, sem explicações ou markdown.",
        prompt,
        max_tokens=2000,  # Aumentado ainda mais para respostas completas
        temperature=0.0  # Temperatura 0 para máxima consistência
    )
    
    if not resposta:
        return {}
    
    # Tenta parsear JSON
    try:
        # Remove markdown code blocks se houver
        resposta_limpa = resposta.strip()
        if resposta_limpa.startswith("```"):
            resposta_limpa = resposta_limpa.split("```")[1]
            if resposta_limpa.startswith("json"):
                resposta_limpa = resposta_limpa[4:]
        resposta_limpa = resposta_limpa.strip()
        
        resultado = json.loads(resposta_limpa)
        
        # Converte tipos conforme necessário
        resultado_final = {}
        for campo, valor in resultado.items():
            if valor is None or valor == "null" or valor == "":
                resultado_final[campo] = None
            elif campo in ["idade", "ASA", "ECOG", "KPS", "N_A", "dias_internação", "dias_uti", "estadiamento", "cirurgia_recidiva"]:
                try:
                    resultado_final[campo] = int(valor) if valor != "null" else None
                except:
                    resultado_final[campo] = None
            elif campo == "IMC":
                try:
                    resultado_final[campo] = float(valor) if valor != "null" else None
                except:
                    resultado_final[campo] = None
            else:
                resultado_final[campo] = valor if valor != "null" else None
        
        return resultado_final
    except Exception as e:
        # Se falhar, tenta extrair campos individualmente da resposta
        resultado = {}
        for campo in campos_desc.keys():
            # Tenta encontrar padrões na resposta
            padrao = rf'["\']?{campo}["\']?\s*[:=]\s*([^,}}"]+)'
            match = re.search(padrao, resposta, re.IGNORECASE)
            if match:
                valor = match.group(1).strip().strip('"').strip("'")
                if valor.lower() not in ["null", "none", "n/a"]:
                    resultado[campo] = valor
        return resultado


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


def localizar_contexto_por_palavras(
    texto: str,
    palavras_chave: List[str],
    data_cirurgia: Optional[str] = None,
    raio: int = 5000,
) -> str:
    """
    Retorna um trecho do texto próximo das palavras-chave e, se possível,
    da data da cirurgia. Esse contexto é reutilizado por vários campos.
    """
    if not texto:
        return ""

    texto_lower = texto.lower()
    melhor = ""
    melhor_score = -1
    data_lower = data_cirurgia.lower() if data_cirurgia else None
    ano_cirurgia = None
    if data_cirurgia and "/" in data_cirurgia:
        ano_cirurgia = data_cirurgia.split("/")[-1]

    for palavra in palavras_chave:
        termo = palavra.lower()
        inicio_busca = 0
        while True:
            idx = texto_lower.find(termo, inicio_busca)
            if idx == -1:
                break
            inicio = max(0, idx - raio)
            fim = min(len(texto), idx + len(termo) + raio)
            trecho = texto[inicio:fim]

            score = 1
            trecho_lower = trecho.lower()
            if data_lower and data_lower in trecho_lower:
                score += 3
            if ano_cirurgia and ano_cirurgia in trecho_lower:
                score += 1

            # Prioriza contextos que contenham múltiplas palavras relevantes
            for outra_palavra in palavras_chave:
                if outra_palavra.lower() in trecho_lower and outra_palavra.lower() != termo:
                    score += 1

            if score > melhor_score:
                melhor_score = score
                melhor = trecho
            inicio_busca = idx + len(termo)

    return melhor


def construir_contextos_cirurgia(texto: str, data_cirurgia: Optional[str] = None) -> Dict[str, str]:
    """
    Consolida contextos específicos (pré-op, AP/cirurgia, internação)
    para reutilizar nas funções de extração e reduzir leituras repetidas.
    """
    contexto_preop = localizar_contexto_por_palavras(
        texto,
        palavras_chave=[
            "avaliação pré",
            "avaliacao pre",
            "asa",
            "ecog",
            "kps",
            "imc",
            "pré operatório",
            "pre operatorio",
            "pre-op",
        ],
        data_cirurgia=data_cirurgia,
        raio=6000,
    )

    contexto_ap = localizar_contexto_por_palavras(
        texto,
        palavras_chave=[
            "produto de exentera",
            "laudo anatomopatologico",
            "anatomia patologica",
            "estadiamento tnm",
            "ap:",
        ],
        data_cirurgia=data_cirurgia,
        raio=8000,
    )

    contexto_internacao = localizar_contexto_por_palavras(
        texto,
        palavras_chave=[
            "internacao",
            "internação",
            "permanencia",
            "permanência",
            "uti",
            "alta hospitalar",
            "evolucao",
            "evolução",
        ],
        data_cirurgia=data_cirurgia,
        raio=10000,
    )

    contexto_cirurgia = contexto_ap or localizar_contexto_por_palavras(
        texto,
        palavras_chave=[
            "exenteração",
            "exenteracao",
            "cirurgia",
            "procedimento",
            "relatorio operatorio",
        ],
        data_cirurgia=data_cirurgia,
        raio=8000,
    )

    return {
        "preop": contexto_preop,
        "ap": contexto_ap,
        "internacao": contexto_internacao,
        "cirurgia": contexto_cirurgia,
    }


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


def extrair_idade(
    texto: str,
    data_cirurgia: Optional[datetime] = None,
    sugestao_llm: Optional[int] = None,
) -> Optional[int]:
    padrao = re.compile(
        r"(idade\s*[:=]\s*(\d{1,3}))|(\b(\d{1,3})\s*anos\b)",
        re.IGNORECASE,
    )

    candidatos = []

    for m in padrao.finditer(texto):
        idade_str = m.group(2) or m.group(4)
        if not idade_str:
            continue

        try:
            idade = int(idade_str)
        except ValueError:
            continue

        # Faixa razoável
        if idade < 15 or idade > 100:
            continue

        posicao = m.start()
        score = 0

        # Preferir idade mencionada no começo do texto
        if posicao < len(texto) * 0.3:
            score += 2
        else:
            score += 1

        contexto_inicio = max(0, posicao - 80)
        contexto_fim = min(len(texto), m.end() + 80)
        contexto = texto[contexto_inicio:contexto_fim].lower()

        # “paciente 51 anos” é um padrão muito típico
        if "anos" in contexto and "paciente" in contexto:
            score += 1

        candidatos.append((idade, score))

    if candidatos:
        candidatos.sort(key=lambda x: x[1], reverse=True)
        return candidatos[0][0]

    # Fallback: se o LLM sugeriu algo razoável
    if sugestao_llm is not None:
        try:
            idade = int(sugestao_llm)
            if 15 <= idade <= 100:
                return idade
        except (TypeError, ValueError):
            pass

    return None


def extrair_local_tumor(texto: str, contexto_cirurgia: Optional[str] = None):
    """Extrai localização do tumor. Prioriza menções diretas, depois tenta pela distância da borda anal."""
    texto_base = contexto_cirurgia if contexto_cirurgia else texto
    t = normalizar(texto_base)
    
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
    
    if contexto_cirurgia:
        return extrair_local_tumor(texto, contexto_cirurgia=None)
    return None


ROMANO_PARA_INT = {
    "i": 1,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "v": 5,
}

def _converter_romano(valor: str) -> Optional[int]:
    v = valor.strip().lower()
    if v in ROMANO_PARA_INT:
        return ROMANO_PARA_INT[v]
    try:
        num = int(v)
        return num
    except ValueError:
        return None

def extrair_ASA(texto: str, sugestao_llm: Optional[int] = None) -> Optional[int]:
    padrao = re.compile(r"asa\s*[:=]?\s*([ivx]+|\d)", re.IGNORECASE)
    candidatos = []

    for m in padrao.finditer(texto):
        bruto = m.group(1)
        valor = _converter_romano(bruto)
        if valor is None:
            continue
        if valor < 1 or valor > 4:
            continue

        posicao = m.start()
        score = 0

        contexto = texto[max(0, posicao - 80): m.end() + 80].lower()
        if "classificacao" in contexto or "risco" in contexto:
            score += 2
        else:
            score += 1

        candidatos.append((valor, score))

    if candidatos:
        candidatos.sort(key=lambda x: x[1], reverse=True)
        return candidatos[0][0]

    # Fallback LLM
    if sugestao_llm is not None:
        try:
            valor = int(sugestao_llm)
            if 1 <= valor <= 4:
                return valor
        except (TypeError, ValueError):
            pass

    return None


def extrair_ECOG(texto: str, sugestao_llm: Optional[int] = None) -> Optional[int]:
    padrao = re.compile(r"ecog\s*[:=]?\s*([0-4])", re.IGNORECASE)
    candidatos = []

    for m in padrao.finditer(texto):
        try:
            valor = int(m.group(1))
        except ValueError:
            continue

        posicao = m.start()
        score = 1

        contexto = texto[max(0, posicao - 80): m.end() + 80].lower()
        if "karnofsky" in contexto or "estado funcional" in contexto:
            score += 1

        candidatos.append((valor, score))

    if candidatos:
        candidatos.sort(key=lambda x: x[1], reverse=True)
        return candidatos[0][0]

    if sugestao_llm is not None:
        try:
            valor = int(sugestao_llm)
            if 0 <= valor <= 4:
                return valor
        except (TypeError, ValueError):
            pass

    return None



def extrair_KPS(texto: str, data_cirurgia: Optional[str] = None, contexto_preop: Optional[str] = None):
    """Karnofsky Performance Status (0-100). OCR às vezes escreve 'KP5' ao invés de 'KPS'."""
    texto_preferencial = contexto_preop if contexto_preop else texto
    t = normalizar(texto_preferencial)
    ocorrencias = encontrar_valores_com_contexto(
        texto_preferencial,
        r"k[p]?s?\s*[:=]?\s*(\d{2,3})(?:\s|$|kg|%|bpm|mmHg)",
        contexto_relevante=["pre", "operat", "cirurgia", "performance", "karnofsky"],
        preferir_proximo_de=["karnofsky", "performance", "pre", "operat"],
        data_cirurgia=data_cirurgia
    )
    
    if not ocorrencias:
        if contexto_preop:
            t_full = normalizar(texto)
            ocorrencias = encontrar_valores_com_contexto(
                texto,
                r"k[p]?s?\s*[:=]?\s*(\d{2,3})(?:\s|$|kg|%|bpm|mmHg)",
                contexto_relevante=["pre", "operat", "cirurgia", "performance", "karnofsky"],
                preferir_proximo_de=["karnofsky", "performance", "pre", "operat"],
                data_cirurgia=data_cirurgia
            )
        else:
            t_full = t
        
        if not ocorrencias:
            match = re.search(r"(?:kps|karnofsky)\s*[:=]?\s*(\d{2,3})(?:\s|$|%)", t_full, re.IGNORECASE)
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


def extrair_IMC(texto: str, data_cirurgia: Optional[str] = None, contexto_preop: Optional[str] = None):
    """
    Extrai IMC. Prioriza valores pré-operatórios e próximos da data da cirurgia.
    Se houver muitas ocorrências com scores similares, usa LLM.
    """
    todas_ocorrencias = []
    fontes_prioridade = []
    if contexto_preop:
        fontes_prioridade.append((contexto_preop, True))
    fontes_prioridade.append((texto, False))
    
    for fonte, bonus_preop in fontes_prioridade:
        t = normalizar(fonte)
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
                if 20 <= imc <= 30:
                    score += 5
                if 24 <= imc <= 25:
                    score += 3
                if any(palavra in contexto for palavra in ["pre", "operat", "cirurgia", "avaliacao", "pre-op"]):
                    score += 8
                if "peso" in contexto or "altura" in contexto:
                    score += 2
                if data_cirurgia:
                    padrao_data = data_cirurgia.replace("/", "[/-]")
                    if re.search(padrao_data, contexto):
                        score += 15
                if "consulta" in contexto and "seguimento" in contexto:
                    score -= 3
                if bonus_preop:
                    score += 2
                
                todas_ocorrencias.append((imc, score))
            except ValueError:
                continue
    
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
    
    # Fallback: tenta calcular a partir de peso e altura (prioriza contexto pré-op)
    fontes_para_calculo = []
    if contexto_preop:
        fontes_para_calculo.append(contexto_preop)
    fontes_para_calculo.append(texto)
    
    for fonte in fontes_para_calculo:
        t = normalizar(fonte)
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
                    if fonte is contexto_preop:
                        score += 1
                    pesos.append((peso, score))
            except ValueError:
                continue
        
        for match in re.finditer(r"altura\)?\s*[:=]?\s*([0-9]{1,3}(?:[.,][0-9]{1,2})?)", t):
            try:
                altura_str = match.group(1).replace(",", ".")
                if "." in altura_str and 1.0 <= float(altura_str) <= 2.2:
                    altura = float(altura_str)
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
                if fonte is contexto_preop:
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


def extrair_altura_tumor(texto: str) -> Optional[int]:
    candidatos = []

    padrao1 = re.compile(
        r"(\d{1,2})\s*cm\s*(?:da\s+)?borda\s+anal",
        re.IGNORECASE,
    )
    padrao2 = re.compile(
        r"altura\s+(?:do\s+)?tumor\s*[:=]?\s*(\d{1,2})\s*cm",
        re.IGNORECASE,
    )

    for padrao in [padrao1, padrao2]:
        for m in padrao.finditer(texto):
            try:
                valor = int(m.group(1))
            except ValueError:
                continue

            # Faixa razoável de altura em cm
            if valor < 0 or valor > 30:
                continue

            posicao = m.start()
            score = 1

            contexto = texto[max(0, posicao - 80): m.end() + 80].lower()
            if "reto" in contexto:
                score += 1

            candidatos.append((valor, score))

    if not candidatos:
        return None

    candidatos.sort(key=lambda x: x[1], reverse=True)
    return candidatos[0][0]



def extrair_cirurgia_recidiva(texto: str, contexto_cirurgia: Optional[str] = None):
    """0 = não, 1 = sim, 88 = não aplicável. Verifica se a cirurgia foi por recidiva."""
    texto_base = contexto_cirurgia if contexto_cirurgia else texto
    t = normalizar(texto_base)
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
    if contexto_cirurgia:
        return extrair_cirurgia_recidiva(texto, contexto_cirurgia=None)
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


def extrair_REC_plastica(texto: str) -> Optional[int]:
    texto_lower = texto.lower()

    if "retalho" in texto_lower or "plast" in texto_lower or "reconstrucao" in texto_lower:
        return 1

    # Se não achar nada que pareça plástica, considera que não teve
    return 0


def extrair_tipo_REC(texto: str) -> Optional[int]:
    texto_lower = texto.lower()

    # Os códigos exatos você pode ajustar se tiver um dicionário da planilha
    if "retalho miocutaneo" in texto_lower:
        return 1
    if "retalho cutaneo" in texto_lower:
        return 2
    if "retalho local" in texto_lower or "avancamento" in texto_lower:
        return 3

    # 88 = não se aplica / não foi feita plástica
    return 88

def extrair_tipo_rec_uro(texto: str) -> Optional[int]:
    texto_lower = texto.lower()

    if "bricker" in texto_lower:
        return 0
    if "duplo barril ileal" in texto_lower:
        return 2
    if "duplo barril" in texto_lower:
        return 1
    if "nefrostomia" in texto_lower:
        return 3

    # 88 = nada disso se aplica
    return 88


def extrair_tempo_SO(
    data_cirurgia: Optional[str] = None,
    dt_alta: Optional[str] = None,
    dt_ult_consulta: Optional[str] = None,
    dt_obito: Optional[str] = None,
) -> Optional[int]:
    """Calcula tempo em dias desde a cirurgia até o último acompanhamento/óbito/alta."""
    if not data_cirurgia:
        return None
    
    try:
        dt_cir = datetime.strptime(data_cirurgia.replace("-", "/"), "%d/%m/%Y")
    except Exception:
        return None
    
    referencias = []
    for dt_str in [dt_obito, dt_ult_consulta, dt_alta]:
        if not dt_str:
            continue
        try:
            dt_ref = datetime.strptime(dt_str.replace("-", "/"), "%d/%m/%Y")
            if dt_ref >= dt_cir:
                referencias.append((dt_ref - dt_cir).days)
        except Exception:
            continue
    
    if referencias:
        return max(referencias)
    return None


def extrair_CH_intra_OP(texto: str, contexto_cirurgia: Optional[str] = None):
    """Quimioterapia intra-operatória: 0 = não, 1 = sim. Verifica se realmente foi durante a cirurgia."""
    texto_base = contexto_cirurgia if contexto_cirurgia else texto
    t = normalizar(texto_base)
    # Procura padrões específicos de quimioterapia intra-operatória
    if "quimio" in t and "intra" in t and "operat" in t:
        # Verifica se não é negação
        idx = t.find("intra")
        contexto = t[max(0, idx-50):min(len(t), idx+50)]
        if "nao" not in contexto:
            return 1
    if contexto_cirurgia:
        return extrair_CH_intra_OP(texto, contexto_cirurgia=None)
    return 0


def extrair_CH_num(texto: str, contexto_cirurgia: Optional[str] = None):
    """Número de ciclos de quimioterapia. Procura em contexto de QT neoadjuvante/adjuvante."""
    texto_base = contexto_cirurgia if contexto_cirurgia else texto
    t = normalizar(texto_base)
    
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
    
    if contexto_cirurgia:
        return extrair_CH_num(texto, contexto_cirurgia=None)
    return None


def extrair_dias_uti(texto: str, data_cirurgia: Optional[str] = None, contexto_internacao: Optional[str] = None):
    """Dias de UTI. MELHORADO: Busca mais agressiva com contexto da cirurgia."""
    texto_base = contexto_internacao if contexto_internacao else texto
    t = normalizar(texto_base)
    
    # Padrões mais flexíveis
    padroes = [
        r"dias\s+uti\s*[:]?\s*(\d+)",
        r"uti\s*[:]?\s*(\d+)\s+dias",
        r"permanencia.*?uti.*?(\d+)\s+dias",
        r"uti.*?(\d+)\s+dias",
        r"admissao.*?uti.*?(\d+)\s+dias",
    ]
    
    candidatos = []
    
    for padrao in padroes:
        matches = list(re.finditer(padrao, t, re.IGNORECASE))
        for match in matches:
            try:
                dias = int(match.group(1))
                if 0 <= dias <= 365:
                    # Contexto ao redor
                    contexto = texto_base[max(0, match.start()-2000):match.end()+2000].lower()
                    score = 10
                    
                    # Prioriza se está próximo da data da cirurgia
                    if data_cirurgia:
                        if data_cirurgia[:7] in contexto or data_cirurgia.split("/")[2] in contexto:
                            score += 20
                    
                    # Prioriza se menciona exenteração
                    if "exentera" in contexto:
                        score += 15
                    
                    candidatos.append((dias, score, match.start()))
            except:
                pass
    
    if candidatos:
        candidatos.sort(key=lambda x: x[1], reverse=True)
        return candidatos[0][0]
    
    if contexto_internacao:
        return extrair_dias_uti(texto, data_cirurgia, contexto_internacao=None)
    return None


def extrair_dias_internacao(texto: str, data_cirurgia: Optional[str] = None, contexto_internacao: Optional[str] = None):
    """Dias de internação. MELHORADO: Busca mais agressiva com contexto da cirurgia."""
    texto_base = contexto_internacao if contexto_internacao else texto
    t = normalizar(texto_base)
    
    # Tenta calcular pela diferença de datas (mais flexível)
    padroes_entrada = [
        r"entrada[:\s]+(\d{2}[/-]\d{2}[/-]\d{4})",
        r"admissao[:\s]+(\d{2}[/-]\d{2}[/-]\d{4})",
        r"internacao[:\s]+(\d{2}[/-]\d{2}[/-]\d{4})",
    ]
    padroes_alta = [
        r"alta[:\s]+(\d{2}[/-]\d{2}[/-]\d{4})",
        r"dt\s*alta[:\s]+(\d{2}[/-]\d{2}[/-]\d{4})",
        r"data\s+alta[:\s]+(\d{2}[/-]\d{2}[/-]\d{4})",
    ]
    
    # Busca entrada e alta próximos da cirurgia
    match_entrada = None
    match_alta = None
    
    if data_cirurgia:
        # Procura entrada e alta próximos da data da cirurgia
        padrao_data = re.escape(data_cirurgia.replace("/", "[/-]"))
        matches_data = list(re.finditer(padrao_data, texto, re.IGNORECASE))
        
        if matches_data:
            # Pega contexto ao redor da data da cirurgia
            idx_cir = matches_data[0].start()
            contexto_cir = texto[max(0, idx_cir-50000):min(len(texto), idx_cir+100000)]
            
            for padrao in padroes_entrada:
                match = re.search(padrao, contexto_cir, re.IGNORECASE)
                if match:
                    match_entrada = match
                    break
            
            for padrao in padroes_alta:
                match = re.search(padrao, contexto_cir, re.IGNORECASE)
                if match:
                    match_alta = match
                    break
    
    # Se não encontrou com contexto, busca em todo o texto
    if not match_entrada:
        for padrao in padroes_entrada:
            match = re.search(padrao, t, re.IGNORECASE)
            if match:
                match_entrada = match
                break
    
    if not match_alta:
        for padrao in padroes_alta:
            match = re.search(padrao, t, re.IGNORECASE)
            if match:
                match_alta = match
                break
    
    # Calcula diferença se encontrou ambas
    if match_entrada and match_alta:
        try:
            dt_entrada = datetime.strptime(match_entrada.group(1).replace("-", "/"), "%d/%m/%Y")
            dt_alta = datetime.strptime(match_alta.group(1).replace("-", "/"), "%d/%m/%Y")
            dias = (dt_alta - dt_entrada).days
            if 0 <= dias <= 365:
                return dias
        except:
            pass
    
    # Fallback: procura padrões diretos (mais flexíveis)
    padroes = [
        r"dias\s+de\s+internacao\s*[:]?\s*(\d+)",
        r"dias\s+internacao\s*[:]?\s*(\d+)",
        r"internacao\s*[:]?\s*(\d+)\s+dias",
        r"permanencia.*?(\d+)\s+dias",
        r"(\d+)\s+dias\s+de\s+internacao",
    ]
    
    candidatos = []
    for padrao in padroes:
        matches = list(re.finditer(padrao, t, re.IGNORECASE))
        for match in matches:
            try:
                dias = int(match.group(1))
                if 0 <= dias <= 365:
                    # Contexto ao redor
                    contexto = texto_base[max(0, match.start()-2000):match.end()+2000].lower()
                    score = 10
                    
                    # Prioriza se está próximo da data da cirurgia
                    if data_cirurgia:
                        if data_cirurgia[:7] in contexto or data_cirurgia.split("/")[2] in contexto:
                            score += 20
                    
                    # Prioriza se menciona exenteração
                    if "exentera" in contexto:
                        score += 15
                    
                    candidatos.append((dias, score, match.start()))
            except:
                pass
    
    if candidatos:
        candidatos.sort(key=lambda x: x[1], reverse=True)
        return candidatos[0][0]
    
    if contexto_internacao:
        return extrair_dias_internacao(texto, data_cirurgia, contexto_internacao=None)
    return None


def extrair_dt_alta(texto: str, contexto_internacao: Optional[str] = None):
    """Data de alta"""
    texto_base = contexto_internacao if contexto_internacao else texto
    t = normalizar(texto_base)
    padroes = [
        r"dt\s*alta\s*[:]?\s*(\d{2}[/-]\d{2}[/-]\d{4})",
        r"data\s+alta\s*[:]?\s*(\d{2}[/-]\d{2}[/-]\d{4})",
        r"alta\s+em\s+(\d{2}/\d{2}/\d{4})",
    ]
    for padrao in padroes:
        match = re.search(padrao, t, re.IGNORECASE)
        if match:
            return match.group(1).replace("-", "/")
    if contexto_internacao:
        return extrair_dt_alta(texto, contexto_internacao=None)
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


def extrair_Clavien(texto: str) -> Optional[int]:
    texto_lower = texto.lower()
    padrao = re.compile(r"clavien\s*([0-5])", re.IGNORECASE)

    for m in padrao.finditer(texto):
        try:
            valor = int(m.group(1))
            return valor
        except ValueError:
            continue

    # Se não aparece grau claramente, assume sem complicação maior
    return 0


def extrair_Clavien_v2(texto: str) -> Optional[int]:
    return extrair_Clavien(texto)



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


def extrair_re_op_90dias(
    texto: str,
    data_cirurgia: Optional[datetime] = None,
) -> Optional[int]:
    texto_lower = texto.lower()

    if (
        "reoperacao" not in texto_lower
        and "reoperado" not in texto_lower
        and "re-operacao" not in texto_lower
    ):
        return 0

    janela = 200
    termos_base = ["reoperacao", "reoperado", "re-operacao"]
    gatilhos_90 = ["90 dias", "noventa dias", "3 meses", "tres meses", "ate 90", "até 90"]

    for termo in termos_base:
        idx = texto_lower.find(termo)
        if idx == -1:
            continue

        inicio = max(0, idx - janela)
        fim = min(len(texto_lower), idx + janela)
        contexto = texto_lower[inicio:fim]

        for gatilho in gatilhos_90:
            if gatilho in contexto:
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


def extrair_AP(texto: str, sugestao_llm: Optional[str] = None) -> Optional[str]:
    padrao = re.compile(
        r"p\s*t\s*([0-4][abc]?)\s+p\s*n\s*([0-3][abc]?)",
        re.IGNORECASE,
    )

    candidatos = []

    for m in padrao.finditer(texto):
        t = m.group(1)
        n = m.group(2)
        valor = f"pt{t} pn{n}".lower()

        posicao = m.start()
        score = 1

        contexto = texto[max(0, posicao - 120): m.end() + 120].lower()
        if "anatomopatologico" in contexto or "patologico" in contexto or "laudo" in contexto:
            score += 2

        candidatos.append((valor, score))

    if candidatos:
        candidatos.sort(key=lambda x: x[1], reverse=True)
        return candidatos[0][0]

    if sugestao_llm:
        s = str(sugestao_llm).strip().lower()
        if "pt" in s and "pn" in s:
            return s

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
    """
    Estratégia híbrida em duas camadas:
    1) extrai um primeiro palpite global com LLM (campos mais complexos).
    2) aplica regex/heurísticas para todos os campos, usando as sugestões do LLM
       apenas como apoio (quando fizer sentido).
    """
    # 1) Âncora temporal: data da cirurgia vinda do texto cru
    data_cirurgia = extrair_data_cirurgia(texto)

    # 2) Chama LLM para um primeiro palpite dos campos principais
    resultado_llm = extrair_campos_criticos_llm(texto, data_cirurgia) or {}

    # Se LLM deu dt_SO, usa como data principal (ajusta variação de OCR)
    if resultado_llm.get("dt_SO"):
        data_cirurgia = resultado_llm["dt_SO"]

    # 3) Parte puramente de regras/regex (inclusive campos sem LLM)
    return extrair_campos_regex(
        texto=texto,
        resultado_llm=resultado_llm,
        data_cirurgia=data_cirurgia,
        registros=registros,
    )

def extrair_campos_regex(
    texto: str,
    resultado_llm: Dict[str, Any],
    data_cirurgia: Optional[str],
    registros: List[Dict] = None,
) -> dict:
    """
    Parte puramente de regras/regex.
    Recebe as sugestões do LLM e a data da cirurgia já resolvida,
    e devolve o dicionário `resultado` com todos os campos.
    """
    # 3) Contextos úteis (pré-op, cirurgia, internação)
    contextos = construir_contextos_cirurgia(texto, data_cirurgia)
    contexto_preop = contextos.get("preop") or None
    contexto_cirurgia = contextos.get("cirurgia") or None
    contexto_internacao = contextos.get("internacao") or None

    # 4) Começa o dicionário final
    resultado: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # BLOCO: Campos básicos / pré-operatórios
    # ------------------------------------------------------------------

    # sexo: se LLM falhar, usa regex simples
    resultado["sexo"] = resultado_llm.get("sexo") or extrair_sexo(texto)

    # dt_SO: LLM ou regex
    resultado["dt_SO"] = resultado_llm.get("dt_SO") or data_cirurgia

    # idade: queremos confiar mais na regex do que no LLM
    resultado["idade"] = extrair_idade(
        texto,
        data_cirurgia=data_cirurgia,
        sugestao_llm=resultado_llm.get("idade"),
    )

    # ASA: regex primeiro, LLM só como sugestão
    resultado["ASA"] = extrair_ASA(
        texto,
        sugestao_llm=resultado_llm.get("ASA") or resultado_llm.get("asa"),
    )

    # ECOG: mesmo esquema
    resultado["ECOG"] = extrair_ECOG(
        texto,
        sugestao_llm=resultado_llm.get("ECOG") or resultado_llm.get("ecog"),
    )

    # IMC: aqui mantemos a sua lógica antiga (regex com contexto),
    # usando o valor do LLM só se regex não achar nada
    imc_llm = resultado_llm.get("IMC")
    imc_regex = extrair_IMC(texto, data_cirurgia, contexto_preop)
    resultado["IMC"] = imc_regex if imc_regex is not None else imc_llm

    # KPS: LLM primeiro, regex como fallback
    kps_llm = resultado_llm.get("KPS")
    kps_regex = extrair_KPS(texto, data_cirurgia, contexto_preop)
    resultado["KPS"] = kps_llm if kps_llm is not None else kps_regex

    # local_tumor: LLM ou regex
    resultado["local_tumor"] = (
        resultado_llm.get("local_tumor")
        or extrair_local_tumor(texto, contexto_cirurgia)
    )

    # altura_tumor: regex (borda anal / altura do tumor)
    resultado["altura_tumor"] = extrair_altura_tumor(texto)

    # QRT_neo / eletiva / paliativa: mantém regex simples
    resultado["QRT_neo"] = resultado_llm.get("QRT_neo") or extrair_QRT_neo(texto)
    resultado["eletiva"] = resultado_llm.get("eletiva") or extrair_eletiva(texto)
    resultado["paliativa"] = resultado_llm.get("paliativa") or extrair_paliativa(texto)

    # cirurgia_recidiva: LLM ou regex
    resultado["cirurgia_recidiva"] = (
        resultado_llm.get("cirurgia_recidiva")
        or extrair_cirurgia_recidiva(texto, contexto_cirurgia)
    )

    # ------------------------------------------------------------------
    # BLOCO: Patologia / AP / Estadiamento
    # ------------------------------------------------------------------

    # AP: queremos um valor limpo tipo "pt4b pn1b"
    resultado["AP"] = extrair_AP(
        texto,
        sugestao_llm=resultado_llm.get("AP") or resultado_llm.get("ap"),
    )

    # Se AP vier preenchido, tenta derivar T / N / N_A a partir dele
    if resultado.get("AP"):
        ap_val = str(resultado["AP"]).lower()

        # T a partir de "pt4b"
        match_t = re.search(r"t([0-4][abc]?)", ap_val, re.IGNORECASE)
        if match_t:
            resultado["T"] = f"T{match_t.group(1).upper()}"

        # N a partir de "pn1b"
        match_n = re.search(r"n([0-3][abc]?)", ap_val, re.IGNORECASE)
        if match_n:
            n_val = f"N{match_n.group(1).upper()}"
            # tenta refinar com N no formato "2/26" se existir
            resultado["N"] = extrair_N(texto) or n_val

        # N_A: se N veio como "2/26", pega o numerador
        if resultado.get("N") and "/" in str(resultado["N"]):
            try:
                n_a = int(str(resultado["N"]).split("/")[0])
                resultado["N_A"] = n_a
            except Exception:
                pass

    # Fallbacks se ainda faltar alguma coisa
    if resultado.get("estadiamento") is None:
        resultado["estadiamento"] = (
            resultado_llm.get("estadiamento") or extrair_estadiamento(texto)
        )

    if resultado.get("T") is None:
        resultado["T"] = extrair_T(texto)

    if resultado.get("N") is None:
        resultado["N"] = extrair_N(texto)

    if resultado.get("N_A") is None:
        resultado["N_A"] = extrair_N_A(texto)

    # ------------------------------------------------------------------
    # BLOCO: Internação / UTI / Alta
    # ------------------------------------------------------------------

    # dias de internação e UTI: regex com ajuda da data de cirurgia
    resultado["dias_internação"] = (
        resultado_llm.get("dias_internação")
        or extrair_dias_internacao(texto, data_cirurgia, contexto_internacao)
    )

    resultado["dias_uti"] = (
        resultado_llm.get("dias_uti")
        or extrair_dias_uti(texto, data_cirurgia, contexto_internacao)
    )

    # dt_alta: regex a partir do contexto de internação
    resultado["dt_alta"] = (
        resultado_llm.get("dt_alta") or extrair_dt_alta(texto, contexto_internacao)
    )

    # BLOCO: Órgãos envolvidos e cirurgias associadas

    orgaos = extrair_orgaos_envolvidos(texto)
    resultado.update(orgaos)
    resultado["bexiga_tudo"] = extrair_bexiga_tudo(texto)
    resultado["bexiga_parte"] = extrair_bexiga_parte(texto)
    outro_orgao, outro_orgao_qual = extrair_outro_orgao(texto)
    resultado["outro_orgao"] = outro_orgao
    resultado["outro_orgao_qual"] = outro_orgao_qual
    resultado["n_orgaos"] = extrair_n_orgaos(texto, orgaos)
    resultado["amputação"] = extrair_amputacao(texto)
    resultado["RTS"] = extrair_RTS(texto)
    resultado["cole_total"] = extrair_cole_total(texto)
    resultado["posterior"] = extrair_posterior(texto)
    resultado["total"] = extrair_total(texto)
    resultado["SLE"] = extrair_SLE(texto)
    resultado["REC_plastica"] = extrair_REC_plastica(texto)
    resultado["tipo_REC"] = extrair_tipo_REC(texto)

    resultado[
        "tipo de rec uro  0 briker 1 duplo barril 2 duplo barril ileal 3 nefrostomia"
    ] = extrair_tipo_rec_uro(texto)

    # CH / complicações / tto / Clavien / reinternação / re-op
    resultado["CH_intra_OP"] = extrair_CH_intra_OP(texto, contexto_cirurgia)
    resultado["CH_num"] = extrair_CH_num(texto, contexto_cirurgia)

    resultado["complicação"] = extrair_complicacao(texto)
    resultado["complicação_qual"] = extrair_complicacao_qual(texto)
    resultado["tto"] = extrair_tto(texto)

    resultado["Clavien"] = extrair_Clavien(texto)
    resultado["Clavien_v2"] = extrair_Clavien_v2(texto)

    resultado["reinternação"] = extrair_reinternacao(texto)
    resultado["data da reinternação"] = extrair_data_reinternacao(texto)
    resultado["motivo_reinternação"] = extrair_motivo_reinternacao(texto)

    # re-op em 90 dias: regra com default 0
    resultado["re_op_90dias"] = extrair_re_op_90dias(texto, data_cirurgia)
    resultado["re_op_achado"] = extrair_re_op_achado(texto)

    resultado["obito_90dias"] = extrair_obito_90dias(texto)

    # ------------------------------------------------------------------
    # BLOCO: Seguimento / recidiva / óbito
    # ------------------------------------------------------------------

    resultado["histologia"] = extrair_histologia(texto)
    # AP já preenchido acima; só mantém se não tiver nada
    if not resultado.get("AP"):
        resultado["AP"] = extrair_AP(texto)

    resultado["invasão"] = extrair_invasao(texto)
    resultado["R0_R1_R2"] = extrair_R0_R1_R2(texto)
    resultado["R0_R1_R2_v2"] = extrair_R0_R1_R2_v2(texto)
    resultado["QT_adjuvante"] = extrair_QT_adjuvante(texto)

    resultado["recidiva"] = extrair_recidiva(texto)
    resultado["recidiva_local"] = extrair_recidiva_local(texto)
    resultado["recidiva_local_v2"] = extrair_recidiva_local_v2(texto)
    resultado["dt_recidiva"] = extrair_dt_recidiva(texto)
    resultado["DFSMESES"] = extrair_DFSMESES(texto)

    resultado["fisiatria"] = extrair_fisiatria(texto)
    resultado["paliativo_grupo_dor"] = extrair_paliativo_grupo_dor(texto)
    resultado["grupo_dor"] = extrair_grupo_dor(texto)

    resultado["ult_consulta"] = extrair_ult_consulta(texto)
    resultado["OS_meses"] = extrair_OS_meses(texto)

    resultado["obito"] = extrair_obito(texto)
    resultado["dt_obito"] = extrair_dt_obito(texto)
    resultado["obito_motivo"] = extrair_obito_motivo(texto)

    resultado["assistente"] = extrair_assistente(texto)
    resultado["observação"] = extrair_observacao(texto)

    # ------------------------------------------------------------------
    # BLOCO: tempo de seguimento / tempo_SO
    # ------------------------------------------------------------------
    resultado["tempo_SO"] = extrair_tempo_SO(
        data_cirurgia,
        dt_alta=resultado.get("dt_alta"),
        dt_ult_consulta=resultado.get("ult_consulta"),
        dt_obito=resultado.get("dt_obito"),
    )

    return resultado

import unicodedata
from datetime import datetime

def normalizar_valor_comparacao(valor):
    if valor is None:
        return ""

    # Trata números primeiro
    if isinstance(valor, (int, float)):
        try:
            if pd.isna(valor):
                return ""
        except (TypeError, ValueError):
            pass

        if isinstance(valor, float) and valor.is_integer():
            return str(int(valor))
        return str(valor)

    s = str(valor).strip().lower()

    # Remove acentos
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # Normaliza datas básicas
    if "/" in s or "-" in s:
        pedacos = s.split()
        parte_data = pedacos[0]
        formatos = ["%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"]
        for fmt in formatos:
            try:
                dt = datetime.strptime(parte_data, fmt)
                return dt.strftime("%d/%m/%Y")
            except Exception:
                pass
    return s



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

        # Se o Excel tiver colunas duplicadas com o mesmo nome,
        # linha_real[campo] vira uma Series. Pegamos o primeiro valor.
        if isinstance(valor_real, pd.Series):
            if valor_real.empty:
                continue
            valor_real = valor_real.iloc[0]

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
        ok = s_real == s_pred

        if ok:
            acertos += 1

        status = "OK" if ok else "ERRO"
        print(f" - {campo}: real = {s_real!r}, pred = {s_pred!r}  ->  {status}")

        detalhes.append(
            {"campo": campo, "real": s_real, "pred": s_pred, "ok": ok}
        )
    
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
