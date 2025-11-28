"""Diagnóstico profundo: entender POR QUE o modelo não está funcionando"""

from ner_llm import carregar_texto_com_contexto, extrair_data_cirurgia, extrair_secoes_relevantes
import re

def diagnosticar_paciente(paciente_slug):
    print(f"\n{'='*80}")
    print(f"DIAGNÓSTICO: {paciente_slug}")
    print(f"{'='*80}\n")
    
    # 1. Carrega texto
    texto, _ = carregar_texto_com_contexto(paciente_slug)
    print(f"1. TAMANHO DO TEXTO: {len(texto):,} caracteres")
    
    # 2. Extrai data cirurgia
    data_cirurgia = extrair_data_cirurgia(texto)
    print(f"2. DATA CIRURGIA ENCONTRADA: {data_cirurgia}")
    
    if not data_cirurgia:
        print("   ERRO: Não encontrou data da cirurgia!")
        return
    
    # 3. Verifica ocorrências da data (tenta diferentes formatos)
    formatos_data = [
        data_cirurgia.replace("/", "/"),
        data_cirurgia.replace("/", "-"),
        data_cirurgia.replace("/", "."),
    ]
    matches_data = []
    for formato in formatos_data:
        padrao_data = re.escape(formato)
        matches = list(re.finditer(padrao_data, texto, re.IGNORECASE))
        matches_data.extend(matches)
    print(f"3. OCORRENCIAS DA DATA: {len(matches_data)}")
    if matches_data:
        print(f"   Primeira ocorrencia na posicao: {matches_data[0].start()}")
        contexto = texto[max(0, matches_data[0].start()-100):matches_data[0].end()+100]
        print(f"   Contexto: ...{contexto}...")
    
    # 4. Verifica se encontra AP próximo da data
    padrao_ap = r"(?:produto\s+de\s+exentera|anatomia\s+patologica|laudo\s+anatomopatologico)"
    matches_ap = list(re.finditer(padrao_ap, texto, re.IGNORECASE))
    print(f"4. LAUDOS DE AP ENCONTRADOS: {len(matches_ap)}")
    
    # Verifica se algum AP está próximo da data da cirurgia
    ap_proximo = False
    for match_ap in matches_ap[:5]:
        contexto_ap = texto[max(0, match_ap.start()-2000):match_ap.end()+5000]
        if data_cirurgia[:7] in contexto_ap or data_cirurgia.split("/")[2] in contexto_ap:
            ap_proximo = True
            print(f"   [OK] AP proximo da cirurgia encontrado (posicao {match_ap.start()})")
            # Procura TNM neste contexto
            tnm_match = re.search(r"p?t([0-4][a-cb]?)\s+p?n([0-3][a-cb]?)\s*(?:p?m([0-1][a-cb]?))?", contexto_ap, re.IGNORECASE)
            if tnm_match:
                print(f"   [OK] TNM encontrado: {tnm_match.group(0)}")
            else:
                print(f"   [ERRO] TNM NAO encontrado no contexto do AP")
            break
    
    if not ap_proximo:
        print(f"   [ERRO] NENHUM AP proximo da cirurgia encontrado!")
    
    # 5. Verifica dias de internação
    padroes_internacao = [
        r"dias\s+de\s+internac[aao]+[^\d]*(\d+)",
        r"permanencia[^\d]*(\d+)\s*dias",
        r"internacao[^\d]*(\d+)\s*dias",
    ]
    dias_encontrados = []
    for padrao in padroes_internacao:
        matches = list(re.finditer(padrao, texto, re.IGNORECASE))
        for match in matches:
            contexto = texto[max(0, match.start()-1000):match.end()+1000]
            if data_cirurgia[:7] in contexto or data_cirurgia.split("/")[2] in contexto:
                dias_encontrados.append((match.group(1), match.start()))
    
    print(f"5. DIAS DE INTERNAÇÃO ENCONTRADOS: {len(dias_encontrados)}")
    if dias_encontrados:
        for dias, pos in dias_encontrados[:3]:
            print(f"   - {dias} dias (posição {pos})")
    else:
        print(f"   [ERRO] NENHUM dia de internacao encontrado proximo da cirurgia!")
    
    # 6. Verifica dias UTI
    padroes_uti = [
        r"dias\s+uti[^\d]*(\d+)",
        r"permanencia\s+uti[^\d]*(\d+)",
        r"uti[^\d]*(\d+)\s*dias",
    ]
    uti_encontrados = []
    for padrao in padroes_uti:
        matches = list(re.finditer(padrao, texto, re.IGNORECASE))
        for match in matches:
            contexto = texto[max(0, match.start()-1000):match.end()+1000]
            if data_cirurgia[:7] in contexto or data_cirurgia.split("/")[2] in contexto:
                uti_encontrados.append((match.group(1), match.start()))
    
    print(f"6. DIAS UTI ENCONTRADOS: {len(uti_encontrados)}")
    if uti_encontrados:
        for uti, pos in uti_encontrados[:3]:
            print(f"   - {uti} dias (posição {pos})")
    else:
        print(f"   [ERRO] NENHUM dia de UTI encontrado proximo da cirurgia!")
    
    # 7. Verifica o que está sendo passado para o LLM
    texto_relevante = extrair_secoes_relevantes(texto, data_cirurgia)
    print(f"\n7. TEXTO RELEVANTE PARA LLM: {len(texto_relevante):,} caracteres")
    print(f"   Redução: {len(texto)/len(texto_relevante):.1f}x")
    
    # Verifica se as informações críticas estão no texto relevante
    if ap_proximo:
        # Verifica se o AP está no texto relevante
        if "produto de exentera" in texto_relevante.lower() or "anatomia patologica" in texto_relevante.lower():
            print(f"   [OK] AP esta no texto relevante")
        else:
            print(f"   [ERRO] AP NAO esta no texto relevante!")
    
    if dias_encontrados:
        if any(f"{dias} dias" in texto_relevante for dias, _ in dias_encontrados[:3]):
            print(f"   [OK] Dias internacao estao no texto relevante")
        else:
            print(f"   [ERRO] Dias internacao NAO estao no texto relevante!")
    
    if uti_encontrados:
        if any(f"{uti} dias" in texto_relevante or "uti" in texto_relevante.lower() for uti, _ in uti_encontrados[:3]):
            print(f"   [OK] Dias UTI estao no texto relevante")
        else:
            print(f"   [ERRO] Dias UTI NAO estao no texto relevante!")

if __name__ == "__main__":
    diagnosticar_paciente("Paciente_0000001")

