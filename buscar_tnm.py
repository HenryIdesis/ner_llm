"""Busca TNM diretamente no texto para entender o formato"""

from ner_llm import carregar_texto_com_contexto
import re

texto, _ = carregar_texto_com_contexto('Paciente_0000001')

# Busca "produto de exenteração"
match_ap = list(re.finditer(r'produto\s+de\s+exentera', texto, re.IGNORECASE))
print(f"Encontrados {len(match_ap)} laudos de AP")

if match_ap:
    match = match_ap[0]
    ctx = texto[match.start():match.end()+150000]
    
    # Busca vários padrões de TNM
    padroes = [
        r'[pt]?[0-4][a-cb]?\s+[pn]?[0-3][a-cb]?',
        r'pT\d+[a-cb]?\s+pN\d+[a-cb]?',
        r'T\d+[a-cb]?\s+N\d+[a-cb]?',
        r'estadiamento[^\n]*[pt]?[0-4]',
    ]
    
    print("\nBuscando TNM...")
    for i, padrao in enumerate(padroes):
        matches = re.findall(padrao, ctx, re.IGNORECASE)
        if matches:
            print(f"Padrão {i+1} encontrou: {matches[:5]}")
    
    # Busca "anatomia patológica" no contexto
    idx_ap = ctx.lower().find('anatomia patologica')
    if idx_ap > 0:
        print(f"\n'Anatomia patológica' encontrado na posição {idx_ap}")
        print("Contexto ao redor:")
        print(ctx[max(0, idx_ap-500):idx_ap+5000])


