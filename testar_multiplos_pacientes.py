"""Testa em múltiplos pacientes diferentes para evitar overfitting"""

from ner_llm import carregar_gabarito, carregar_texto_com_contexto, extrair_campos_por_regras, comparar_com_gabarito
from pathlib import Path
import pandas as pd
import random

def testar_multiplos_pacientes():
    df = carregar_gabarito()
    
    # Usa a pasta data onde estão os dados reais
    from ner_llm import DATA_DIR
    pasta_dados = DATA_DIR
    
    if not pasta_dados.exists():
        print(f"Pasta não encontrada: {pasta_dados}")
        return
    
    pacientes = sorted([p.name for p in pasta_dados.iterdir() if p.is_dir()])
    
    # Filtra apenas pacientes que existem no gabarito
    pacientes_validos = []
    for p in pacientes:
        nome_gabarito = p.replace("_", " ")
        if len(df[df['Nome'] == nome_gabarito]) > 0:
            pacientes_validos.append(p)
    
    # Testa em 5 pacientes DIFERENTES (aleatórios para evitar overfitting)
    pacientes_teste = random.sample(pacientes_validos, min(5, len(pacientes_validos)))
    
    print(f"Testando {len(pacientes_teste)} pacientes DIFERENTES para evitar overfitting...\n")
    print(f"Pacientes: {', '.join(pacientes_teste)}\n")
    
    total_campos = 0
    total_acertos = 0
    acuracias = []
    detalhes_por_paciente = {}
    
    for i, paciente_slug in enumerate(pacientes_teste, 1):
        try:
            nome_paciente = paciente_slug.replace("_", " ")
            
            # Usa a função do ner_llm que já funciona
            texto, registros = carregar_texto_com_contexto(paciente_slug)
            
            if not texto or len(texto) < 100:
                print(f"[{i}/{len(pacientes_teste)}] {paciente_slug}: Texto muito curto ({len(texto)} chars)")
                continue
            
            print(f"[{i}/{len(pacientes_teste)}] {paciente_slug}: Processando... (texto: {len(texto)} chars)")
            
            pred = extrair_campos_por_regras(texto)
            
            todas_colunas = [col for col in df.columns 
                            if col not in ["Idnum", "valido", "Nome", "nan"] and pd.notna(col)]
            
            stats = comparar_com_gabarito(paciente_slug, pred, df, todas_colunas)
            
            if "acuracia" in stats:
                acc = stats["acuracia"]
                acertos = stats.get("acertos", 0)
                total = stats.get("total", 0)
                total_campos += total
                total_acertos += acertos
                acuracias.append(acc)
                detalhes_por_paciente[paciente_slug] = {
                    "acuracia": acc,
                    "acertos": acertos,
                    "total": total,
                    "detalhes": stats.get("detalhes", [])
                }
                
                print(f"[{i}/{len(pacientes_teste)}] {paciente_slug}: {acc:.1%} ({acertos}/{total})")
        except Exception as e:
            print(f"[{i}/{len(pacientes_teste)}] {paciente_slug}: ERRO - {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if total_campos > 0:
        acuracia_geral = total_acertos / total_campos
        print(f"\n{'='*60}")
        print(f"ACURÁCIA GERAL (múltiplos pacientes): {acuracia_geral:.1%} ({total_acertos}/{total_campos})")
        print(f"Média por paciente: {sum(acuracias)/len(acuracias):.1%}" if acuracias else "")
        if len(acuracias) > 1:
            print(f"Desvio padrão: {pd.Series(acuracias).std():.1%}")
            print(f"Min: {min(acuracias):.1%}, Max: {max(acuracias):.1%}")
        print(f"{'='*60}")
        
        # Mostra detalhes dos campos que mais erram
        print("\nCAMPOS QUE MAIS ERRAM (acima de 50% de erro):")
        erros_por_campo = {}
        for paciente, dados in detalhes_por_paciente.items():
            for detalhe in dados.get("detalhes", []):
                campo = detalhe.get("campo", "")
                if not campo:
                    continue
                if campo not in erros_por_campo:
                    erros_por_campo[campo] = {"erros": 0, "total": 0}
                erros_por_campo[campo]["total"] += 1
                if detalhe.get("status") == "ERRO":
                    erros_por_campo[campo]["erros"] += 1
        
        campos_problematicos = []
        for campo, stats in erros_por_campo.items():
            taxa_erro = stats["erros"] / stats["total"] if stats["total"] > 0 else 0
            if taxa_erro > 0.5:
                campos_problematicos.append((campo, taxa_erro, stats["erros"], stats["total"]))
        
        campos_problematicos.sort(key=lambda x: x[1], reverse=True)
        for campo, taxa, erros, total in campos_problematicos[:15]:
            print(f"  {campo:30s} | Taxa erro: {taxa:.1%} | {erros}/{total}")

if __name__ == "__main__":
    testar_multiplos_pacientes()
