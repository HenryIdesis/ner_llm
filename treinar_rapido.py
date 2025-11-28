"""Treinamento rápido para avaliar acurácia geral"""

from ner_llm import carregar_gabarito, carregar_texto_com_contexto, extrair_campos_por_regras, comparar_com_gabarito, DATA_DIR
from pathlib import Path
import pandas as pd

def treinar_rapido():
    df = carregar_gabarito()
    pasta_treino = Path("dataset/treino/treino_validacao")
    
    if not pasta_treino.exists():
        print(f"Pasta não encontrada: {pasta_treino}")
        return
    
    pacientes = sorted([p.name for p in pasta_treino.iterdir() if p.is_dir()])
    pacientes = pacientes[:10]  # Testa apenas 10 para ser rápido
    
    print(f"Testando {len(pacientes)} pacientes...\n")
    
    total_campos = 0
    total_acertos = 0
    acuracias = []
    
    for i, paciente_slug in enumerate(pacientes, 1):
        try:
            nome_paciente = paciente_slug.replace("_", " ")
            # Usa pasta_treino como base_dir
            from ner_llm import carregar_jsonl_paciente
            pasta_paciente = pasta_treino / paciente_slug
            if not pasta_paciente.exists():
                print(f"[{i}/{len(pacientes)}] {paciente_slug}: Pasta não encontrada")
                continue
            
            # Carrega texto manualmente
            registros = []
            for arquivo in sorted(pasta_paciente.glob("*.jsonl")):
                import json
                with open(arquivo, "r", encoding="utf-8") as f:
                    for linha in f:
                        linha = linha.strip()
                        if linha:
                            try:
                                registros.append(json.loads(linha))
                            except:
                                continue
            
            texto = "\n".join([r.get("texto", "") for r in registros])
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
                print(f"[{i}/{len(pacientes)}] {paciente_slug}: {acc:.1%} ({acertos}/{total})")
        except Exception as e:
            print(f"[{i}/{len(pacientes)}] {paciente_slug}: ERRO - {e}")
            continue
    
    if total_campos > 0:
        acuracia_geral = total_acertos / total_campos
        print(f"\n{'='*60}")
        print(f"ACURÁCIA GERAL: {acuracia_geral:.1%} ({total_acertos}/{total_campos})")
        print(f"Média por paciente: {sum(acuracias)/len(acuracias):.1%}" if acuracias else "")
        print(f"{'='*60}")

if __name__ == "__main__":
    treinar_rapido()

