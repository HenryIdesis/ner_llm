"""
Gera relatório detalhado de acurácia por coluna/campo.

Uso:
    python analise_erros_colunas.py [limite_pacientes]

Se nenhum limite for informado, analisa todos os pacientes disponíveis em
`data/`. O objetivo é identificar rapidamente quais colunas estão com maior
taxa de erro para priorizar novos experimentos ou treino focado.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ner_llm import (
    carregar_gabarito,
    carregar_texto_com_contexto,
    extrair_campos_por_regras,
    normalizar_valor_comparacao,
)

IGNORED_COLS = {"Idnum", "valido", "Nome", "nan"}
DATA_DIR = Path("data")


def obter_colunas(df: pd.DataFrame) -> List[str]:
    colunas = []
    for col in df.columns:
        if pd.isna(col):
            continue
        if col in IGNORED_COLS:
            continue
        colunas.append(str(col))
    return colunas


def comparar_paciente(
    paciente_slug: str,
    pred: Dict,
    df: pd.DataFrame,
    colunas: List[str],
) -> List[Dict]:
    nome = paciente_slug.replace("_", " ")
    linha = df.loc[df["Nome"] == nome]

    if linha.empty:
        return []

    linha = linha.iloc[0]
    resultados = []

    for campo in colunas:
        if campo not in linha.index:
            continue

        valor_real = linha[campo]

        # Se tiver colunas duplicadas com o mesmo nome, isso vira uma Series.
        # Pegamos o primeiro valor não-nulo.
        if isinstance(valor_real, pd.Series):
            if valor_real.empty:
                continue
            valor_real = valor_real.dropna()
            if valor_real.empty:
                continue
            valor_real = valor_real.iloc[0]

        # Ignora valores NaN / vazios
        try:
            if pd.isna(valor_real):
                continue
        except (TypeError, ValueError):
            pass

        real_norm = normalizar_valor_comparacao(valor_real)
        if real_norm in {"", "nan", "none", "nat", "na"}:
            continue

        pred_norm = normalizar_valor_comparacao(pred.get(campo))
        ok = (real_norm == pred_norm)

        resultados.append(
            {
                "campo": campo,
                "real": real_norm,
                "pred": pred_norm,
                "ok": ok,
            }
        )

    return resultados



def gerar_relatorio(pacientes: List[str], limite: int | None = None):
    df = carregar_gabarito()
    colunas = obter_colunas(df)

    per_coluna = defaultdict(
        lambda: {"total": 0, "acertos": 0, "erros": []}
    )
    per_paciente = []

    lista = pacientes[:limite] if limite else pacientes
    for idx, slug in enumerate(lista, 1):
        texto, _ = carregar_texto_com_contexto(slug)
        pred = extrair_campos_por_regras(texto)
        resultados = comparar_paciente(slug, pred, df, colunas)
        if not resultados:
            continue

        acertos = sum(1 for r in resultados if r["ok"])
        total = len(resultados)
        per_paciente.append(
            {"paciente": slug, "acuracia": acertos / total, "total": total}
        )

        for r in resultados:
            dados = per_coluna[r["campo"]]
            dados["total"] += 1
            if r["ok"]:
                dados["acertos"] += 1
            else:
                if len(dados["erros"]) < 5:  # mantém amostras curtas
                    dados["erros"].append(
                        {
                            "paciente": slug,
                            "real": r["real"],
                            "pred": r["pred"],
                        }
                    )

    relatorio = []
    for campo, info in per_coluna.items():
        if info["total"] == 0:
            continue
        acc = info["acertos"] / info["total"]
        relatorio.append(
            {
                "campo": campo,
                "total": info["total"],
                "acuracia": acc,
                "erros": info["erros"],
            }
        )

    relatorio.sort(key=lambda x: x["acuracia"])
    return relatorio, per_paciente


def main():
    parser = argparse.ArgumentParser(
        description="Analisa acurácia por coluna para priorizar melhorias."
    )
    parser.add_argument(
        "--limite",
        type=int,
        default=None,
        help="Número de pacientes a serem avaliados (default: todos).",
    )
    args = parser.parse_args()

    pacientes = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())
    relatorio, per_paciente = gerar_relatorio(pacientes, args.limite)

    print("\n=== TOP CAMPOS COM MAIOR ERRO ===")
    for entrada in relatorio[:15]:
        print(
            f"{entrada['campo']:25s} "
            f"- acc={entrada['acuracia']:.1%} ({entrada['total']} casos)"
        )
        for erro in entrada["erros"]:
            print(
                f"    • {erro['paciente']}: real='{erro['real']}' "
                f"x pred='{erro['pred']}'"
            )
        if not entrada["erros"]:
            print("    • Sem amostras de erro (todos corretos).")

    if per_paciente:
        media = sum(p["acuracia"] for p in per_paciente) / len(per_paciente)
        print("\n=== RESUMO POR PACIENTE ===")
        for info in per_paciente:
            print(
                f"{info['paciente']:20s} "
                f"- acc={info['acuracia']:.1%} ({info['total']} campos)"
            )
        print(f"\nMédia por paciente: {media:.1%}")


if __name__ == "__main__":
    main()

