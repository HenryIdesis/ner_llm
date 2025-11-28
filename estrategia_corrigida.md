# ESTRATÉGIA CORRIGIDA - Análise Profunda

## Problemas Identificados:

1. **AP não está sendo incluído no texto relevante** - mesmo sendo encontrado na posição 2291353
2. **TNM não está sendo encontrado no contexto do AP** - padrão regex pode estar errado
3. **Dias de internação não estão sendo encontrados** - padrões não estão capturando
4. **LLM não está extraindo campos críticos** - mesmo com contexto, não está funcionando

## Solução Real:

**ABORDAGEM EM DUAS ETAPAS:**

1. **ETAPA 1: Localização com Regex** - Usar regex para encontrar ONDE está cada informação:
   - AP: encontrar laudo de AP e extrair TNM diretamente com regex
   - Dias internação: encontrar padrões de "dias de internação" próximo da cirurgia
   - Dias UTI: encontrar padrões de "dias UTI" próximo da cirurgia
   - Outros campos: usar regex primeiro, LLM só se regex falhar

2. **ETAPA 2: LLM apenas para campos complexos** - Usar LLM apenas para:
   - Campos que realmente precisam de contexto (idade calculada, cirurgia_recidiva)
   - Campos onde regex falhou
   - Passar APENAS o contexto específico necessário, não todo o prontuário

## Implementação:

- Melhorar regex de AP para capturar TNM corretamente
- Melhorar regex de dias internação/UTI
- Usar LLM apenas como fallback, não como método principal
- Passar contexto específico para LLM, não todo o prontuário

