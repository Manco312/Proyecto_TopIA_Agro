#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
P="python -X utf8 -u"
echo "### B · ablacion del filtro de salida (todas las variantes CPU)"
$P scripts/ablacion_filtro.py resultados/eficiencia_cpu_crudo.json 2>&1 | grep -vE "it/s\]|Loading weights"
echo; echo "### C · scorecard M5 vs M3 vs M4"
$P scripts/puntuar.py resultados/eficiencia_cpu_crudo.json --sin-juez 2>&1 | grep -E "^\| |Escrito"
$P scripts/scorecard_m5.py --variante fp32_cpu 2>&1 | grep -vE "it/s\]|Loading weights"
echo; echo "### D · plan B"
$P scripts/plan_b.py --variante fp32_cpu 2>&1 | tail -2
echo; echo "### TERMINADO"
