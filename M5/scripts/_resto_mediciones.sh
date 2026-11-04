#!/usr/bin/env bash
# Encadena lo que falta medir después del bench de CPU. Uso: bash scripts/_resto_mediciones.sh
set -u
cd "$(dirname "$0")/.."
P="python -X utf8 -u"

echo "### 1/5 ablación del filtro de salida"
$P scripts/ablacion_filtro.py resultados/eficiencia_cpu_crudo.json 2>&1 | grep -vE "it/s\]|Loading weights"

echo; echo "### 2/5 scorecard de M5 junto a M3 y M4"
$P scripts/scorecard_m5.py 2>&1 | grep -vE "it/s\]|Loading weights"

echo; echo "### 3/5 cuantización en GPU: fp16 (el sistema de M4) frente a NF4 del lab S15"
$P scripts/bench_eficiencia.py --variantes fp16_gpu,int4_gpu --eval-set eval_set_m3.json \
   --etiqueta gpu --base fp16_gpu 2>&1 | grep -vE "it/s\]|Loading weights"

echo; echo "### 4/5 ablación del planificador ReAct"
$P scripts/bench_ablacion.py --backend fp16_gpu 2>&1 | grep -vE "it/s\]|Loading weights"

echo; echo "### 5/5 W&B (offline si no hay clave)"
$P scripts/wandb_publicar.py 2>&1 | tail -20

echo; echo "### TODO TERMINADO"
