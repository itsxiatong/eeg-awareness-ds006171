#!/bin/bash
# 依次预处理所有被试（已完成的会跳过）
cd "$(dirname "$0")"
for d in ../ds006171/sub-*; do
  s=$(basename "$d")
  [ -f "../derivatives/${s}_qc.csv" ] && [ "../derivatives/${s}_qc.csv" -nt 01_preprocess.py ] && continue
  echo "== $s"; python3 01_preprocess.py "$s" 2>&1 | grep -E "bad|kept|DONE|Error"
done
