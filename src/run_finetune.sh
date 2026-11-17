#!/bin/bash
# Wait for the token cache, then federated partial fine-tuning for seeds 0-4 (sequential).
cd ${FEDPCT_ROOT}
export HF_HOME=${FEDPCT_ROOT}/.hf PYTHONPATH=src
until grep -q CACHE_DONE logs/cache_tokens.log; do
  pgrep -f cache_tokens.py > /dev/null || { echo "cache_tokens stopped without CACHE_DONE"; exit 1; }
  sleep 20
done
for s in 0 1 2 3 4; do
  python src/finetune_fed.py $s > logs/finetune_seed$s.log 2>&1 || { echo "seed $s failed"; exit 1; }
done
echo ALL_FT_DONE
