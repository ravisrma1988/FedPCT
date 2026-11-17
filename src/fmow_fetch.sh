#!/bin/bash
# Parallel ranged download of FMoW-WILDS v1.1 (WILDS release, Hugging Face mirror named in the WILDS code).
# Each part is resumable: rerunning the script continues incomplete parts. TLS verification stays on.
set -u
URL="https://huggingface.co/datasets/pangwei/wilds-fmow/resolve/401e7c233898df2005ad4c680720407912f30854/fmow_v1.1.tar.gz"
SIZE=53893501163
N=${1:-16}
DIR=${FEDPCT_ROOT}/data/fmow_download
cd "$DIR" || exit 1
rm -f fmow_v1.1.tar.gz            # partial single-stream file from the first attempt
CH=$(( (SIZE + N - 1) / N ))
for i in $(seq 0 $((N - 1))); do
  S=$(( i * CH )); E=$(( S + CH - 1 )); [ $E -ge $SIZE ] && E=$((SIZE - 1))
  WANT=$(( E - S + 1 ))
  P=$(printf "part.%02d" "$i")
  (
    for try in 1 2 3 4 5 6 7 8 9 10; do
      HAVE=$(stat -c %s "$P" 2>/dev/null || echo 0)
      [ "$HAVE" -ge "$WANT" ] && break
      curl -sSL --retry 5 -r $(( S + HAVE ))-$E "$URL" >> "$P"
    done
    echo "$P $(stat -c %s "$P") / $WANT"
  ) &
done
wait
TOTAL=$(cat part.* | wc -c)
echo "downloaded $TOTAL of $SIZE bytes"
if [ "$TOTAL" -eq "$SIZE" ]; then
  cat $(ls part.* | sort) > fmow_v1.1.tar.gz && rm part.* && echo FMOW_DOWNLOAD_DONE
fi
