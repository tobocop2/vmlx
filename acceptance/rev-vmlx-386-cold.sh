#!/bin/bash
# Review arm for jjang-ai/vmlx#386: a repo-id launch with a COLD Hugging Face cache (the engine
# downloads the model during launch) against the same launch with a WARM cache (control).
set -u
MODEL="mlx-community/Qwen3-0.6B-4bit"
PORT="${PORT:-18386}"
WORK=$(mktemp -d "${WORK_ROOT:-${TMPDIR:-/tmp}}/cold386.XXXXXX")
WARM_HF="${HF_HOME:-$HOME/.cache/huggingface}"
python - > "$WORK/body.json" <<'PY'
import json
print(json.dumps({"model": "mlx-community/Qwen3-0.6B-4bit", "max_tokens": 4,
                  "messages": [{"role": "user", "content": "w " * 45000}]}))
PY
FAILS=0
arm() {
  label="$1"; hf="$2"
  log="$WORK/server-$label.log"
  HF_HOME="$hf" vmlx serve "$MODEL" --port "$PORT" > "$log" 2>&1 &
  pid=$!
  i=0; up=0
  while [ $i -lt 180 ]; do
    if curl -s -m 3 -o /dev/null "http://127.0.0.1:$PORT/v1/capabilities"; then up=1; break; fi
    kill -0 $pid 2>/dev/null || break
    sleep 5; i=$((i+1))
  done
  cap=none
  if [ $up = 1 ]; then
    cap=$(curl -s -m 20 "http://127.0.0.1:$PORT/v1/capabilities" | python -c "import sys,json; print(json.load(sys.stdin).get('max_prompt_tokens'))")
    t0=$(date +%s)
    code=$(curl -s -m 90 -o "$WORK/resp-$label.json" -w '%{http_code}' -H 'Content-Type: application/json' -d @"$WORK/body.json" "http://127.0.0.1:$PORT/v1/chat/completions")
    el=$(( $(date +%s) - t0 ))
    errcode=$(python -c "import json;print(json.load(open('$WORK/resp-$label.json'))['error']['code'])" 2>/dev/null || echo none)
  else
    code=NOSERVER; el=0; errcode=none
  fi
  dec=$(grep -c 'Declared model context ceiling' "$log"); ref=$(grep -c 'Refreshed max prompt/context length' "$log")
  ctl=$(grep -c 'Loading model' "$log")
  dl=$(grep -c 'Fetching [0-9]* files' "$log")
  echo "RESULT cell=COLD386-$label hf_home_is_cold=$([ "$label" = cold ] && echo yes || echo no) server_up=$up polls=$i download_lines=$dl control_loading_line_count=$ctl capabilities.max_prompt_tokens=$cap declared_line_count=$dec refreshed_line_count=$ref over_context_chat_45000_words http=$code elapsed_s=$el error_code=$errcode"
  kill $pid 2>/dev/null; sleep 3; kill -9 $pid 2>/dev/null
}
arm cold "$WORK/cold-hf"
arm warm "$WARM_HF"
echo "SUMMARY done work=$WORK"
