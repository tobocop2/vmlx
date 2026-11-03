#!/bin/bash
# Acceptance script for jjang-ai/vmlx#385: a chat prompt longer than the model's declared
# context must be answered with the server's existing HTTP 413 "prompt_too_long" response.
#
# Runs on a macos-15 runner with the package under test and mlx-lm 0.31.3 installed and the
# model prefetched. Every wait is bounded (curl -m, readiness polls with a count, one global
# deadline). Each RESULT row is decided by an oracle that is not the server: the model's
# tokenizer counts the rendered prompt, and the declared context is read from the model's
# own config.json.
#
# Env: ONLY (comma list of cell ids, default all), WAIT_HANG, WAIT_FIT, TOTAL_BUDGET, PORT.
set -u

MODEL="${MODEL:-mlx-community/Qwen3-0.6B-4bit}"
PORT="${PORT:-18385}"
BASE_URL="http://127.0.0.1:${PORT}"
WAIT_HANG="${WAIT_HANG:-120}"
WAIT_FIT="${WAIT_FIT:-900}"
READY_POLLS="${READY_POLLS:-120}"
TOTAL_BUDGET="${TOTAL_BUDGET:-5400}"
ONLY="${ONLY:-all}"
START_TS=$(date +%s)
DEADLINE=$((START_TS + TOTAL_BUDGET))
WORK=$(mktemp -d "${WORK_ROOT:-${TMPDIR:-/tmp}}/vmlx385.XXXXXX")
SERVER_PID=""
FAILS=0
ROWS=0
OPEN=0
TIMEOUTS=0

say() { printf '%s\n' "$*"; }

PYHELPER="$WORK/oracle.py"
cat > "$PYHELPER" <<'PY'
import glob, json, os, sys

MODEL = os.environ["MODEL"]


def snapshot():
    from huggingface_hub import snapshot_download
    return snapshot_download(MODEL, local_files_only=True)


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(snapshot())


def declared():
    with open(os.path.join(snapshot(), "config.json")) as f:
        cfg = json.load(f)
    return int(cfg["max_position_embeddings"])


def chat_count(tok, n):
    text = tok.apply_chat_template(
        [{"role": "user", "content": "w " * n}], tokenize=False, add_generation_prompt=True
    )
    return len(tok.encode(text, add_special_tokens=False))


def raw_count(tok, n):
    return len(tok.encode("w " * n, add_special_tokens=False))


def solve(count, tok, target):
    n = max(1, target - (count(tok, 1) - 1))
    for _ in range(64):
        c = count(tok, n)
        if c == target:
            return n, c
        n += target - c
    return n, count(tok, n)


def cmd_declared():
    print(declared())


def cmd_snapshot():
    print(snapshot())


def cmd_body():
    # body <route> <target_tokens> <max_tokens> <stream 0|1> <outfile>
    route, target, max_tokens, stream, out = sys.argv[2:7]
    target, max_tokens, stream = int(target), int(max_tokens), stream == "1"
    tok = tokenizer()
    if route == "chat":
        n, c = solve(chat_count, tok, target)
        body = {"model": MODEL, "messages": [{"role": "user", "content": "w " * n}],
                "max_tokens": max_tokens, "stream": stream}
    else:
        n, c = solve(raw_count, tok, target)
        body = {"model": MODEL, "prompt": "w " * n, "max_tokens": max_tokens, "stream": stream}
    with open(out, "w") as f:
        json.dump(body, f)
    print(c)


def cmd_expect():
    # expect <prompt_tokens> <max_tokens> <explicit_cap>  -> reject|ok
    p, mt, cap = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    d = declared()
    if cap > 0 and p > cap:
        print("reject")
    elif mt > 0 and p >= d:
        print("reject")
    else:
        print("ok")


def cmd_classify():
    # classify <bodyfile> <stream 0|1>  -> "<form> <code> <completion_tokens> <prompt_tokens> <finish>"
    path, stream = sys.argv[2], sys.argv[3] == "1"
    try:
        raw = open(path, errors="replace").read()
    except OSError:
        raw = ""
    form, code, ctoks, ptoks, finish = "none", "-", "-", "-", "-"
    if not stream:
        try:
            doc = json.loads(raw)
        except ValueError:
            doc = None
        if isinstance(doc, dict):
            if "error" in doc and isinstance(doc["error"], dict):
                form, code = "http_json_error", str(doc["error"].get("code") or "-")
            elif "choices" in doc:
                form = "completion"
                u = doc.get("usage") or {}
                ctoks = str(u.get("completion_tokens", "-"))
                ptoks = str(u.get("prompt_tokens", "-"))
                ch = doc["choices"][0] if doc["choices"] else {}
                finish = str(ch.get("finish_reason") or "-")
    else:
        saw_chunk = False
        for line in raw.splitlines():
            if not line.startswith("data: ") or line.startswith("data: [DONE]"):
                continue
            try:
                d = json.loads(line[6:])
            except ValueError:
                continue
            if isinstance(d, dict) and d.get("error"):
                form, code = "sse_error_frame", str(d["error"].get("code") or "-")
                break
            if isinstance(d, dict) and d.get("choices"):
                saw_chunk = True
                fr = d["choices"][0].get("finish_reason")
                if fr:
                    finish = str(fr)
            if isinstance(d, dict) and isinstance(d.get("usage"), dict):
                ptoks = str(d["usage"].get("prompt_tokens", ptoks))
                ctoks = str(d["usage"].get("completion_tokens", ctoks))
        if form == "none" and saw_chunk:
            form = "sse_stream"
    print(form, code, ctoks, ptoks, finish)


def cmd_msgcount():
    # msgcount <bodyfile>: token count the 413 message names, or "-"
    import re
    try:
        raw = open(sys.argv[2], errors="replace").read()
    except OSError:
        raw = ""
    m = re.search(r"([\d,]+) tokens", raw)
    print(m.group(1).replace(",", "") if m else "-")


globals()["cmd_" + sys.argv[1]]()
PY
export MODEL
PY=python3

oracle() { "$PY" "$PYHELPER" "$@"; }

selected() {
  [ "$ONLY" = "all" ] && return 0
  local short="${1%%-*}"
  case ",$ONLY," in *",$1,"*|*",$short,"*) return 0 ;; esac
  return 1
}

stop_server() {
  [ -n "$SERVER_PID" ] || return 0
  kill -TERM "$SERVER_PID" 2>/dev/null
  local i
  for i in $(seq 1 30); do
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 1
  done
  kill -KILL "$SERVER_PID" 2>/dev/null
  wait "$SERVER_PID" 2>/dev/null
  SERVER_PID=""
}

# start_server <label> <model-arg> [extra flags...]; returns 0 when /health answers.
start_server() {
  local label="$1" marg="$2"
  shift 2
  stop_server
  SERVER_LOG="$WORK/server-${label}-$(date +%s).log"
  vmlx serve "$marg" --port "$PORT" "$@" > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  local i code
  for i in $(seq 1 "$READY_POLLS"); do
    kill -0 "$SERVER_PID" 2>/dev/null || { say "SERVER label=$label state=exited log=$SERVER_LOG"; return 1; }
    code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "$BASE_URL/health" 2>/dev/null)
    [ "$code" = "200" ] && { say "SERVER label=$label state=ready polls=$i log=$SERVER_LOG"; return 0; }
    sleep 5
  done
  say "SERVER label=$label state=not_ready_after_polls=$READY_POLLS log=$SERVER_LOG"
  return 1
}

# loglines <label>: positive control (the engine load line) in the same call as the two absence checks.
loglines() {
  local label="$1" ctl dec ref
  ctl=$(grep -c 'Loading model with' "$SERVER_LOG")
  dec=$(grep -c 'Declared model context ceiling' "$SERVER_LOG")
  ref=$(grep -c 'Refreshed max prompt/context length' "$SERVER_LOG")
  say "LOGCTX $label $(grep -h -m3 'Context memory advisory\|Max prompt/context length\|Declared model context\|Refreshed max prompt' "$SERVER_LOG" | cut -c1-220 | tr '\n' '~')"
  say "RESULT cell=LOG-$label control_load_line_count=$ctl declared_ceiling_line=$([ "$dec" -gt 0 ] && echo present || echo absent)($dec) refreshed_line=$([ "$ref" -gt 0 ] && echo present || echo absent)($ref) expected=present,present"
  ROWS=$((ROWS + 1))
  if [ "$ctl" -eq 0 ]; then say "RESULT cell=LOG-$label verdict=VOID control line missing"; FAILS=$((FAILS + 1)); return; fi
  if [ "$dec" -eq 0 ] || [ "$ref" -eq 0 ]; then say "RESULT cell=LOG-$label verdict=FAIL"; FAILS=$((FAILS + 1)); else say "RESULT cell=LOG-$label verdict=PASS"; fi
}

# capcell <label> <expected: set|unset>: the cap the server reports on /v1/capabilities.
capcell() {
  local label="$1" f="$WORK/cap-$1.json" code cap
  code=$(curl -s -m 20 -o "$f" -w '%{http_code}' "$BASE_URL/v1/capabilities")
  cap=$("$PY" - "$f" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(d.get("max_prompt_tokens", "key-missing"))
except Exception as e:
    print("unreadable")
PY
)
  local d
  d=$(oracle declared)
  ROWS=$((ROWS + 1))
  if [ "$cap" = "$d" ]; then
    say "RESULT cell=CAP-$label http=$code capabilities.max_prompt_tokens=$cap declared=$d expected=$d verdict=PASS"
  else
    say "RESULT cell=CAP-$label http=$code capabilities.max_prompt_tokens=$cap declared=$d expected=$d verdict=FAIL"
    FAILS=$((FAILS + 1))
  fi
}

# cell <id> <label> <route chat|completions> <stream 0|1> <target_tokens> <max_tokens> <wait> <explicit_cap> [open|known]
# Returns 28 on a client timeout so the caller restarts the server.
cell() {
  local id="$1" label="$2" route="$3" stream="$4" target="$5" mt="$6" wait="$7" cap="$8" open="${9:-}"
  selected "$id" || return 0
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    say "RESULT cell=$id mode=$label verdict=NOT-RUN deadline"
    ROWS=$((ROWS + 1))
    return 0
  fi
  local body="$WORK/body-$id.json" out="$WORK/out-$id.txt" url ptoks expect t0 t1 status rc form code ctoks stoks finish msgc verdict
  ptoks=$(oracle body "$route" "$target" "$mt" "$stream" "$body")
  expect=$(oracle expect "$ptoks" "$mt" "$cap")
  if [ "$route" = "chat" ]; then url="$BASE_URL/v1/chat/completions"; else url="$BASE_URL/v1/completions"; fi
  t0=$(date +%s)
  status=$(curl -s -N -m "$wait" -o "$out" -w '%{http_code}' -X POST "$url" -H 'content-type: application/json' -d @"$body")
  rc=$?
  t1=$(date +%s)
  read -r form code ctoks stoks finish <<EOF
$(oracle classify "$out" "$stream")
EOF
  msgc=$(oracle msgcount "$out")
  ROWS=$((ROWS + 1))
  local got
  if [ "$rc" -eq 28 ]; then
    got="timeout"
    [ "$form" = "none" ] && status="TIMEOUT"
    TIMEOUTS=$((TIMEOUTS + 1))
  elif [ "$code" = "prompt_too_long" ]; then
    got="reject"
  elif [ "$form" = "completion" ] || [ "$form" = "sse_stream" ]; then
    got="ok"
  else
    got="other"
  fi
  if [ "$stoks" != "-" ] && [ "$stoks" != "$ptoks" ]; then
    say "RESULT cell=$id verdict=VOID oracle token count $ptoks differs from the server's usage.prompt_tokens $stoks"
    FAILS=$((FAILS + 1))
  fi
  if [ "$msgc" != "-" ] && [ "$msgc" != "$ptoks" ]; then
    say "RESULT cell=$id verdict=VOID oracle token count $ptoks differs from the token count in the server's message $msgc"
    FAILS=$((FAILS + 1))
  fi
  if [ "$open" = "open" ]; then
    verdict="OPEN"
    OPEN=$((OPEN + 1))
  elif [ "$open" = "known" ]; then
    verdict="KNOWN-OTHER-DEFECT"
  elif [ "$got" = "$expect" ]; then
    verdict="PASS"
  else
    verdict="FAIL"
    FAILS=$((FAILS + 1))
  fi
  say "RESULT cell=$id mode=$label route=$route stream=$stream oracle_prompt_tokens=$ptoks max_tokens=$mt status=$status form=$form code=$code curl_rc=$rc elapsed_s=$((t1 - t0)) server_prompt_tokens=$stoks message_prompt_tokens=$msgc completion_tokens=$ctoks finish=$finish oracle_expect=$expect observed=$got verdict=$verdict"
  say "BODY cell=$id $(head -c 300 "$out" | tr '\n' ' ')"
  [ "$rc" -eq 28 ] && return 28
  return 0
}

D=$(oracle declared)
say "ENV model=$MODEL declared_context=$D port=$PORT wait_hang=$WAIT_HANG wait_fit=$WAIT_FIT budget=$TOTAL_BUDGET"
say "ENV vmlx=$(python3 -c 'import importlib.metadata as m; print(m.version("vmlx"))') mlx-lm=$(python3 -c 'import importlib.metadata as m; print(m.version("mlx-lm"))') mlx=$(python3 -c 'import importlib.metadata as m; print(m.version("mlx"))') os=$(sw_vers -productVersion 2>/dev/null || uname -sr) mem_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo unknown)"
SNAP=$(oracle snapshot)
say "ENV snapshot=$SNAP"

UNDER=$((D - 1))
AT=$D
OVER=$((D + 1))

# serve_with <label> <model-arg> <flags...> -- runs the cells named in CELLS_<label> in order,
# restarting after a client timeout so a stuck prefill cannot poison the next cell.
declare -a CELLSPEC

# Each spec: id|route|stream|target_tokens|max_tokens|wait_s|explicit_cap|open
run_group() {
  local label="$1" marg="$2"
  shift 2
  local flags=("$@") spec started=0 any=0 logged=0
  local id route stream target mt wait cap open
  for spec in "${CELLSPEC[@]}"; do
    selected "${spec%%|*}" && any=1
  done
  [ "$any" -eq 1 ] || return 0
  for spec in "${CELLSPEC[@]}"; do
    IFS='|' read -r id route stream target mt wait cap open <<EOF
$spec
EOF
    selected "$id" || continue
    if [ "$started" -eq 0 ]; then
      if ! start_server "$label" "$marg" ${flags[@]+"${flags[@]}"}; then
        say "RESULT cell=$id mode=$label verdict=FAIL server did not start"
        FAILS=$((FAILS + 1)); ROWS=$((ROWS + 1))
        return 0
      fi
      started=1
      if [ "$logged" -eq 0 ]; then
        logged=1
        loglines "$label"
        if [ "$label" != "explicit" ]; then capcell "$label"; fi
      fi
    fi
    cell "$id" "$label" "$route" "$stream" "$target" "$mt" "$wait" "$cap" "$open"
    rc=$?
    if [ "$rc" -eq 28 ]; then started=0; stop_server; fi
    # A long prefill leaves its KV cache resident and the server's Metal working-set guard then
    # answers 503 to the next request, so every long prompt gets a fresh server.
    if [ "$target" -ge 20000 ] && [ "$started" -eq 1 ]; then started=0; stop_server; fi
  done
  stop_server
}

# 1. default launch, model given as a Hugging Face repo id (the issue's repro)
CELLSPEC=(
  "C1-fits-short|chat|0|20|8|$WAIT_FIT|0|"
  "C5-bug-60008|chat|0|60008|16|$WAIT_HANG|0|"
  "C6-bug-60008-stream|chat|1|60008|16|$WAIT_HANG|0|"
  "C8-bug-60008-completions|completions|0|60008|16|$WAIT_HANG|0|"
  "C4-over-by-one|chat|0|$OVER|1|$WAIT_HANG|0|"
  "C3-at-ceiling|chat|0|$AT|1|$WAIT_HANG|0|open"
  "C2-under-by-one|chat|0|$UNDER|1|$WAIT_FIT|0|"
  "C7-under-by-one-stream|chat|1|$UNDER|1|$WAIT_FIT|0|"
  "C9-fits-output-over-remaining|chat|0|$((D - 4))|64|$WAIT_FIT|0|open"
)
run_group default "$MODEL"

# 2. direct single-request engine (the other engine path), repo id
CELLSPEC=(
  "N1-fits-short|chat|0|20|8|$WAIT_FIT|0|known"
  "N5-bug-60008|chat|0|60008|16|$WAIT_HANG|0|"
)
run_group simple "$MODEL" --no-continuous-batching

# 3. context that must NOT change: model given as a local directory (declared ceiling is read at base)
CELLSPEC=(
  "L1-fits-short|chat|0|20|8|$WAIT_FIT|0|"
  "L5-bug-60008|chat|0|60008|16|$WAIT_HANG|0|"
  "L6-bug-60008-stream|chat|1|60008|16|$WAIT_HANG|0|"
)
run_group local "$SNAP"

# 4. option mode: explicit --max-prompt-tokens wins (base already honours it)
CELLSPEC=(
  "E1-explicit-fits|chat|0|100|8|$WAIT_FIT|8192|"
  "E2-explicit-over|chat|0|9000|8|$WAIT_FIT|8192|"
)
run_group explicit "$MODEL" --max-prompt-tokens 8192

stop_server
say "SUMMARY rows=$ROWS fail=$FAILS open=$OPEN timeouts=$TIMEOUTS elapsed_s=$(( $(date +%s) - START_TS )) work=$WORK"
[ "$ROWS" -gt 0 ] || { say "SUMMARY no rows ran: void"; exit 3; }
[ "$FAILS" -eq 0 ] && exit 0
exit 1
