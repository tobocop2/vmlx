#!/bin/bash
# Acceptance script for jjang-ai/vmlx#398: /v1/rerank with a BGE / XLM-RoBERTa
# cross-encoder checkpoint must answer a clear 4xx naming the unsupported
# architecture, not an HTTP 500 that lists the checkpoint weights. Rerankers
# that work today (a ModernBERT classification encoder and the jina
# late-interaction model) must keep scoring "Paris is the capital of France."
# first.
#
# Runs on a macos-15 runner with the package under test and mlx-lm 0.31.3
# installed. The workflow prefetches only Qwen3-0.6B-4bit (the chat model the
# server serves); this script downloads the rerankers itself through the
# server's lazy load. Every wait is bounded (curl -m, readiness polls with a
# count, one global deadline). Each RESULT row is decided by an oracle that is
# not the server: the unit-test rows parse pytest's own summary counts, the
# rerank rows parse the response body, and the refusal rows assert a positive
# twin (the architecture IS named) beside each absence (no weight list).
#
# Env: ONLY (comma list of cell ids, default all), SRC (checkout under test,
# default $GITHUB_WORKSPACE/src), WAIT_FAST, WAIT_LOAD, TOTAL_BUDGET, PORT.
set -u

CHAT_MODEL="${CHAT_MODEL:-mlx-community/Qwen3-0.6B-4bit}"
BGE_ORIG="${BGE_ORIG:-BAAI/bge-reranker-v2-m3}"
BGE_MLX="${BGE_MLX:-flaglow/BAAI-bge-reranker-v2-m3-mlx-fp16}"
MODERNBERT="${MODERNBERT:-Alibaba-NLP/gte-reranker-modernbert-base}"
JINA="${JINA:-jinaai/jina-reranker-v3-mlx}"
ARCH="XLMRobertaForSequenceClassification"
PORT="${PORT:-18398}"
BASE_URL="http://127.0.0.1:${PORT}"
WAIT_FAST="${WAIT_FAST:-120}"
WAIT_LOAD="${WAIT_LOAD:-900}"
READY_POLLS="${READY_POLLS:-120}"
TOTAL_BUDGET="${TOTAL_BUDGET:-5400}"
ONLY="${ONLY:-all}"
SRC="${SRC:-${GITHUB_WORKSPACE:-$PWD}/src}"
START_TS=$(date +%s)
DEADLINE=$((START_TS + TOTAL_BUDGET))
WORK=$(mktemp -d "${WORK_ROOT:-${TMPDIR:-/tmp}}/vmlx398.XXXXXX")
SERVER_PID=""
FAILS=0
ROWS=0
TIMEOUTS=0

say() { printf '%s\n' "$*"; }

PYHELPER="$WORK/oracle.py"
cat > "$PYHELPER" <<'PY'
import json, os, re, sys


def cmd_fixture():
    # fixture <dir>: a local XLM-R classification bundle, config + two
    # checkpoint-shaped keys (the issue's fingerprint in miniature).
    import numpy as np
    from safetensors.numpy import save_file
    d = sys.argv[2]
    os.makedirs(d, exist_ok=True)
    config = {
        "model_type": "xlm-roberta",
        "architectures": ["XLMRobertaForSequenceClassification"],
        "hidden_size": 4, "num_hidden_layers": 1, "intermediate_size": 8,
        "num_attention_heads": 2, "max_position_embeddings": 16,
        "vocab_size": 32,
    }
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(config, f)
    save_file(
        {
            "roberta.embeddings.LayerNorm.weight": np.zeros(4, dtype=np.float32),
            "classifier.out_proj.weight": np.zeros((1, 4), dtype=np.float32),
        },
        os.path.join(d, "model.safetensors"),
    )
    print(d)


def cmd_body():
    # body <model> <outfile>
    body = {
        "model": sys.argv[2],
        "query": "What is the capital of France?",
        "documents": [
            "Bananas are yellow.",
            "Paris is the capital of France.",
            "Dogs bark.",
        ],
    }
    with open(sys.argv[3], "w") as f:
        json.dump(body, f)


def cmd_classify():
    # classify <outfile> <arch> ->
    # form topidx indices desc arch_present params_present weights_present n
    path, arch = sys.argv[2], sys.argv[3]
    try:
        raw = open(path, errors="replace").read()
    except OSError:
        raw = ""
    form, topidx, indices, desc = "none", "-", "-", "-"
    arch_p = params_p = weights_p = 0
    n = "-"
    try:
        doc = json.loads(raw)
    except ValueError:
        doc = None
    if isinstance(doc, dict):
        if isinstance(doc.get("results"), list):
            form = "results"
            rs = doc["results"]
            n = str(len(rs))
            if rs:
                topidx = str(rs[0].get("index", "-"))
                indices = ",".join(str(r.get("index")) for r in
                                   sorted(rs, key=lambda r: r.get("index", -1)))
                scores = [r.get("relevance_score") for r in rs]
                desc = "1" if all(
                    isinstance(a, (int, float)) and isinstance(b, (int, float))
                    and a >= b for a, b in zip(scores, scores[1:])
                ) else "0"
        elif "detail" in doc:
            form = "error"
            detail = str(doc["detail"])
            arch_p = 1 if arch in detail else 0
            params_p = 1 if "parameters not in model" in detail else 0
            weights_p = 1 if ("classifier.out_proj.weight" in detail
                              or "classifier.dense.weight" in detail
                              or "roberta.embeddings.LayerNorm" in detail) else 0
        elif "choices" in doc:
            form = "chat"
    print(form, topidx, indices, desc, arch_p, params_p, weights_p, n)


def cmd_pytest_summary():
    # pytest_summary <logfile> -> passed failed errors skipped
    raw = open(sys.argv[2], errors="replace").read()
    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    summaries = re.findall(
        r"^(?:=+ )?([0-9a-z, ]+?) in [0-9.]+s.*$", raw, re.MULTILINE)
    if summaries:
        for part in summaries[-1].split(","):
            m = re.match(r"\s*(\d+)\s+(\w+)", part)
            if m:
                num, word = int(m.group(1)), m.group(2)
                if word in ("error", "errors"):
                    counts["errors"] += num
                elif word in counts:
                    counts[word] += num
    print(counts["passed"], counts["failed"], counts["errors"], counts["skipped"])


globals()["cmd_" + sys.argv[1]]()
PY
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

start_server() {
  stop_server
  SERVER_LOG="$WORK/server-$(date +%s).log"
  vmlx serve "$CHAT_MODEL" --port "$PORT" > "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  local i code
  for i in $(seq 1 "$READY_POLLS"); do
    kill -0 "$SERVER_PID" 2>/dev/null || { say "SERVER state=exited log=$SERVER_LOG"; return 1; }
    code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "$BASE_URL/health" 2>/dev/null)
    [ "$code" = "200" ] && { say "SERVER state=ready polls=$i log=$SERVER_LOG"; return 0; }
    sleep 5
  done
  say "SERVER state=not_ready_after_polls=$READY_POLLS log=$SERVER_LOG"
  return 1
}

# unit_cell <id> <expected_total> <pytest args...>
unit_cell() {
  local id="$1" want="$2"
  shift 2
  selected "$id" || return 0
  ROWS=$((ROWS + 1))
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    say "RESULT cell=$id verdict=NOT-RUN deadline"; FAILS=$((FAILS + 1)); return 0
  fi
  local log="$WORK/pytest-$id.log" rc passed failed errors skipped total
  (cd "$SRC" && "$PY" -m pytest "$@") > "$log" 2>&1
  rc=$?
  read -r passed failed errors skipped <<EOF
$(oracle pytest_summary "$log")
EOF
  total=$((passed + failed + errors + skipped))
  say "PYTEST cell=$id rc=$rc passed=$passed failed=$failed errors=$errors skipped=$skipped total=$total expected_total=$want"
  tail -n 25 "$log" | sed "s/^/PYTESTLOG $id /"
  if [ "$total" -ne "$want" ] || [ "$passed" -ne "$want" ] || [ "$rc" -ne 0 ]; then
    say "RESULT cell=$id verdict=FAIL expected=$want passed, 0 failed"
    FAILS=$((FAILS + 1))
  else
    say "RESULT cell=$id verdict=PASS"
  fi
}

# rerank_cell <id> <model> <kind refuse|score> <wait>
rerank_cell() {
  local id="$1" model="$2" kind="$3" wait="$4"
  selected "$id" || return 0
  ROWS=$((ROWS + 1))
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    say "RESULT cell=$id verdict=NOT-RUN deadline"; FAILS=$((FAILS + 1)); return 0
  fi
  local body="$WORK/body-$id.json" out="$WORK/out-$id.txt" attempt status rc verdict
  oracle body "$model" "$body"
  verdict="FAIL"
  for attempt in 1 2; do
    local t0 t1 form topidx indices desc arch_p params_p weights_p n
    t0=$(date +%s)
    status=$(curl -s -m "$wait" -o "$out" -w '%{http_code}' -X POST "$BASE_URL/v1/rerank" \
      -H 'content-type: application/json' -d @"$body")
    rc=$?
    t1=$(date +%s)
    read -r form topidx indices desc arch_p params_p weights_p n <<EOF
$(oracle classify "$out" "$ARCH")
EOF
    if [ "$rc" -eq 28 ]; then
      TIMEOUTS=$((TIMEOUTS + 1))
      say "RESULT cell=$id attempt=$attempt curl_rc=28 timeout after $((t1 - t0))s"
      start_server || { verdict="FAIL"; break; }
      continue
    fi
    if [ "$status" = "503" ]; then
      say "RESULT cell=$id attempt=$attempt http=503 pressure; restarting server"
      start_server || { verdict="FAIL"; break; }
      continue
    fi
    if [ "$kind" = "refuse" ]; then
      if [ "$status" = "400" ] && [ "$form" = "error" ] && [ "$arch_p" = "1" ] \
         && [ "$params_p" = "0" ] && [ "$weights_p" = "0" ]; then
        verdict="PASS"
      else
        verdict="FAIL"
      fi
    else
      if [ "$status" = "200" ] && [ "$form" = "results" ] && [ "$n" = "3" ] \
         && [ "$indices" = "0,1,2" ] && [ "$topidx" = "1" ] && [ "$desc" = "1" ]; then
        verdict="PASS"
      else
        verdict="FAIL"
      fi
    fi
    say "RESULT cell=$id kind=$kind model=$model attempt=$attempt http=$status curl_rc=$rc elapsed_s=$((t1 - t0)) form=$form top_index=$topidx indices=$indices scores_desc=$desc arch_named=$arch_p params_listed=$params_p weights_listed=$weights_p n_results=$n expected=$([ "$kind" = refuse ] && echo '400,arch_named,no_weights' || echo '200,paris_first') verdict=$verdict"
    say "BODY cell=$id $(head -c 400 "$out" | tr '\n' ' ')"
    break
  done
  [ "$verdict" = "PASS" ] || FAILS=$((FAILS + 1))
}

say "ENV src=$SRC head=$(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown) dirty=$(git -C "$SRC" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
say "ENV chat_model=$CHAT_MODEL bge_orig=$BGE_ORIG bge_mlx=$BGE_MLX modernbert=$MODERNBERT jina=$JINA port=$PORT budget=$TOTAL_BUDGET"
say "ENV vmlx=$("$PY" -c 'import importlib.metadata as m; print(m.version("vmlx"))') mlx-lm=$("$PY" -c 'import importlib.metadata as m; print(m.version("mlx-lm"))') mlx-embeddings=$("$PY" -c 'import importlib.metadata as m; print(m.version("mlx-embeddings"))') mlx=$("$PY" -c 'import importlib.metadata as m; print(m.version("mlx"))') os=$(sw_vers -productVersion 2>/dev/null || uname -sr) mem_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo unknown)"

# Unit rows: the new red-at-base tests, then the existing reranker suites as
# must-not-change guards. pytest is pinned to the contract CI's version.
if selected U1 || selected U2; then
  "$PY" -m pip install --disable-pip-version-check 'pytest==8.4.2' > "$WORK/pip-pytest.log" 2>&1 \
    || say "ENV pytest install failed (see $WORK/pip-pytest.log)"
fi
unit_cell U1 10 tests/test_reranker_encoder_load.py -q
unit_cell U2 17 tests/test_reranker_encoder.py tests/test_reranker_endpoint.py -q

# Live rows: one server serving the prefetched chat model; the reranker is
# lazy-loaded per request and hot-swapped when the model changes.
if selected C0 || selected R5 || selected R1 || selected R2 || selected R3 || selected R4; then
  if start_server; then
    if selected C0; then
      ROWS=$((ROWS + 1))
      cbody="$WORK/body-C0.json"
      cout="$WORK/out-C0.txt"
      printf '%s' '{"model":"'"$CHAT_MODEL"'","messages":[{"role":"user","content":"Say ok."}],"max_tokens":8}' > "$cbody"
      cstatus=$(curl -s -m "$WAIT_FAST" -o "$cout" -w '%{http_code}' -X POST "$BASE_URL/v1/chat/completions" \
        -H 'content-type: application/json' -d @"$cbody")
      cform=$(oracle classify "$cout" "$ARCH" | cut -d' ' -f1)
      if [ "$cstatus" = "200" ] && [ "$cform" = "chat" ]; then
        say "RESULT cell=C0 http=$cstatus form=$cform verdict=PASS"
      else
        say "RESULT cell=C0 http=$cstatus form=$cform verdict=FAIL"
        FAILS=$((FAILS + 1))
      fi
    fi
    FIXTURE=$(oracle fixture "$WORK/xlm-r-fixture")
    say "ENV fixture=$FIXTURE"
    rerank_cell R5 "$FIXTURE" refuse "$WAIT_FAST"
    rerank_cell R1 "$BGE_ORIG" refuse "$WAIT_LOAD"
    rerank_cell R2 "$BGE_MLX" refuse "$WAIT_LOAD"
    rerank_cell R3 "$MODERNBERT" score "$WAIT_LOAD"
    rerank_cell R4 "$JINA" score "$WAIT_LOAD"
  else
    for id in C0 R5 R1 R2 R3 R4; do
      selected "$id" || continue
      ROWS=$((ROWS + 1))
      say "RESULT cell=$id verdict=FAIL server did not start"
      FAILS=$((FAILS + 1))
    done
  fi
fi

stop_server
say "SUMMARY rows=$ROWS fail=$FAILS timeouts=$TIMEOUTS elapsed_s=$(( $(date +%s) - START_TS )) work=$WORK"
[ "$ROWS" -gt 0 ] || { say "SUMMARY no rows ran: void"; exit 3; }
[ "$FAILS" -eq 0 ] && exit 0
exit 1