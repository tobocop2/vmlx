#!/bin/bash
# Stability repeat for jjang-ai/vmlx#398: run the new reranker load tests 20
# times, each time together with the two existing reranker suites in one
# process, so shared module state (the server's reranker global, the MLX
# allocator) is exercised under repetition. Reports the failure count.
#
# Env: SRC (checkout under test, default $GITHUB_WORKSPACE/src), REPEATS.
set -u

SRC="${SRC:-${GITHUB_WORKSPACE:-$PWD}/src}"
REPEATS="${REPEATS:-20}"
WORK=$(mktemp -d "${WORK_ROOT:-${TMPDIR:-/tmp}}/vmlx398rep.XXXXXX")
FAILRUNS=0
TOTALPASSED=0
TOTALFAILED=0

say() { printf '%s\n' "$*"; }

say "ENV src=$SRC head=$(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown) repeats=$REPEATS"
python3 -m pip install --disable-pip-version-check 'pytest==8.4.2' > "$WORK/pip.log" 2>&1 \
  || say "ENV pytest install failed"

for i in $(seq 1 "$REPEATS"); do
  log="$WORK/run-$i.log"
  (cd "$SRC" && python3 -m pytest tests/test_reranker_encoder_load.py \
    tests/test_reranker_encoder.py tests/test_reranker_endpoint.py -q) > "$log" 2>&1
  rc=$?
  line=$(grep -E '[0-9]+ (passed|failed)' "$log" | tail -1)
  passed=$(printf '%s\n' "$line" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' || true)
  failed=$(printf '%s\n' "$line" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+' || true)
  passed=${passed:-0}
  failed=${failed:-0}
  TOTALPASSED=$((TOTALPASSED + passed))
  TOTALFAILED=$((TOTALFAILED + failed))
  if [ "$rc" -ne 0 ] || [ "$failed" -ne 0 ] || [ "$passed" -ne 27 ]; then
    FAILRUNS=$((FAILRUNS + 1))
    say "REPEAT run=$i rc=$rc passed=$passed failed=$failed verdict=FAIL"
    tail -n 15 "$log" | sed "s/^/REPEATLOG $i /"
  else
    say "REPEAT run=$i rc=$rc passed=$passed failed=$failed verdict=PASS"
  fi
done

say "SUMMARY repeats=$REPEATS fail_runs=$FAILRUNS total_passed=$TOTALPASSED total_failed=$TOTALFAILED expected_per_run=27 work=$WORK"
[ "$FAILRUNS" -eq 0 ] || exit 1
[ "$TOTALPASSED" -eq $((REPEATS * 27)) ] || exit 1
exit 0