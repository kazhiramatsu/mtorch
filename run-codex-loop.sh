#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
ROOT_DIR="$(cd "$ROOT_DIR" && pwd -P)"

LOG_DIR="${LOG_DIR:-.codex-runs}"
SANDBOX="${SANDBOX:-workspace-write}"
SLEEP_SEC="${SLEEP_SEC:-5}"
STOP_FILE="${STOP_FILE:-.codex-stop}"
PROMPT_FILE="${PROMPT_FILE:-}"
APPEND_STOP_INSTRUCTIONS="${APPEND_STOP_INSTRUCTIONS:-1}"
MAX_RUNS="${MAX_RUNS:-0}"

usage() {
  cat <<EOF
usage:
  $0 [options] [prompt...]
  $0 --prompt-file prompt.md
  PROMPT_FILE=prompt.md $0
  $0 - < prompt.md

options:
  -f, --file, --prompt-file FILE
      Read the user prompt from FILE.
  -
      Read the user prompt from stdin.
  --no-stop-instructions
      Do not append the default stop-file instructions.
  -h, --help
      Show this help.

environment:
  ROOT_DIR, LOG_DIR, SANDBOX, SLEEP_SEC, STOP_FILE
  PROMPT_FILE, APPEND_STOP_INSTRUCTIONS=0|1, MAX_RUNS=0|N
EOF
}

mkdir -p "$ROOT_DIR/$LOG_DIR"

if ! command -v codex >/dev/null 2>&1; then
  echo "error: codex command not found" >&2
  exit 1
fi

if [[ "$MAX_RUNS" == "" || "$MAX_RUNS" == *[!0-9]* ]]; then
  echo "error: MAX_RUNS must be 0 or a positive integer" >&2
  exit 1
fi

if [[ "$APPEND_STOP_INSTRUCTIONS" != "0" && "$APPEND_STOP_INSTRUCTIONS" != "1" ]]; then
  echo "error: APPEND_STOP_INSTRUCTIONS must be 0 or 1" >&2
  exit 1
fi

prompt_args=()

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -f|--file|--prompt-file)
      shift
      if [[ "$#" -eq 0 ]]; then
        echo "error: missing prompt file after --prompt-file" >&2
        exit 1
      fi
      PROMPT_FILE="$1"
      ;;
    --no-stop-instructions)
      APPEND_STOP_INSTRUCTIONS=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      prompt_args+=("$@")
      break
      ;;
    -)
      PROMPT_FILE="-"
      ;;
    *)
      prompt_args+=("$1")
      ;;
  esac
  shift
done

USER_PROMPT=""
prompt_source=""

if [[ -n "$PROMPT_FILE" ]]; then
  if [[ "${#prompt_args[@]}" -ne 0 ]]; then
    echo "error: pass either prompt text or a prompt file, not both" >&2
    exit 1
  fi

  if [[ "$PROMPT_FILE" == "-" ]]; then
    USER_PROMPT="$(cat)"
    prompt_source="stdin"
  else
    if [[ ! -r "$PROMPT_FILE" ]]; then
      echo "error: prompt file is not readable: $PROMPT_FILE" >&2
      exit 1
    fi
    USER_PROMPT="$(<"$PROMPT_FILE")"
    prompt_source="$PROMPT_FILE"
  fi
elif [[ "${#prompt_args[@]}" -ne 0 ]]; then
  USER_PROMPT="${prompt_args[*]}"
  prompt_source="argv"
elif [[ ! -t 0 ]]; then
  USER_PROMPT="$(cat)"
  prompt_source="stdin"
else
  usage >&2
  exit 1
fi

if [[ -z "$USER_PROMPT" ]]; then
  echo "error: prompt is empty" >&2
  exit 1
fi

if [[ "$APPEND_STOP_INSTRUCTIONS" == "1" ]]; then
  PROMPT=$(cat <<EOF
$USER_PROMPT

追加ルール:
- もう作業が残っていないと判断した場合は、次の終了ファイルを作成してください。
$STOP_FILE

例:
touch "$STOP_FILE"

- 終了ファイルを作成した場合は、それ以上不要な変更をしないでください。
- ターン終了時に作業内容をメモリに保存してください。
- メモリには過去の作業内容は削除して再開に必要な情報を最低限記述してください。
EOF
)
else
  PROMPT="$USER_PROMPT"
fi

run_count=0

while true; do
  cd "$ROOT_DIR"

  if [[ -e "$STOP_FILE" ]]; then
    echo "終了ファイルを検出したため終了します: $STOP_FILE"
    exit 0
  fi

  if [[ "$MAX_RUNS" -ne 0 && "$run_count" -ge "$MAX_RUNS" ]]; then
    echo "最大実行回数に達したため終了します: $MAX_RUNS"
    exit 0
  fi

  run_count=$((run_count + 1))
  ts="$(date '+%Y%m%d-%H%M%S')"
  log_file="$ROOT_DIR/$LOG_DIR/run-${ts}-${run_count}.log"

  echo "========================================"
  echo "Codex exec run: $run_count"
  echo "root:      $ROOT_DIR"
  echo "sandbox:   $SANDBOX"
  echo "stop file: $STOP_FILE"
  echo "prompt:    $prompt_source"
  echo "log:       $log_file"
  echo "========================================"

  {
    echo "========================================"
    echo "Codex exec run: $run_count"
    echo "started:   $(date '+%Y-%m-%d %H:%M:%S')"
    echo "root:      $ROOT_DIR"
    echo "sandbox:   $SANDBOX"
    echo "stop file: $STOP_FILE"
    echo "prompt:    $prompt_source"
    echo "prompt bytes: ${#PROMPT}"
    echo "========================================"
  } >"$log_file"

  set +e
  codex exec \
    --skip-git-repo-check \
    --cd "$ROOT_DIR" \
    --sandbox "$SANDBOX" \
    -c 'approval_policy="never"' \
    - >>"$log_file" 2>&1 <<<"$PROMPT"

  status=$?
  set -e

  {
    echo "========================================"
    echo "finished:  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "status:    $status"
    echo "========================================"
  } >>"$log_file"

  if [[ -e "$STOP_FILE" ]]; then
    echo "終了ファイルを検出したため終了します: $STOP_FILE"
    exit 0
  fi

  if [[ "$status" -ne 0 ]]; then
    echo "Codex failed/interrupted with status $status. 終了します。"
    echo "詳細ログ: $log_file"
    exit "$status"
  fi

  sleep "$SLEEP_SEC"
done
