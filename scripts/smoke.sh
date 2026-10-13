#!/usr/bin/env bash
# AML Add/Search 契约自测（对应 agentmemories.ai/api-guide 的 Add/Search integration contract）
# 用法：
#   BASE=https://your-domain MEMORYSUM_TOKEN=$(head -1 keys.txt) ./scripts/smoke.sh
# 凭据只从环境变量读，不要写进脚本、URL、截图或聊天里。
set -euo pipefail

: "${MEMORYSUM_TOKEN:?请先 export MEMORYSUM_TOKEN=<keys.txt 中的一行>}"
BASE="${BASE:-http://127.0.0.1:8001}"
TOP_K="${TOP_K:-100}"          # 正式 Full 用 100
RUN_ID="smoke-$(date +%s)"     # 每次跑用新 request_id，避免撞幂等/422
USER_ID="${USER_ID:-smoke-user}"
SESSION_ID="${SESSION_ID:-smoke-session}"

api() { # method path json
  curl -sS --fail-with-body -X "$1" "$BASE$2" \
    -H "Authorization: Bearer $MEMORYSUM_TOKEN" \
    -H 'Content-Type: application/json' \
    -d "$3"
}

echo "== health（免鉴权 GET，任意 2xx 即健康）"
code=$(curl -sS -o /dev/null -w '%{http_code}' "$BASE/health")
[[ "$code" == 2* ]] || { echo "FAIL health HTTP:$code"; exit 1; }
echo "OK HTTP:$code"

echo "== add（必须 success=true 且回显三个 ID）"
add=$(api POST /add "$(jq -nc --arg rid "$RUN_ID" --arg uid "$USER_ID" --arg sid "$SESSION_ID" '{
  request_id: $rid, user_id: $uid, session_id: $sid,
  messages: [{role:"user", timestamp:(now*1000|floor), content:"我在上海工作，偏好周五下午开会。"}]
}')")
jq -e --arg rid "$RUN_ID" --arg uid "$USER_ID" --arg sid "$SESSION_ID" \
  '.success==true and .request_id==$rid and .user_id==$uid and .session_id==$sid' \
  <<<"$add" >/dev/null || { echo "FAIL add 回显不符: $add"; exit 1; }
echo "OK $(jq -c . <<<"$add")"

echo "== search（data 必须是数组、每项有 id+content、条数 <= top_k=${TOP_K}）"
search=$(api POST /search "$(jq -nc --arg uid "$USER_ID" --argjson k "$TOP_K" \
  '{query:"我偏好什么时候开会？", user_id:$uid, top_k:$k}')")
jq -e --argjson k "$TOP_K" \
  '.data|type=="array" and length<=$k and all(.[]; ((.id|type)=="string") and ((.content|type)=="string" and (.content|length)>0))' \
  <<<"$search" >/dev/null || { echo "FAIL search 契约: $search"; exit 1; }
echo "OK hits=$(jq '.data|length' <<<"$search")"
jq -c '.data[0] // "（无命中）"' <<<"$search"

echo "== 负例：无凭据应 401（AML: 401 不自动重试）"
code=$(curl -sS -o /dev/null -w '%{http_code}' -X POST "$BASE/search" \
  -H 'Content-Type: application/json' \
  -d "$(jq -nc '{query:"x", user_id:"smoke-user", top_k:10}')")
[[ "$code" == "401" ]] || { echo "FAIL 期望 401，实得 $code"; exit 1; }
echo "OK HTTP:401"

echo "== 全部通过"
