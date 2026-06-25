#!/usr/bin/env bash
#
# Stage the Parakeet-TDT-v2 CoreML runtime models into Resources/Models so the
# folder reference in project.yml copies them verbatim into the app bundle (no
# runtime download).
#
# Source: ../parakeet_coreml_v2_final (the INT8-encoder pipeline; see the top-level
# README's four-step recipe). The four runtime models already exist there as
# compiled .mlmodelc, so we just copy them; if only a .mlpackage is present we
# compile it with `xcrun coremlcompiler`. Override the source with MODELS_SRC=...
#
# Idempotent: re-running refreshes the staged copies. Safe to run from anywhere.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
COREML_DIR="$(cd "${APP_DIR}/.." && pwd)"          # .../parakeet-tdt-v2-0.6b/coreml

SRC="${MODELS_SRC:-${COREML_DIR}/parakeet_coreml_v2_final}"
DEST="${APP_DIR}/Resources/Models"

MODELS=(
  parakeet_preprocessor
  parakeet_encoder
  parakeet_decoder
  parakeet_joint_decision_single_step
)
VOCAB="parakeet_vocab.json"

echo "==> Source      : ${SRC}"
echo "==> Destination : ${DEST}"
[ -d "${SRC}" ] || { echo "ERROR: missing source dir ${SRC}"; exit 1; }

rm -rf "${DEST}"
mkdir -p "${DEST}"

for name in "${MODELS[@]}"; do
  compiled="${SRC}/${name}.mlmodelc"
  pkg="${SRC}/${name}.mlpackage"
  if [ -d "${compiled}" ]; then
    echo "    copying ${name}.mlmodelc"
    cp -R "${compiled}" "${DEST}/${name}.mlmodelc"
  elif [ -d "${pkg}" ]; then
    echo "    compiling ${name}.mlpackage -> .mlmodelc"
    xcrun coremlcompiler compile "${pkg}" "${DEST}" >/dev/null
  else
    echo "ERROR: missing ${compiled} or ${pkg}"; exit 1
  fi
done

if [ -f "${SRC}/${VOCAB}" ]; then
  cp -f "${SRC}/${VOCAB}" "${DEST}/${VOCAB}"
  echo "    copied ${VOCAB}"
else
  echo "ERROR: missing ${SRC}/${VOCAB}"; exit 1
fi

echo ""
echo "==> Done. Staged sizes:"
du -sh "${DEST}" 2>/dev/null || true
du -sh "${DEST}"/*.mlmodelc 2>/dev/null || true
echo ""
echo "Next: cd ${APP_DIR} && xcodegen generate && open ParakeetTranscribe.xcodeproj"
