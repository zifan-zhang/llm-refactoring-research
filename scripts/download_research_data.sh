#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_DIR="${ROOT_DIR}/tools"
DATA_DIR="${ROOT_DIR}/data"
REPOS_DIR="${ROOT_DIR}/repos"

mkdir -p "${TOOLS_DIR}" "${DATA_DIR}" "${REPOS_DIR}"

echo "[1/4] Downloading RefactoringMiner CLI (v3.0.13)"
RM_VERSION="3.0.13"
RM_ZIP_URL_PRIMARY="https://github.com/tsantalis/RefactoringMiner/releases/download/v${RM_VERSION}/RefactoringMiner-${RM_VERSION}.zip"
RM_ZIP_URL_FALLBACK="https://github.com/tsantalis/RefactoringMiner/releases/download/${RM_VERSION}/RefactoringMiner-${RM_VERSION}.zip"

RM_ZIP_PATH="${TOOLS_DIR}/refactoringminer_${RM_VERSION}.zip"
RM_EXTRACT_DIR="$(find "${TOOLS_DIR}" -maxdepth 1 -type d -name "RefactoringMiner*" | head -n 1 || true)"
RM_BIN_PATH="${RM_EXTRACT_DIR}/bin/RefactoringMiner"

if [[ -n "${RM_EXTRACT_DIR}" && -x "${RM_BIN_PATH}" ]]; then
  echo "RefactoringMiner already exists at ${RM_BIN_PATH}. Skipping download."
else
  rm -rf "${TOOLS_DIR}/RefactoringMiner" "${TOOLS_DIR}/RefactoringMiner-"*
  if ! curl -fL "${RM_ZIP_URL_PRIMARY}" -o "${RM_ZIP_PATH}"; then
    echo "Primary download URL failed. Trying fallback URL."
    curl -fL "${RM_ZIP_URL_FALLBACK}" -o "${RM_ZIP_PATH}"
  fi
  unzip -q "${RM_ZIP_PATH}" -d "${TOOLS_DIR}"
  rm -f "${RM_ZIP_PATH}"

  RM_EXTRACT_DIR="$(find "${TOOLS_DIR}" -maxdepth 1 -type d -name "RefactoringMiner*" | head -n 1 || true)"
  RM_BIN_PATH="${RM_EXTRACT_DIR}/bin/RefactoringMiner"
  if [[ -z "${RM_EXTRACT_DIR}" || ! -x "${RM_BIN_PATH}" ]]; then
    echo "RefactoringMiner binary not found or not executable at ${RM_BIN_PATH}."
    exit 1
  fi
fi

RM_BIN_DIR="${RM_EXTRACT_DIR}/bin"
export PATH="${RM_BIN_DIR}:${PATH}"

ENV_FILE="${ROOT_DIR}/env.sh"
ENV_LINE="export PATH=\"${RM_BIN_DIR}:\$PATH\""
if [[ ! -f "${ENV_FILE}" ]] || ! grep -qF "${ENV_LINE}" "${ENV_FILE}"; then
  echo "${ENV_LINE}" >> "${ENV_FILE}"
fi

echo "RefactoringMiner PATH ensured in ${ENV_FILE}"

echo "[2/4] Ensuring git-lfs is available"
GIT_LFS_DIR="${TOOLS_DIR}/git-lfs"
GIT_LFS_BIN_DIR="${GIT_LFS_DIR}/bin"
GIT_LFS_BIN="${GIT_LFS_BIN_DIR}/git-lfs"

if command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs already available in PATH."
elif [[ -x "${GIT_LFS_BIN}" ]]; then
  export PATH="${GIT_LFS_BIN_DIR}:${PATH}"
  ENV_LINE="export PATH=\"${GIT_LFS_BIN_DIR}:\$PATH\""
  if [[ ! -f "${ENV_FILE}" ]] || ! grep -qF "${ENV_LINE}" "${ENV_FILE}"; then
    echo "${ENV_LINE}" >> "${ENV_FILE}"
  fi
  echo "git-lfs found at ${GIT_LFS_BIN}. PATH updated."
else
  echo "git-lfs not found. Downloading binary release."
  GIT_LFS_API_URL="https://api.github.com/repos/git-lfs/git-lfs/releases/latest"
  GIT_LFS_TAR_URL="$(python3 - <<'PY'
import json
import platform
import urllib.request

url = "https://api.github.com/repos/git-lfs/git-lfs/releases/latest"
with urllib.request.urlopen(url) as f:
    data = json.load(f)

arch = platform.machine().lower()
if arch in ("x86_64", "amd64"):
    arch_key = "amd64"
elif arch in ("aarch64", "arm64"):
    arch_key = "arm64"
else:
    arch_key = arch

asset_url = ""
for asset in data.get("assets", []):
    name = asset.get("name", "")
    if name.endswith(".tar.gz") and "linux" in name and arch_key in name:
        asset_url = asset.get("browser_download_url", "")
        break

print(asset_url)
PY
)"

  if [[ -z "${GIT_LFS_TAR_URL}" ]]; then
    echo "Unable to locate a suitable git-lfs Linux binary for this architecture."
    exit 1
  fi

  mkdir -p "${GIT_LFS_BIN_DIR}"
  GIT_LFS_TAR_PATH="${TOOLS_DIR}/git-lfs.tar.gz"
  rm -f "${GIT_LFS_TAR_PATH}"
  curl -L "${GIT_LFS_TAR_URL}" -o "${GIT_LFS_TAR_PATH}"
  tar -xzf "${GIT_LFS_TAR_PATH}" -C "${TOOLS_DIR}"
  rm -f "${GIT_LFS_TAR_PATH}"

  GIT_LFS_EXTRACT_DIR="$(find "${TOOLS_DIR}" -maxdepth 1 -type d -name "git-lfs-*" | head -n 1 || true)"
  if [[ -z "${GIT_LFS_EXTRACT_DIR}" || ! -x "${GIT_LFS_EXTRACT_DIR}/git-lfs" ]]; then
    echo "git-lfs binary not found after extraction."
    exit 1
  fi

  cp "${GIT_LFS_EXTRACT_DIR}/git-lfs" "${GIT_LFS_BIN}"
  chmod +x "${GIT_LFS_BIN}"
  export PATH="${GIT_LFS_BIN_DIR}:${PATH}"
  ENV_LINE="export PATH=\"${GIT_LFS_BIN_DIR}:\$PATH\""
  if [[ ! -f "${ENV_FILE}" ]] || ! grep -qF "${ENV_LINE}" "${ENV_FILE}"; then
    echo "${ENV_LINE}" >> "${ENV_FILE}"
  fi
  echo "git-lfs installed at ${GIT_LFS_BIN}. PATH updated."
fi

echo "[3/4] Downloading Multi-SWE-bench dataset (Java)"
if ! command -v git-lfs >/dev/null 2>&1; then
  echo "git-lfs still not available in PATH."
  exit 1
fi

MSB_DIR="${DATA_DIR}/Multi-SWE-bench"
if [[ -d "${MSB_DIR}/.git" ]]; then
  echo "Multi-SWE-bench already exists at ${MSB_DIR}. Skipping download."
else
  git lfs install
  git clone https://huggingface.co/datasets/ByteDance-Seed/Multi-SWE-bench "${MSB_DIR}"
  git -C "${MSB_DIR}" lfs pull
fi

echo "[4/4] Cloning Java repositories"
repos=(
  "https://github.com/alibaba/fastjson2.git"
  "https://github.com/apache/dubbo.git"
  "https://github.com/elastic/logstash.git"
  "https://github.com/FasterXML/jackson-core.git"
  "https://github.com/FasterXML/jackson-databind.git"
  "https://github.com/FasterXML/jackson-dataformat-xml.git"
  "https://github.com/google/gson.git"
  "https://github.com/GoogleContainerTools/jib.git"
  "https://github.com/mockito/mockito.git"
)

for repo_url in "${repos[@]}"; do
  repo_name="$(basename "${repo_url}" .git)"
  target_dir="${REPOS_DIR}/${repo_name}"
  if [[ -d "${target_dir}/.git" ]]; then
    echo "Repository already exists at ${target_dir}. Skipping clone."
  else
    git clone "${repo_url}" "${target_dir}"
  fi
done

echo "[5/5] Downloading experiments dataset from Hugging Face (curl)"
HF_DATASET_BASE="https://huggingface.co/datasets/Azusa434/LLM-Refactoring-Research/resolve/main"
EXP_DIR="${DATA_DIR}"

download_file() {
  local filename="$1"
  local url="${HF_DATASET_BASE}/${filename}"
  local target="${EXP_DIR}/${filename}"
  if [[ -f "${target}" ]]; then
    echo "File already exists: ${target}. Skipping download."
    return
  fi
  echo "Downloading ${filename}"
  curl -fL "${url}" -o "${target}"
}

download_file "java_experiment.zip"
download_file "refactoring_classification.xlsx"
download_file "issue_types.xlsx"

echo "Dataset available at ${EXP_DIR}"

echo "Extracting archives in ${EXP_DIR}"
ARCHIVE_FILES=()
while IFS= read -r -d '' file; do
  ARCHIVE_FILES+=("$file")
done < <(find "${EXP_DIR}" -type f \( \
  -name "*.zip" -o -name "*.tar.gz" -o -name "*.tgz" -o -name "*.tar.xz" -o \
  -name "*.tar.bz2" -o -name "*.tar.zst" \
\) -print0)

for archive in "${ARCHIVE_FILES[@]}"; do
  marker="${archive}.extracted"
  if [[ -f "${marker}" ]]; then
    echo "Archive already extracted: ${archive}"
    continue
  fi
  echo "Extracting ${archive}"
  case "${archive}" in
    *.zip)
      unzip -q "${archive}" -d "$(dirname "${archive}")"
      ;;
    *.tar.gz|*.tgz)
      tar -xzf "${archive}" -C "$(dirname "${archive}")"
      ;;
    *.tar.xz)
      tar -xJf "${archive}" -C "$(dirname "${archive}")"
      ;;
    *.tar.bz2)
      tar -xjf "${archive}" -C "$(dirname "${archive}")"
      ;;
    *.tar.zst)
      tar --zstd -xf "${archive}" -C "$(dirname "${archive}")"
      ;;
  esac
  touch "${marker}"
done

echo "Cleaning up dataset directory"
rm -rf "${EXP_DIR}/__MACOSX" "${EXP_DIR}/.git"
find "${EXP_DIR}" -type f \( -name "*.zip" -o -name "*.tar.gz" -o -name "*.tgz" -o -name "*.tar.xz" -o -name "*.tar.bz2" -o -name "*.tar.zst" -o -name "*.extracted" \) -delete

echo "Dataset download and extraction complete."

echo "Done."
