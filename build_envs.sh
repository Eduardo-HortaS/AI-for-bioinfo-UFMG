#!/bin/sh
# Build one uv environment per module, plus one for JupyterLab, and register
# each module environment as a Jupyter kernel named after its folder.
#
#   sh build_envs.sh             environments go in each project's .venv
#   sh build_envs.sh /opt/envs   environments go in /opt/envs/<name>
#
# Needs only uv; it installs the Python version each project asks for.
set -eu

root=$(cd "$(dirname "$0")" && pwd)
envs=${1:-}

env_dir() {
    if [ -n "$envs" ]; then echo "$envs/$1"; else echo "$2/.venv"; fi
}

sync() {
    UV_PROJECT_ENVIRONMENT=$1 uv sync --locked --project "$2"
}

jupyter_env=$(env_dir jupyter "$root/jupyter")
sync "$jupyter_env" "$root/jupyter"
mkdir -p "$jupyter_env/share/jupyter/lab/settings" "$jupyter_env/etc/jupyter"
cp "$root/jupyter/overrides.json" "$jupyter_env/share/jupyter/lab/settings/"
# JupyterLab's own environment has no course packages, so do not offer it as
# a kernel: a notebook started on it would fail at the first import.
cp "$root/jupyter/server_config.json" "$jupyter_env/etc/jupyter/jupyter_server_config.json"
rm -rf "$jupyter_env/share/jupyter/kernels/python3"

for dir in "$root"/modules/*/; do
    dir=${dir%/}
    name=$(basename "$dir")
    env=$(env_dir "$name" "$dir")
    sync "$env" "$dir"
    kernel=$jupyter_env/share/jupyter/kernels/$name
    mkdir -p "$kernel"
    printf '{"argv": ["%s", "-m", "ipykernel_launcher", "-f", "{connection_file}"], "display_name": "%s", "language": "python"}\n' \
        "$env/bin/python" "$name" > "$kernel/kernel.json"
done
