#!/bin/sh
# Opt in only this checkout. Never replace another hook chain.
set -eu
cd "$(dirname "$0")/.."
root=$(git rev-parse --show-toplevel)
[ "$root" = "$(pwd -P)" ] || {
    echo "install-hooks: run from a standalone recipe checkout" >&2
    exit 1
}
local_path=$(git config --local --get-all core.hooksPath || :)
global_path=$(git config --global --get-all core.hooksPath || :)
effective_path=$(git config --get-all core.hooksPath || :)
if [ -n "$global_path" ]; then
    echo "install-hooks: global hook routing exists; refusing to shadow it. Review integration manually." >&2
    exit 1
fi
if [ -n "$local_path" ] && [ "$local_path" != .githooks ]; then
    echo "install-hooks: different local hook routing exists; refusing to replace it." >&2
    exit 1
fi
if [ -n "$effective_path" ] && [ "$effective_path" != .githooks ]; then
    echo "install-hooks: other hook routing exists; refusing to replace it." >&2
    exit 1
fi
if [ -z "$local_path" ]; then
    hooks=$(git rev-parse --git-path hooks)
    for hook in "$hooks"/*; do
        [ -e "$hook" ] || [ -L "$hook" ] || continue
        case "$hook" in *.sample) continue ;; esac
        echo "install-hooks: existing default hooks found; refusing to bypass them." >&2
        exit 1
    done
fi
for hook in .githooks/pre-commit .githooks/commit-msg; do
    [ -f "$hook" ] && [ -x "$hook" ] || {
        echo "install-hooks: tracked hooks must exist and be executable" >&2
        exit 1
    }
done
[ -f .gitleaks.toml ] || {
    echo "install-hooks: missing .gitleaks.toml" >&2
    exit 1
}
git config --local core.hooksPath .githooks
echo "Installed repository-local staged-content and commit-message hooks."
if ! command -v gitleaks >/dev/null 2>&1; then
    echo "WARNING: gitleaks is missing; hooks pass without scanning until it is installed." >&2
fi
