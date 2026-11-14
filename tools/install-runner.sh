#!/usr/bin/env bash
# Install the GitHub Actions self-hosted runner that deploys animaleyes on this box.
#
# The box already runs robot-computer's runner in ~/actions-runner. Runners on a personal
# account belong to one repo, so animaleyes needs its own: separate directory, separate
# systemd service, label `animaleyes`. The two never pick up each other's jobs.
#
# Outbound only (it long-polls GitHub): no port, no firewall change. It runs as the invoking
# user, who must be able to run docker without sudo, because deploying means
# `docker compose up -d --build` in the live checkout.
#
# With gh installed and logged in, this fetches the short-lived registration token itself:
#
#   tools/install-runner.sh
#
# Otherwise pass one from
# https://github.com/octamagic/animaleyes/settings/actions/runners/new
#
#   tools/install-runner.sh <registration-token>
#
# Safety note: .github/workflows/deploy.yml runs on CI success for pushes to main and on
# manual dispatch, never on `pull_request`. Keep it that way: a self-hosted runner runs the
# workflow from the triggering ref, and this box holds the feeder login and the camera.
set -euo pipefail

REPO="octamagic/animaleyes"
REPO_URL="https://github.com/$REPO"

if ! docker info >/dev/null 2>&1; then
  echo "$USER cannot talk to docker without sudo. Add the group and log in again:" >&2
  echo "  sudo usermod -aG docker $USER" >&2
  exit 1
fi

TOKEN="${1:-}"
if [[ -z "$TOKEN" ]]; then
  if ! command -v gh >/dev/null; then
    echo "No token given and gh is not installed." >&2
    echo "Either install gh, or pass a token from:" >&2
    echo "  $REPO_URL/settings/actions/runners/new" >&2
    exit 64
  fi
  echo "==> requesting a registration token from GitHub"
  if ! TOKEN="$(gh api -X POST "repos/$REPO/actions/runners/registration-token" \
                  --jq .token 2>/dev/null)" || [[ -z "$TOKEN" ]]; then
    echo "Could not get a registration token." >&2
    echo "gh needs admin rights on $REPO. Check: gh auth status" >&2
    exit 1
  fi
fi

# Not ~/actions-runner: that one is robot-computer's.
RUNNER_DIR="$HOME/actions-runner-animaleyes"
# deploy.yml targets `runs-on: [self-hosted, animaleyes]`.
LABELS="animaleyes"
VERSION="2.337.0"

if [[ -d "$RUNNER_DIR" ]]; then
  echo "$RUNNER_DIR already exists."
  echo "To reconfigure: (cd $RUNNER_DIR && sudo ./svc.sh uninstall) && rm -rf $RUNNER_DIR"
  exit 1
fi

mkdir -p "$RUNNER_DIR"
cd "$RUNNER_DIR"

arch="$(uname -m)"
case "$arch" in
  x86_64) pkg_arch=x64 ;;
  aarch64|arm64) pkg_arch=arm64 ;;
  *) echo "unsupported architecture: $arch" >&2; exit 1 ;;
esac

tarball="actions-runner-linux-${pkg_arch}-${VERSION}.tar.gz"
echo "==> downloading $tarball"
curl -fsSL -o "$tarball" \
  "https://github.com/actions/runner/releases/download/v${VERSION}/${tarball}"
tar xzf "$tarball"
rm -f "$tarball"

echo "==> registering with $REPO_URL"
./config.sh \
  --url "$REPO_URL" \
  --token "$TOKEN" \
  --name "$(hostname)-animaleyes" \
  --labels "$LABELS" \
  --work _work \
  --unattended \
  --replace

# systemd, so it survives a reboot. The service name includes the repo, so it doesn't
# collide with robot-computer's.
echo "==> installing the service"
sudo ./svc.sh install "$USER"
sudo ./svc.sh start

echo
echo "==> done"
sudo ./svc.sh status | head -20
echo
echo "Verify it registered:  gh api repos/$REPO/actions/runners --jq '.runners[].name'"
echo "Deploy manually:       gh workflow run deploy.yml -R $REPO"
