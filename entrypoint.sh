#!/usr/bin/env bash
# Size every math library's thread pool from the container's cgroup quota. numpy BLAS and
# OpenMP read the HOST's core count, and an 8 vCPU Railway container on a 64+ core host
# oversubscribes (measured 5x on torch, 78x on llama.cpp elsewhere in this repo).
set -euo pipefail
log() { echo "[entrypoint] $*"; }

cpu_quota() {
    local quota period
    if [ -r /sys/fs/cgroup/cpu.max ]; then
        read -r quota period < /sys/fs/cgroup/cpu.max || return 1
        [ "$quota" = "max" ] && return 1
        echo $(( (quota + period - 1) / period )); return 0
    fi
    if [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ] && [ -r /sys/fs/cgroup/cpu/cpu.cfs_period_us ]; then
        quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
        [ "$quota" -le 0 ] && return 1
        echo $(( (quota + period - 1) / period )); return 0
    fi
    return 1
}

VISIBLE="$(nproc)"
if CPUS="$(cpu_quota)"; then :; else CPUS="$VISIBLE"; log "no cgroup quota, using visible cores"; fi
[ "$CPUS" -lt 1 ] && CPUS=1

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPUS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CPUS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$CPUS}"
export ASR_THREADS="${ASR_THREADS:-$CPUS}"
log "cpu: visible=$VISIBLE quota=$CPUS -> threads=$ASR_THREADS PORT=${PORT:-unset}"
exec "$@"
