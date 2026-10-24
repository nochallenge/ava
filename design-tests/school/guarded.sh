#!/usr/bin/env bash
# TESTING / DESIGN ONLY.
# Run one Ava command behind a RAM watchdog: if the WHOLE MACHINE's free physical memory
# drops under FLOOR_GB, the run is killed (v9's rule: guard the machine, not the process).
#   usage: FLOOR_GB=3 guarded.sh <out-file> <command...>
out="$1"; shift
floor="${FLOOR_GB:-3}"
free_gb() { powershell -NoProfile -Command "[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB,2)"; }
"$@" > "$out" 2>&1 &
pid=$!
low=99
while kill -0 "$pid" 2>/dev/null; do
  f=$(free_gb)
  awk -v f="$f" -v l="$low" 'BEGIN{exit !(f<l)}' && low="$f"
  if awk -v f="$f" -v fl="$floor" 'BEGIN{exit !(f<fl)}'; then
    powershell -NoProfile -Command "Get-Process compose -ErrorAction SilentlyContinue | Stop-Process -Force"
    echo "!! KILLED: free RAM $f GB under floor $floor GB" | tee -a "$out"
    break
  fi
  sleep 3
done
wait "$pid" 2>/dev/null
echo "done: $out  (lowest free RAM seen: $low GB)"
