#!/bin/bash
# Resource watchdog: logs every 60s; alerts on pressure; emergency-stops
# afarbin's newest training if memory becomes critical. Reboot-proof home.
L=/storage/afarbin/jetreg/logs
while true; do
  ts=$(date "+%F %T")
  read -r _ mt ma <<< "$(grep -E "MemTotal|MemAvailable" /proc/meminfo | awk "{print \$2}" | tr "\n" " " | sed "s/^/x /")"
  ma_gb=$((ma / 1048576))
  load=$(cut -d" " -f1 /proc/loadavg)
  d_stor=$(df --output=pcent /storage 2>/dev/null | tail -1 | tr -dc 0-9)
  d_root=$(df --output=pcent / | tail -1 | tr -dc 0-9)
  echo "$ts load=$load mem_avail_gb=$ma_gb storage=${d_stor}% root=${d_root}%" >> $L/resources.log
  alert=""
  [ "$ma_gb" -lt 20 ] && alert="mem_avail=${ma_gb}GB"
  awk "BEGIN{exit !($load > 80)}" && alert="$alert load=$load"
  [ "${d_root:-0}" -gt 90 ] && alert="$alert root_disk=${d_root}%"
  [ "${d_stor:-0}" -gt 95 ] && alert="$alert storage=${d_stor}%"
  if [ -n "$alert" ]; then
    echo "$ts RESOURCE_ALERT $alert" >> $L/alerts.log
  fi
  # soft throttle: pause newest training below 12GB, resume above 25GB
  if [ "$ma_gb" -lt 12 ]; then
    v=$(pgrep -u afarbin -f "dfm.jetreg.train" -n)
    if [ -n "$v" ] && ! grep -q T "/proc/$v/stat" 2>/dev/null; then
      kill -STOP "$v" && echo "$ts PAUSED pid=$v (mem_avail=${ma_gb}GB)" >> $L/alerts.log
    fi
  elif [ "$ma_gb" -gt 25 ]; then
    for v in $(pgrep -u afarbin -f "dfm.jetreg.train"); do
      state=$(awk "{print \$3}" "/proc/$v/stat" 2>/dev/null)
      if [ "$state" = "T" ]; then
        kill -CONT "$v" && echo "$ts RESUMED pid=$v (mem_avail=${ma_gb}GB)" >> $L/alerts.log
        break
      fi
    done
  fi
  # emergency valve: critical memory -> stop newest of MY trainings
  if [ "$ma_gb" -lt 8 ]; then
    victim=$(pgrep -u afarbin -f "dfm.jetreg.train" -n)
    if [ -n "$victim" ]; then
      kill "$victim"
      echo "$ts EMERGENCY_KILL pid=$victim (mem_avail=${ma_gb}GB)" >> $L/alerts.log
    fi
  fi
  sleep 60
done
