#!/usr/bin/env bash

max_size=$1
args=("$@")

list=()
for i in $(seq 1 $#); do
  list+=("${args[i]}")
done

groups=()
group=()
i=0
for a in "${list[@]}"; do
  group+=($a)
  ((i++))
  if [[ "$i" -ge "$max_size" ]]; then
    groups+=("${group[*]}")
    i=0
    group=()
  fi
done

groups=$( IFS=$'\n'; echo "${groups[*]}" )
echo "$groups"
