#!/bin/sh
set -eu
exec ssh -N -T -p 32781 -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 127.0.0.1:8768:127.0.0.1:8767 root@10.255.1.213
