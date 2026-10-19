#!/usr/bin/env bash
# Runs as root on every container start, before sshd.
# Job: make sure the shared work folder is writable by `student`.
# The chown is allowed to fail (some Windows/Mac mounts refuse it) - the
# container must still start either way.
set -eu

if [ -d /home/student/work ]; then
    chown -R student:student /home/student/work \
        || echo "WARN: could not chown /home/student/work (continuing)"
fi

exec "$@"
