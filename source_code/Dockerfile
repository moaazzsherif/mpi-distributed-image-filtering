# SCx61 Week 2 - node image (Week 1 image + MPI stack).
# One image, used for node0 / node1 / node2.
FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive

# --- Week 1 part: SSH + basic tools -------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server openssh-client python3 \
        iputils-ping iproute2 net-tools procps sudo \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd

# Non-root user all lab work runs as (Open MPI refuses to run as root).
RUN useradd -m -s /bin/bash student \
    && echo "student ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/student \
    && mkdir -p /home/student/.ssh \
    && chmod 700 /home/student/.ssh

# One shared SSH keypair for the whole cluster, generated at build time.
RUN ssh-keygen -t ed25519 -N "" -f /home/student/.ssh/id_ed25519 -C "scx61-cluster" \
    && cat /home/student/.ssh/id_ed25519.pub >> /home/student/.ssh/authorized_keys \
    && chmod 600 /home/student/.ssh/authorized_keys /home/student/.ssh/id_ed25519 \
    && chmod 644 /home/student/.ssh/id_ed25519.pub \
    && chown -R student:student /home/student/.ssh

# No host-key prompts between OUR OWN lab nodes (never do this on real hosts).
RUN printf 'Host node0 node1 node2 node3 node4\n\tStrictHostKeyChecking no\n\tUserKnownHostsFile /dev/null\n\tLogLevel ERROR\n' \
    > /home/student/.ssh/config \
    && chmod 600 /home/student/.ssh/config \
    && chown student:student /home/student/.ssh/config

# Same identity for root, so `docker compose exec node0 ssh node1` also works.
RUN mkdir -p /root/.ssh \
    && cp /home/student/.ssh/id_ed25519 /home/student/.ssh/id_ed25519.pub /root/.ssh/ \
    && cp /home/student/.ssh/id_ed25519.pub /root/.ssh/authorized_keys \
    && cp /home/student/.ssh/config /root/.ssh/config \
    && chmod 700 /root/.ssh \
    && chmod 600 /root/.ssh/id_ed25519 /root/.ssh/config /root/.ssh/authorized_keys \
    && chmod 644 /root/.ssh/id_ed25519.pub

RUN ssh-keygen -A

EXPOSE 22

# --- Week 2 part: MPI + Python stack (all from Debian apt) ---------------
#   openmpi-bin    : mpirun (Open MPI 4.1.4 on bookworm)
#   python3-numpy  : image arrays
#   python3-pil    : writes the .png preview
#   python3-mpi4py : Python bindings for MPI
RUN apt-get update && apt-get install -y --no-install-recommends \
        openmpi-bin python3-numpy python3-pil python3-mpi4py \
    && rm -rf /var/lib/apt/lists/*

# Shared work folder (compose bind-mounts your ../work over it).
RUN mkdir -p /home/student/work && chown student:student /home/student/work

# entrypoint.sh fixes the ownership of the shared folder, then starts sshd.
# (sed strips Windows line endings if the file was ever saved with CRLF.)
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh && chmod 755 /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/usr/sbin/sshd", "-D", "-e"]
