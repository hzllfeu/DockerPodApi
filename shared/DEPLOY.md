# Deployment notes

## Target platform

Koyeb GPU pods. Tested on NVIDIA L4 (23 GB VRAM, sm_89). Should also work
on L40S (48 GB, same Ada Lovelace architecture).

## Image registry

Images are pushed to Docker Hub under the `lexiapro` org.

    docker.io/lexiapro/echo-l4:<tag>

Login with a Docker Hub personal access token (PAT), not your account
password:

    docker login -u lexiapro
    # paste PAT at the prompt

Store the PAT in a secret manager (1Password, Vault). Rotate it
regularly. The PAT used during the first build was rotated after the
build session — never reuse a PAT that has been pasted into a chat.

## Building on a Koyeb / Runpod GPU pod

The pod has neither a Docker daemon nor user namespaces, which blocks
`docker build`, `buildah`, `kaniko`, `buildctl`. We use Bazel with
`rules_oci` which constructs OCI tarballs without any chroot or mount.

Build user must be **non-root** (rules_python refuses root):

    useradd -m -s /bin/bash builder
    chown -R builder:builder /data/build

Build commands:

    cd models/<model>/bazel
    bazel build //:image

Outputs an OCI tarball at `bazel-bin/image/`. Inspect it with `tar tvf`.

To push:

    bazel run //:push -- --tag=<tag>

The push target uses `rules_oci`'s `oci_push` rule, which talks to the
registry HTTP API directly (no daemon).

## Deploying on Koyeb

1. Create a service from the Docker Hub image:

       koyeb service create echo-l4 \
         --image docker.io/lexiapro/echo-l4:<tag> \
         --instance-type gpu-nvidia-l4-small \
         --region fra \
         --regions fra \
         --privileged=false \
         --port 8000:http \
         --env ECHO_INTERNAL_SECRET=@echo-internal-secret \
         --env ECHO_WEBHOOK_HMAC_SECRET=@echo-webhook-hmac \
         --volume echo-cache:/cache \
         --autoscaling-min 0 \
         --autoscaling-max 4 \
         --autoscaling-target rps=4

2. Set the secrets first via `koyeb secret create echo-internal-secret`.

3. Mount a persistent volume at `/cache` for async job state across
   restarts. (TRT engines cache was originally intended for `/cache/trt`
   but echo-l4 does not use TRT.)

4. Set the health check to:

       readiness:  HTTP GET /readyz
       liveness:   HTTP GET /healthz
       start grace: 180s   (model loads in ~60s on L4)

## Internal-only access

The pod must not be exposed to the public internet.

- On Koyeb, give the service an **internal** routing tag (so the public
  domain is not bound to it), and call it from the Rust API by its
  internal hostname.
- The shared secret `X-Internal-Secret` is the auth layer. Don't rely on
  network isolation alone.

## Cold start

L4 cold start sequence:

1. Image pull from Docker Hub: 30–90 s depending on Koyeb region.
2. Container start, Python boot, FastAPI init: 5 s.
3. PyTorch model load (DiariZen + SE-DiCoW + Echo Dia weights): 30–45 s.
4. Total cold start: ~60–120 s.

Koyeb's `start grace` should be ≥ 180 s to avoid liveness checks
killing the pod before it's ready. `/readyz` will return 503 during
loading and 200 once active.

## Scale to zero

If the Koyeb service is configured for scale-to-zero, the pod stops
after the configured idle timeout. The next request triggers a full
cold start. Plan accordingly: bursty workloads should keep at least
one warm instance.

## Per-region considerations

For EU/sovereign deployments use `--region fra` (Frankfurt). Other GPU
regions vary by Koyeb plan. L4 is in `fra` and `sin` last we checked.

## Monitoring

- `GET /metrics` exposes Prometheus counters. Scrape from Koyeb metrics
  or the Rust API.
- `GET /readyz` should be polled by Koyeb's load balancer.
- Container stdout/stderr are JSON lines with `{ts, lvl, name, msg}`.

## Upgrading

1. Tag the new image: `lexiapro/echo-l4:v<n>`.
2. Push to Docker Hub.
3. Update the Koyeb service to the new tag. Koyeb will do a rolling
   deploy: spin up a new instance, wait for `/readyz`, then drain the
   old instance.
4. If the schema of the response changed, deploy the Rust API first
   with backwards-compatible parsing.
