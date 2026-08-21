# Releasing

How maintainers cut an OpenRL release. Everything here is driven by a git tag; there is no
manual upload step.

## Versioning

Releases are semantic versions with a `v` prefix — `v0.0.1`, `v0.0.2`. Pushing a `v*` tag is what
produces a release; nothing else does.

`main` keeps publishing `latest` on every push. `latest` is a moving target and is not a release.

## What a release contains

Images, published to GHCR by `.github/workflows/build-and-push.yml` and tagged with the git tag
verbatim:

- `ghcr.io/gke-labs/open-rl/server:<tag>`
- `ghcr.io/gke-labs/open-rl/gateway:<tag>`
- `ghcr.io/gke-labs/open-rl/client:<tag>`

Assets, attached to the GitHub Release:

| Asset | Rendered from |
| --- | --- |
| `openrl-distributed-shared.yaml` | `k8s/deploy/distributed-shared` |
| `openrl-distributed-lustre.yaml` | `k8s/deploy/distributed-lustre` |
| `checksums.sha256` | the two YAML files above |

Both bundles have their images pinned to the release tag.

## Cut a release

1. **Dry run locally.** Requires `kustomize`. Substitute the version you are about to cut.

   ```bash
   make release-bundle VERSION=v0.0.2
   grep 'image:' dist/*.yaml
   ```

   `dist/` is gitignored, and this is the same target CI runs. The images it pins do not exist yet;
   the point is to confirm the bundles render and every OpenRL image reads the new tag.

   To check a recipe overlay, which CI does not exercise:

   ```bash
   make render OVERLAY=examples/text-to-sql VERSION=v0.0.2 | grep 'image:'
   ```

2. **Tag `main`** at the commit you want to release, and push the tag.

   ```bash
   git checkout main && git pull
   git tag v0.0.2
   git push origin v0.0.2
   ```

3. **Watch CI.** The tag push runs `build-and-push.yml`. Its `build-and-publish` job pushes the
   images; the `release` job then renders the bundles and calls `gh release create` with
   `--generate-notes`. The release job only runs once the images are published, so a build failure
   means no release is created.

4. **Verify on a clean cluster.**

   ```bash
   kubectl apply -f https://github.com/gke-labs/open-rl/releases/download/v0.0.2/openrl-distributed-shared.yaml
   kubectl get pods -o jsonpath='{..image}'
   ```

   Every OpenRL image should read `:v0.0.2` and none should read `:latest`.

5. **Edit the release notes** if the generated changelog needs a summary or upgrade instructions.
