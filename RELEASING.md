# Releasing

How maintainers cut an OpenRL release. Everything here is driven by a git tag; there is no
manual upload step.

## Versioning

Releases are semantic versions with a `v` prefix — `v0.1.0`, `v0.2.0`. Pushing a `v*` tag is what
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

**Asset names carry no version, deliberately.** That is what makes
`https://github.com/gke-labs/open-rl/releases/latest/download/openrl-distributed-shared.yaml`
resolve — GitHub only serves `latest/download/<name>` when `<name>` is stable across releases.
Renaming an asset to include the version breaks every install command in the docs. If a bundle is
ever added or renamed, update `Makefile`'s `release-bundle` target and the docs in the same change.

Recipe overlays (`examples/text-to-sql`, the autoresearch recipes) are not shipped as assets. Users
render them from a tagged checkout with `make render`.

## Cut a release

1. **Dry run locally.** Requires `kustomize`. Substitute the version you are about to cut.

   ```bash
   make release-bundle VERSION=v0.2.0
   grep 'image:' dist/*.yaml
   ```

   `dist/` is gitignored, and this is the same target CI runs. The images it pins do not exist yet;
   the point is to confirm the bundles render and every OpenRL image reads the new tag.

   To check a recipe overlay, which CI does not exercise:

   ```bash
   make render OVERLAY=examples/text-to-sql VERSION=v0.2.0 | grep 'image:'
   ```

2. **Tag `main`** at the commit you want to release, and push the tag.

   ```bash
   git checkout main && git pull
   git tag v0.2.0
   git push origin v0.2.0
   ```

3. **Watch CI.** The tag push runs `build-and-push.yml`. Its `build-and-publish` job pushes the
   images; the `release` job then renders the bundles and calls `gh release create` with
   `--generate-notes`. The release job only runs once the images are published, so a build failure
   means no release is created.

4. **Verify on a clean cluster.**

   ```bash
   kubectl apply -f https://github.com/gke-labs/open-rl/releases/download/v0.2.0/openrl-distributed-shared.yaml
   kubectl get pods -o jsonpath='{..image}'
   ```

   Every OpenRL image should read `:v0.2.0` and none should read `:latest`.

5. **Edit the release notes** if the generated changelog needs a summary or upgrade instructions,
   and record the dependency versions below.

## Prereleases

Not currently used, and the tooling does not support them as-is. `docker/metadata-action` is
configured with `type=semver,pattern=v{{version}}`, which it honours for release tags but ignores
for prerelease tags — `v0.1.0-rc.1` publishes the image tag `0.1.0-rc.1`, without the `v`. The
release bundle pins `${GITHUB_REF_NAME}` (`v0.1.0-rc.1`), so the manifests would point at an image
tag that was never published and the deploy would fail with `ImagePullBackOff`.

To enable prereleases, switch the pattern to `type=semver,pattern={{raw}}`, which emits the git tag
verbatim in both cases, and confirm the resulting tag set before tagging.

## Backports

`/releases/latest/` is decided by release creation date, not by version order. Tagging `v0.1.1`
after `v0.2.0` has shipped makes `v0.1.1` the "latest" release and silently downgrades every
`latest/download/` link in the docs and the README badge.

CI has no way to know a tag is a backport, so it always publishes a normal release. Fix it by hand
straight after the run finishes:

```bash
gh release edit v0.1.1 --prerelease          # excludes it from /releases/latest
gh release view --json tagName               # confirm latest is back on v0.2.0
```

The image tags are unaffected — `metadata-action` reads the git tag, not the release's prerelease
flag, so `v0.1.1` still publishes `v0.1.1` images and the bundle's pin resolves.

## Compatibility

The versions each release was built and tested against. Clients must run from the same tag as the
deployed gateway: the Tinker SDK pin lives in `examples/pyproject.toml` and moves independently on
`main`, so a client from `main` can disagree with a released gateway about the API.

| Release | Tinker SDK | tinker-cookbook | vLLM | torch | Python |
| --- | --- | --- | --- | --- | --- |
| `v0.1.0` | 0.22.7 | 0.4.2 | 0.20.0 | 2.11.0+cu129 | 3.12 |

Add a row for every release. The sources of truth are `examples/pyproject.toml` (Tinker SDK,
cookbook), `pyproject.toml` (vLLM, Python) and `uv.lock` (resolved torch build).
