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
- `ghcr.io/gke-labs/open-rl/api-server:<tag>`
- `ghcr.io/gke-labs/open-rl/client:<tag>` (the e2e client, `src/server/Dockerfile.client`)
- `ghcr.io/gke-labs/open-rl/scheduler:<tag>`

Assets, attached to the GitHub Release:

| Asset | Rendered from |
| --- | --- |
| `openrl-lora.yaml` | `k8s/deploy/lora` |
| `openrl-fft.yaml` | `k8s/deploy/fft` |
| `openrl-skill.tar.gz` | `skills/openrl`, the agent skill, stamped with the tag |
| `checksums.sha256` | the files above |

Every bundle has its images pinned to the release tag. The API server reports the tag as
`openrl.version` in `get_server_capabilities` and links the skill asset as `openrl.skill_url`.

## Cut a release

1. **Dry run locally.** Requires `kustomize`. Substitute the version you are about to cut.

   ```bash
   make release-bundle VERSION=v0.0.2
   grep 'image:' dist/*.yaml
   ```

   `dist/` is gitignored, and this is the same target CI runs. The images it pins do not exist yet;
   the point is to confirm the bundles render and every OpenRL image reads the new tag.
   `tar -tzf dist/openrl-skill.tar.gz` should list `openrl/SKILL.md` and `openrl/references/`.

   Re-read `skills/openrl/references/lifecycle.md` and `runbook.md` against this release. The
   settings come from the server's own OpenAPI document and cannot drift; the durations and
   failure modes are hand-written, and an agent will trust them.

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
   kubectl apply --server-side -f https://github.com/gke-labs/open-rl/releases/download/v0.0.2/openrl-lora.yaml
   kubectl -n openrl-system get pods -o jsonpath='{..image}'
   ```

   Server-side apply is required: the Workload CRD is larger than the 256 KiB
   `last-applied-configuration` annotation a client-side apply would write.

   Every OpenRL image should read `:v0.0.2` and none should read `:latest`.

5. **Edit the release notes** if the generated changelog needs a summary or upgrade instructions.
