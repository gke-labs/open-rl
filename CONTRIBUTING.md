# Contributing Guide

* [Ways to Contribute](#ways-to-contribute)
* [Find an Issue](#find-an-issue)
* [Ask for Help](#ask-for-help)
* [Development Environment Setup](#development-environment-setup)
* [Pull Request Lifecycle](#pull-request-lifecycle)
* [Sign Your Commits](#sign-your-commits)
* [Pull Request Checklist](#pull-request-checklist)

Welcome! We are glad that you want to contribute to OpenRL. 💖

As you get started, you are in the best position to give us feedback on areas of the project
that we need help with, including:

* Problems found while setting up a new development environment
* Gaps in the Quick Start or documentation
* Bugs in our automation scripts

If anything doesn't make sense, or doesn't work when you run it, please
[open an issue](https://github.com/gke-labs/open-rl/issues/new) and let us know.

Please note that this project has a [Code of Conduct](CODE_OF_CONDUCT.md) that all
contributors are expected to follow.

## Ways to Contribute

We welcome many different types of contributions, including:

* New features
* Bug fixes
* Builds and CI/CD
* Documentation, guides, and example recipes
* Issue triage
* Answering questions from other users
* Release management

Not everything happens through a GitHub pull request. If you would like to discuss an idea
before writing code, open a [GitHub issue](https://github.com/gke-labs/open-rl/issues) and
let's talk it through.

<!-- TODO: OpenRL does not yet have a public developer meeting, mailing list, or chat
     channel. Once these exist, add them here and in the README. -->

## Find an Issue

Issues labelled [`good first issue`](https://github.com/gke-labs/open-rl/labels/good%20first%20issue)
have extra context to help you make your first contribution.
[`help wanted`](https://github.com/gke-labs/open-rl/labels/help%20wanted) issues are suitable
for anyone who isn't a core maintainer and are a good next step after your first pull request.

Sometimes there won't be any issues with these labels. That's OK — there is likely still
something for you to work on. If you want to contribute but can't find a suitable issue, open
an issue describing what you would like to work on and a maintainer will help you scope it.

Once you find an issue you'd like to work on, please comment on it saying so, to avoid
duplicate effort.

## Ask for Help

The best way to reach us with a question is to comment on the relevant GitHub issue or pull
request. If there isn't one yet, open a new issue.

## Development Environment Setup

OpenRL uses [`uv`](https://docs.astral.sh/uv/) for environment isolation. There are two
primary environments:

* **Server** (`src/server`) — the gateway server and worker controllers.
* **Client / examples** (`examples`) — recipes, client SDK compatibility checks, and
  end-to-end integration test scripts.

Most tasks are driven through the `Makefile`, which invokes `uv` under the hood. Make sure
`uv` is on your `PATH` (typically `~/.local/bin`):

```bash
export PATH=$PATH:$HOME/.local/bin
```

Run `make help` to see the available targets and their default knobs.

### Running the server locally

```bash
make server                                   # defaults to SAMPLING_BACKEND=torch on port 9003
make server SAMPLING_BACKEND=vllm             # use vLLM for sampling
```

### Running tests

```bash
make test                                     # fast unit tests
make test piglatin                            # pig-latin example end-to-end tests
```

End-to-end GPU integration tests boot a real backend and run actual SFT/RL training. They
require a GPU and use the `gpu` extra:

```bash
make test e2e tiny-lora                       # minimal LoRA overfit test
make test e2e tiny-fft                        # full fine-tuning (requires redis-server)
make test e2e tiny-rl                         # sample -> reward -> train loop
```

Most contributors develop on a machine without a local NVIDIA GPU and use a remote GPU VM as
the test target. `make push-vm REMOTE_HOST=<host>` and `make pull-vm REMOTE_HOST=<host>` sync
your workspace to and from that host. On a fresh GPU VM you may also need `redis-server` and
`python3-dev` installed.

### Linting and formatting

The project uses [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting, and
CI enforces both on every pull request:

```bash
make lint
make fmt
```

You can have this run automatically on every commit by installing the pre-commit hooks:

```bash
pre-commit install
```

## Pull Request Lifecycle

1. Open an issue first for anything larger than a small fix, so the approach can be agreed on
   before you invest time in it.
2. Fork the repository and create a branch for your change.
3. Make your change, including tests and documentation updates.
4. Run `make lint` and `make test` locally.
5. Open a pull request against `main`. Describe what changed and why, and link the issue it
   addresses.
6. A maintainer will review your pull request. All submissions, including those from project
   members, require review. Expect a first response within a few business days; feel free to
   ping the pull request if it goes quiet.
7. Address review feedback by pushing additional commits to your branch. Once approved and
   with CI green, a maintainer will merge it.

## Sign Your Commits

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA). You (or your
employer) retain the copyright to your contribution; the CLA simply gives us permission to use
and redistribute your contributions as part of the project.

If you or your current employer have already signed the Google CLA (even if it was for a
different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to sign a new one.

## Pull Request Checklist

When you submit your pull request, our automated systems will run checks on your code. We
require that your pull request passes these checks. Before submitting, please check the
following locally:

* [ ] You have signed the [CLA](https://cla.developers.google.com/).
* [ ] `make lint` passes (Ruff lint and format checks).
* [ ] `make test` passes.
* [ ] New code has tests covering it.
* [ ] Documentation under `docs/` or `examples/` is updated if behavior changed.
* [ ] The pull request description explains what changed and why, and links the related issue.
* [ ] Commits are scoped and have clear messages.
