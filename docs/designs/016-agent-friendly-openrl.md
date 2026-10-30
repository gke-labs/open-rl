# Design Doc 016: Making OpenRL agent friendly

**Status:** Draft
**Author:** Sunil Arora
**Last updated:** 2026-10-12

## 1. Summary

Most OpenRL users will drive it through an AI coding agent: the agent reads
the docs, writes the training loop, runs it, and debugs it. An agent is a
user that cannot ask a colleague, so everything it needs has to be
discoverable from the endpoint it is connected to and from the files in the
repository.

Today it is not. During the v0.0.6 release test (`openrl-release-test/v0.0.6/FINDINGS.md`),
an agent following `docs/getting-started.md` had to read the server source,
`kubectl get workloads`, and `kubectl exec` into Redis to find out why a run
hung without an error. This document proposes:

1. **A self-describing endpoint.** `get_server_capabilities` reports the
   release, the `openrl.*` settings with their types and defaults, the
   worker launcher mode, and limits. The OpenAPI document gets a real title
   and version.
2. **Status without kubectl.** One read-only endpoint,
   `GET /api/v1/openrl/models/{model_id}`, returns what the API server
   already knows about a model: its workers, each in one of five
   launcher-independent states with a human-readable detail, whether its
   session is live, and a one-line summary. Fleet-wide and list views are
   deferred until tenancy is designed (section 10).
3. **Every failure reaches the client, with the fix in the message.** When
   workers are released, pending futures fail instead of polling forever.
   While a worker is starting, `try_again` says what it is waiting on.
4. **A reference written for agents.** A skill (`skills/openrl/`) with
   worker states and normal durations and a symptom-first runbook. It
   carries no copy of the settings; the server describes them once, in
   its OpenAPI document, and the skill sends agents there.
5. **A doctor script and parseable examples.** `openrl-doctor` verifies a
   deployment in under a minute; the tiny examples flush output and exit
   non-zero on failure.

All API changes are additive. No Tinker wire path changes shape.

## 2. Evidence

What the agent needed during the v0.0.6 test, and where it found it:

| Question | Where the answer was | Where it should be |
|---|---|---|
| Why has the run shown nothing for 8 minutes? | `kubectl get workloads`; then `session_registry.py`, `api_server.py`; then `redis-cli` inside the Redis pod | The client's `try_again` response, or an error |
| What are the `openrl.*` settings and their values? | Release notes and `class Settings` in `api_server.py` | `get_server_capabilities` and the skill |
| Does this server support gemma? | Unclear: `get_server_capabilities` listed only `Qwen/Qwen2.5-0.5B`, the guide trains gemma | `get_server_capabilities` |
| Is the trainer pulling an image or stuck? | `kubectl -n openrl-system get pods` | The model view |
| Which release is deployed? | `kubectl get deploy -o jsonpath=...` | `get_server_capabilities`, `/openapi.json` |
| Is the client hung or buffering? | Guess; `ps` and `lsof` | Flushed output |

The hang itself (FINDINGS F1): a port-forward dropped, heartbeats failed
for longer than the 120 s session TTL, the reaper released the workers,
and the heartbeat then recreated the session key without its owner
binding. The `create_model` future was never resolved, so the client
polled `try_again` indefinitely. The API server log filters heartbeats and
`retrieve_future`, so nothing recorded the revival. Issues #246 and #247
describe the same class of failure.

## 3. Goals and non-goals

Goals:

- An agent with only the HTTP endpoint can learn the deployment's version,
  settings, limits and current state, and gets an actionable error for
  every failure the API server can detect.
- An agent with the repository checked out finds a reference it can load
  as a skill, kept honest by tests.
- A human reading the same endpoints and docs is served equally well.
- No change to Tinker SDK compatibility. Existing clients, including
  `tinker-cookbook`, keep working unchanged.

Non-goals for this round:

- An MCP server. REST plus OpenAPI plus a skill is enough for every agent
  that can call HTTP. Revisit when there are operations that do not map
  to the API, such as streaming worker logs.
- Authentication or per-user authorization. The model view exposes node
  and Workload names to whoever holds the model id, which is the same
  capability every Tinker call on that model already requires. Views that
  would list other clients' models or the fleet are deferred for exactly
  this reason (section 10).
- Changing the session TTL or the reaper policy (#247). This document makes
  the consequences visible; the policy is a separate decision.
- A log streaming endpoint.

## 4. Principles

1. **The endpoint is the documentation.** If an agent would have to read
   source to learn a fact about the running server, the server should
   report it.
2. **Silence is a bug.** A request that cannot be served fails with a
   message that names the cause and the next step. Waiting is fine;
   waiting without saying why is not.
3. **Status needs no cluster access.** Most agents run on a laptop behind
   a port-forward. Everything `kubectl get workloads` shows about a
   model, the API server can show too. The API server reads the Workload,
   the object it created; the pod behind it is the scheduler's.
4. **Docs for agents are references, not narratives.** Tables, exact
   commands, normal durations, and symptom-first runbooks.
   `getting-started.md` stays a narrative for humans.
5. **Facts in docs are tested or fetched.** A settings table that drifts
   from `class Settings` is worse than none; the settings are therefore
   published by the server in one place and not copied into docs.

## 5. Design

### 5.1 Self-describing endpoint

`GET /api/v1/get_server_capabilities` keeps its current fields and gains an
`openrl` object. The SDK's response models ignore unknown fields
(`extra="ignore"` in `tinker/_models.py`), so this is safe for every SDK
version OpenRL supports.

```json
{
  "supported_models": [{"model_name": "Qwen/Qwen2.5-0.5B"}],
  "default_model": "Qwen/Qwen2.5-0.5B",
  "single_process": false,
  "openrl": {
    "version": "v0.0.6",
    "worker_launcher": "scheduler",
    "sampler_backend": "vllm",
    "fft_enabled": true,
    "models": {
      "any_huggingface_model": true,
      "default": "Qwen/Qwen2.5-0.5B",
      "examples": ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen3-0.6B", "google/gemma-4-e2b", "Qwen/Qwen3-8B"]
    },
    "settings_schema_url": "/openapi.json#/components/schemas/OpenRLSettings",
    "settings_howto": "Pass as user_metadata on create_session or create_model, or as TINKER_TAGS=openrl.key=value,...",
    "limits": {"vllm_max_model_len": 8192, "train_token_budget": 65536,
               "session_ttl_seconds": 120, "session_reap_interval_seconds": 30},
    "skill_url": "https://github.com/gke-labs/open-rl/releases/download/v0.0.6/openrl-skill.tar.gz",
    "getting_started_url": "https://github.com/gke-labs/open-rl/blob/main/docs/getting-started.md"
  }
}
```

- The settings are described in one place: `components.schemas.OpenRLSettings`
  in `/openapi.json`, pydantic's JSON Schema for `class Settings` as is,
  rendered for humans at `/docs`. Capabilities carry only
  `settings_schema_url`, so an agent that starts there knows where to
  look and no second copy can disagree. The `openrl.` names are the
  fields' aliases, set by an alias generator on the model, so pydantic
  validates `user_metadata` keys as the client wrote them, names them
  that way in 400s, and emits them in the schema; nothing adds or strips
  the prefix by hand. The `user_metadata` fields of the request models
  point at the schema. The skill carries no settings table; it says how
  to read the schema and how to pass settings.
- `worker_launcher` is one of `single_process`, `process`, `scheduler`,
  or `none` (standing worker deployments the server does not launch),
  derived from the same environment the server already reads. Several
  settings are refused outside `scheduler` mode; the client can see why.
  The model view's summary uses it too, so a single-process server does
  not report "no workers yet" for a trainer that runs in-process.
- `models`: in `scheduler` and `process` modes any Hugging Face model
  works and `BASE_MODEL` is only a default. `supported_models` stays as
  the SDK expects it; `models.any_huggingface_model` says the truth, and
  `examples` lists models the examples and tests use.
- `version` comes from a build argument `OPEN_RL_VERSION` baked into the
  image and exposed as an environment variable. `/app/VERSION` inside the
  v0.0.6 image reads `0.6.18`, the internal counter, so it cannot serve as
  the release version. The release workflow passes `GITHUB_REF_NAME`; dev
  builds get the short commit.
- `limits` reports the environment values the server already forwards to
  workers (`VLLM_MAX_MODEL_LEN`, `OPEN_RL_TRAIN_TOKEN_BUDGET`) and the
  session TTL.

The FastAPI app metadata changes from `Open-RL Server MVP 0.1.0` to
`OpenRL API server`, the release version, and a description linking the
repository and the skill. `/openapi.json` and `/docs` already exist; this
makes them worth reading. Endpoint docstrings are already good; a few
request models gain field descriptions.

### 5.2 The model view

One read-only endpoint, JSON, not used by the Tinker SDK:

| Endpoint | Returns |
|---|---|
| `GET /api/v1/openrl/models/{model_id}` | the model's metadata and resolved `settings`, `session_live`, `shares_workers`, one entry per worker (below), `workers_error`, and `summary`, one line saying what to know first |

```json
{
  "model_id": "df85...", "base_model": "google/gemma-4-e2b", "fine_tuning_type": "lora",
  "status": "active", "created_at": 1760227200.0, "shares_workers": true, "session_live": true,
  "settings": {"openrl.fine_tuning_type": "lora", "openrl.weight_sync": "delta", "...": "..."},
  "workers": [
    {"role": "trainer", "state": "ready",
     "detail": "Running (PodCreated) on gke-open-rl-dra-l4-trainer-2x-ed601d8b-lvw8",
     "workload": "lora-google-gemma-4-e2b-0-trainer", "node": "gke-...-lvw8", "age_seconds": 252,
     "summary": "trainer ready: Running (PodCreated) on gke-...-lvw8, 252s"},
    {"role": "sampler", "state": "starting",
     "detail": "Placing (PodCreated) on gke-...-2208, pod ContainerCreating",
     "workload": "lora-google-gemma-4-e2b-0-sampler", "node": "gke-...-2208", "age_seconds": 48,
     "summary": "sampler starting: Placing (PodCreated) on gke-...-2208, pod ContainerCreating, 48s"}
  ],
  "summary": "trainer ready: ...; sampler starting: ..."
}
```

**Scope.** The view is scoped to one model, and the response contains
nothing that is not that model's: no owner key, no session ids, no pod
names or pids, no other models. A model id is a random UUID the server
minted for the client that created it, so knowing it is the capability,
exactly as for `forward_backward` or `sample` on that model. This is why
the view can ship before a tenancy design and the fleet view cannot.

**State vocabulary.** A client branches on `state`, one of five values
that every launcher reports:

| `state` | Meaning | Scheduler mode | Process mode |
|---|---|---|---|
| `waiting_for_placement` | no accelerator assigned yet | Workload without a `nodeName` (`Pending`, early `Placing`) | never |
| `starting` | placed; pulling the image, mounting volumes, loading the model | `nodeName` set, phase not yet `Running`; or `Running` but the worker has not announced itself | process alive, worker not announced |
| `ready` | serving requests | Workload `Running`, and for a sampler the pod `Ready`, which its readiness probe allows only while vLLM serves the weights the pool expects; a trainer announces itself with `open_rl:trainer_ready:<runtime>` once it holds a loaded model and drains its queue | process alive, and a trainer announced |
| `failed` | exited with an error, or the request cannot be satisfied | Workload `Failed` | exit code not zero |
| `stopped` | exited normally, or being released | Workload `Succeeded`, or a deletion timestamp | exit code zero |

The announcement is a TTL key the trainer refreshes each minute and
deletes on exit, so a dead process stops looking ready on its own. It
comes from the request processor asking the worker whether it
holds a model (`is_ready()`), so a LoRA trainer is `starting` during its
preload and a full fine-tuning trainer until its `create_model` has
loaded the weights.

The scheduler's phase names and reasons reach the client only inside `detail`, a free-text string documented as
for humans and not a contract, the same split Kubernetes conditions make
between `reason` and `message`. Renaming a Workload phase changes a
string an agent shows to a user; it does not break a branch. The field is
`state`, not `status`, because the model metadata already has a `status`
of `active` or `deleted`.

The mapping lives in one function per launcher (`worker_state` in
`scheduler_worker_manager.py`; the process launcher's is three lines) and
is table-tested. The node assignment is the placement boundary; the
worker's own announcement is the readiness boundary.

**Sources and cost.** The Workloads this API server created, read with
the RBAC it already has for them and cached for 5 seconds since the
examples poll the view once a minute. No pod reads: the API server asks
the scheduler for workers through the Workload and learns about them the
same way, so the boundary between the two stays where it is. What the
scheduler puts in Workload status is what the client can see. Today that
is pod phase mapped onto Workload phase, so an image pull shows as
`starting` with `Placing (PodCreated)` and a crash-looping container as
`ready` until the pod fails; surfacing container waiting reasons in
Workload `reason` is a scheduler change (section 10). No RBAC change, no
watches. `session_live` comes
from `session_ids` on the model's metadata, which `bind_session` appends
to for every model in every launcher mode; it is the model's own
sessions, not the owner's, because on a shared LoRA runtime another
client's session keeps the workers up but says nothing about this one.
Keeping it on the metadata means it has the metadata's lifetime and
there is no extra key to clean up. When the cluster
API cannot be read the view still answers from the store, with `workers`
empty and `workers_error` set, rather than failing a GET that clients
poll. Without a worker manager (`single_process`, `none`) the summary
says where the workers are instead of reporting none.

### 5.3 Failures reach the client

Three changes in the API server, each covered by a unit test in
`tests/test_session_lifecycle.py`:

**Releasing workers fails their pending futures.** `reap_owner` and
`delete_model` already know the owner's model ids. Before deleting the
Workloads, every unresolved future for those models is set to
`RequestFailedResponse` with a message like:

> workers for model `df85...` were released: no live session for 120 s
> (last heartbeat 2026-10-12T00:14:51Z). Create a new model, or resume
> from `tinker://df85.../weights/step-9` if you saved one. Keep the client
> process running and the port-forward up; see
> `/api/v1/openrl/models/df85...`.

Queued commands for those models are drained at the same time, so a worker
started later for the same owner does not execute stale work.

**A heartbeat after expiry does not revive the session.** An unknown
session id gets a `200` whose body carries `openrl.expired: true`, the key
is not recreated, and one unfiltered log line records it. Requests that
name an expired session get a 400 that says so and names the model view
of the affected model when there is one.

**`try_again` says what it is waiting on.** While a future is pending,
the poll response gains an `openrl` object: the stage (`launching
trainer`, `queued`, `running`), the worker's `state` and `detail` in the
vocabulary of 5.2, and how long the request has waited. This also closes
the one gap the model view leaves: during `create_model` the client has
only a request id, not a model id, so the first and longest wait cannot be
looked up; the pending response carries the answer instead. If the Workload is `Pending` with a
`NoCapacity` reason for longer than a configurable limit (default 10
minutes), the future fails with that reason. `WaitingForCapacity` keeps
waiting, because the scheduler expects a seat to free up.

### 5.4 Logs

The `FilterNoisyEndpoints` filter stays, since heartbeats and polls are
noise. Session expiry, revival attempts, worker release, and every future
the server fails get one structured line each, so a post-mortem from the
API server log alone is possible.

### 5.5 A reference for agents

`skills/openrl/SKILL.md` plus `references/` in this repository, in the
skill format coding agents share (a directory with a `SKILL.md` whose
frontmatter carries `name` and `description`, plus references the agent
loads on demand), and linked from `docs/getting-started.md` and the
README. The skill text names agents by nothing but that format; install
steps that differ per agent live in one section of the getting-started
guide, not in the skill.

**Scope: only the OpenRL bits.** Users reach OpenRL through
`tinker-cookbook`, which ships its own skills (`research`, `debug`,
`inkling`) for how to train: methodology, recipes, hyperparameters,
evaluation, logging, checkpoints, the SDK's threading model. Those are
better than anything this repository should write, and they move weekly.
The OpenRL skill does not repeat them. It covers what differs when the
server is OpenRL, names the cookbook skill to load for everything else,
and carries one table, `references/cookbook.md`, of the places where the
cookbook skills assume the hosted service (API keys, the model lineup,
`tinker run` and the checkpoint REST and CLI, billing and rate-limit
errors, the "is Tinker down" smoke test, heartbeat triage) with what is
true on OpenRL instead. The skill's description is scoped to deployment
and connection words so the two skills do not compete for the same
prompts; skills have no dependency mechanism, so the getting-started
guide installs both, with the per-agent steps in its own section. The
skill refers to the cookbook skills by their names, `research` and
`debug`, not by any one agent's slash commands. Vendoring the cookbook skills was
rejected: thirty thousand words that track hosted features, with the
drift ours to own. A small cookbook-side change (name `TINKER_BASE_URL`;
on an `openrl` key in capabilities, follow `skill_url`) would make the
discovery run both ways from the one endpoint every client hits; it is
noted, not proposed yet.

Contents:

- Cookbook: which tinker-cookbook skill to load for what, and the table
  of where their hosted-service advice differs on OpenRL; the supported
  Tinker surface.
- Connect: the endpoint, the port-forward, what to check first
  (`healthz`, `get_server_capabilities`), the model view and its state
  vocabulary.
- Settings: how to read the server's schema and how to pass them. No
  table; an earlier cut generated one from the server code and tested it
  for drift, and dropped it once the server published the schema itself.
- Worker states and normal durations: `waiting_for_placement` to `ready`
  for a cold node (image pull), a warm node, and a reused worker; vLLM
  start time; the 120 s idle release. The skill tells agents to branch on
  `state` and show `detail`.
- Runbook, symptom first: no output for N minutes, `NoCapacity`,
  `WaitingForCapacity`, a 400 naming a setting, a released-workers error.
  Each entry: what to check from the client, the fix, what to tell the
  platform owner.
- Lifecycle rules: keep the client process alive, keep the port-forward
  up, call `delete_model` at exit, save checkpoints if a run must survive
  a disconnect.

The earlier draft at `~/work/lora-without-regret/skills/openrl` (verified
on v0.0.4) is the starting point; its facts are re-verified against the
current release and its gateway-era names are updated.

**Shipped with every release.** `make release-bundle` adds
`openrl-skill.tar.gz`, the contents of `skills/openrl/` at the tag with
the "verified on" stamp set to the release version, listed in
`checksums.sha256` like the manifests. The repository copy stays
canonical; the asset is a copy, so a clone at the tag and the asset are
the same thing. `get_server_capabilities` reports
`openrl.skill_url` for the running version
(`https://github.com/gke-labs/open-rl/releases/download/<version>/openrl-skill.tar.gz`),
so an agent that starts from an endpoint finds the matching reference in
one call. The tarball is the primary install path, since it works for
every agent, for scripted installs, and for air-gapped clusters. Claude
Code users get a second path, a `.claude-plugin/marketplace.json` so
`/plugin marketplace add gke-labs/open-rl` works and can pin a tag; other
agents' package mechanisms can be added the same way, beside the tarball
rather than instead of it. Installing is still a manual step for every agent, so the
getting-started guide gets one install command right after
`kubectl apply`. `RELEASING.md` gets a step to re-read the timings and
runbook, since nothing checks those by test.

**Discovery.** Agents fetch the page they are pointed at and follow its
links; none probes `/llms.txt` on its own. So the landing page and the
README each get a plain link to the skill near the top, and a
`site/llms.txt` is added as a convenience for anyone told to read it.

### 5.6 Doctor and examples

`openrl-doctor` (a script in the examples environment, also copied into
the client image) runs in under a minute without a GPU:

1. `healthz` and `get_server_capabilities`; prints version, launcher mode,
   the settings, the limits, and whether `/openapi.json` answers.
2. Optional `--train`: a 3-step `tiny_sft` round trip with cleanup, timing
   each stage, so an agent learns what "normal" takes on this cluster. It
   is the only check that proves a worker can start; whether the GPU
   nodes have room is not visible from the endpoint until the fleet view
   lands (section 10).

One line per check, `OK`/`WARN`/`FAIL` with the reason, non-zero exit on
`FAIL`.

The tiny examples exit non-zero on failure and, right after
`create_model`, print the model id and the `curl` for its model view.
That is the usage model: launch the job, then ask the server about the
model from another shell (or a CLI, later) while a call blocks. An
earlier cut had the examples poll the view themselves and print a line a
minute; it was removed as a second status surface the examples had no
business owning.

## 6. Decisions

| Decision | Recommendation | Why | Status |
| --- | --- | --- | --- |
| API change scope | Additive only: new fields nested under `openrl` in capabilities, and one new path, `/api/v1/openrl/models/{model_id}` | SDK response models ignore unknown fields; Tinker wire paths stay identical. Fewer new paths means fewer contracts to keep | decided (narrowed 2026-10-12, PR #328 review) |
| Which status views ship now | Only the model view, scoped to one model id | A model id is a per-client capability, so the view adds no tenancy assumption; list and fleet views would expose every client's ids and need a tenancy design first | decided |
| Worker state contract | Five OpenRL-owned states in `state`; scheduler words only in free-text `detail` | Workload phases and pod states are scheduler internals; a rename must not break clients | decided |
| Where OpenRL-specific data lives in Tinker responses | One `openrl` object | Easy to find, easy to strip, never collides with SDK fields | open |
| Settings documentation | Published once by the server as `components.schemas.OpenRLSettings` in `/openapi.json`; capabilities point at it; no copy in the skill or docs | One artifact, pydantic's schema, for the spec, the 400 messages and every reader; a generated table was tried first and removed as a second copy | decided |
| Version source | Build arg `OPEN_RL_VERSION`, env var at runtime | `/app/VERSION` holds the internal counter, not the release tag | open |
| Expired session semantics | No revival; 200 with `expired: true` on heartbeat, 400 on other calls | Avoids a live-looking session with no workers; keeps the SDK heartbeat loop from erroring | open |
| Agent reference location | `skills/openrl/` in this repository, copied into each release as `openrl-skill.tar.gz` | Versioned with the code the tests check it against; the asset matches the manifests a user deployed | decided |
| Expired session semantics | No revival; 200 with `expired: true` on heartbeat, 400 on other calls | Avoids a live-looking session with no workers | decided |
| Version source | Build arg `OPEN_RL_VERSION`, env var at runtime | `/app/VERSION` holds the internal counter, not the release tag | decided |
| MCP server | Not now | REST plus skill covers current agents | decided |
| Fleet summary from ResourceSlices | Deferred with the fleet view | Needs the tenancy decision; the code exists on a side branch | deferred |

## 7. Phases

**Phase 1: see what is happening.** Implemented on this branch. The
first cut (verified on 2026-10-12 against `open-rl-dra`, api-server image
`agent1-27de66b` over the v0.0.6 bundle) had four endpoints and a fleet
summary; review of PR #328 raised tenancy and the scheduler-internals
contract, and the scope was narrowed the same day to the capabilities
`openrl` object and OpenAPI metadata (5.1), the model view with the state
vocabulary (5.2), the version build arg, the tiny examples' exit codes
and model-id line, and `openrl-doctor` on capabilities alone.
Unit tests with a fake `CustomObjectsApi` cover the state mapping and the
view. API server only, no cluster or RBAC change. The narrowed cut is to
be re-verified on `open-rl-dra` before release.

**Phase 2: failures reach the client.** Fail futures on release, drain
queues, expired-session semantics, `try_again` detail, `NoCapacity`
timeout, the log lines (5.3, 5.4). Tests in `test_session_lifecycle.py`.
E2E: drop the port-forward for three minutes during `tiny_sft` and assert
the client gets the released-workers error within one TTL plus one reaper
interval.

**Phase 3: the reference.** Implemented on this branch on 2026-10-12,
ahead of phase 2 by decision: `skills/openrl/` (SKILL.md, connect,
lifecycle, runbook, settings), `tests/test_skill_docs.py`, `make skill-bundle` inside
`release-bundle`, `.claude-plugin/marketplace.json`, `site/static/llms.txt`,
links from the landing page, README and getting-started, and the
`RELEASING.md` step, doctor in the client image (5.5, 5.6). The
generated settings table and its drift test were part of this phase and
were removed once the server published the schema itself. Re-verify
every number in the skill on the release the phase ships in.

Phases 1 and 2 are server changes and ship in the next release; phase 3
can land in parallel since the skill depends only on the endpoints from
phase 1.

## 8. Testing

- Unit: the model view against a fake Kubernetes client and the
  in-memory store, with a table test over the state mapping and a test
  that the response has only the model's own fields; the OpenAPI
  document's settings schema against `class Settings`, and the skill kept
  free of a settings table; future failure on release; heartbeat after
  expiry.
- Compatibility: `tiny_sft`, `tiny_rl` and one cookbook recipe against the
  changed server with the pinned SDK and the newest SDK, confirming the
  added fields are ignored.
- E2E on `open-rl-dra`: the getting-started steps, the doctor, and the
  port-forward drop scenario.

## 9. Risks

- The SDK could one day validate `try_again` strictly. Mitigation: the
  compatibility run above, and the `openrl` object is the only addition.
- The model view adds Kubernetes reads on a path agents may poll.
  Mitigation: a short in-process cache (5 s) and no watches.
- `detail` will be parsed by someone despite the documentation.
  Mitigation: the skill says so twice, and every example branches on
  `state`; if a need to branch on more appears, add a state, do not
  promote the string.
- Failing futures on release changes behaviour for clients that relied on
  waiting through a worker restart. No such path exists today: a released
  worker never comes back for the same session.

## 10. Deferred

Removed from the first cut on 2026-10-12 after review of PR #328, kept on
the branch `feat/agent-friendly-fleet-views` for when the questions below
are answered.

| Deferred | What it did | Open question |
|---|---|---|
| `GET /api/v1/openrl/status` | every worker with phase, node and pod state; live session count; store health; a fleet summary from ResourceSlices (devices per accelerator type, how many run an OpenRL worker) | OpenRL is multi-tenant in practice and has no auth. Which workers, sessions and fleet numbers may a given client see? A per-client scope needs an identity on the request, which nothing supplies today |
| `GET /api/v1/openrl/models` | every model, newest first, with owner and session liveness | same: listing other clients' model ids is a disclosure, and the ids are capabilities |
| `GET /api/v1/openrl/sessions/{session_id}` | live or expired, TTL, `user_metadata`, owners attached | session ids are not exposed by the SDK, so no client could use it without reading server state; revisit with phase 2 if the expired-session 400 needs a pointer |
| `openrl.endpoints` in capabilities | advertised the paths above | nothing to advertise with one path; the skill names it |
| `openrl-doctor` fleet and store checks | "is there a free GPU", "is the store reachable" | both come back with the fleet view. The store check could return earlier as a field in capabilities if a deployment needs it |
| progress during `create_model` | the client has no model id yet, so nothing can be asked about the model until the call returns | phase 2 puts `state` and `detail` into the pending `try_again` response, which needs no new path |
| pod facts in `detail` (`ContainerCreating`, `ImagePullBackOff`, `CrashLoopBackOff`, restart counts) | the first cut joined pods to Workloads in the API server | the pod is the scheduler's. The controller already reads it when it writes status; setting `reason` from the first container's waiting reason while `Placing`, and `Failed` on the terminal image and crash-loop reasons, gives the view and `kubectl get workloads` the same answer without the API server crossing into pods |

When the fleet view returns, it should report the same five worker
states and the same `detail` split as the model view, so a client that
learned one view can read the other.
