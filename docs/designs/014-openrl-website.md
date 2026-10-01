# Design Doc 014: openrl.dev, the project website

**Status:** Draft
**Author:** Sunil Arora
**Last updated:** 2026-09-18

## 1. Summary

OpenRL gets a website at openrl.dev: one landing page and a blog. The
source lives in this repository under `site/`, is built with Hugo, and
is deployed to GitHub Pages by a workflow on every push to `main` that
touches `site/`. Blog posts are markdown files with front matter. The
landing page is a hand-written HTML template that carries its own CSS
and script, so it stays self-contained, while sharing the header,
footer and stylesheet with the blog so both read as one site. The look
comes from the OpenRL deck (`docs/openrl-deck.html`): paper background,
ink text, blue for the Tinker API, rust for the researcher's side, teal
for the platform's side, and the wireframe landscape as the hero.

openrl.io redirects to openrl.dev.

## 2. Goals and non-goals

Goals:

- One place that says what OpenRL is, shows the measured results, and
  gets a reader to the repo and the getting-started path.
- A blog we can publish to by merging a markdown file.
- No external requests from any page: fonts, scripts and images are all
  served from the site. No analytics in the first version.
- The whole site, including deployment, moves with the repository. A
  future move to a CNCF org must not require rebuilding anything.
- Site work proceeds in parallel with feature work on other branches
  without touching them (section 10).

Non-goals for the first version:

- A documentation section. `docs/` stays in the repo and on GitHub.
  The site links to it. A docs section can be added later under
  `/docs/` without changing the layout.
- Dark mode. The September 3 single-file draft had one; the deck look
  is light only, and one look is easier to keep consistent. It can be
  added later behind `prefers-color-scheme`.
- Search, comments, newsletters.

## 3. Decisions

| Decision | Recommendation | Why | Status |
| --- | --- | --- | --- |
| Canonical domain | openrl.dev, with openrl.io as a 301 redirect | The existing draft already assumes .dev; .dev is HTTPS-only by design; the .io country code has an uncertain long-term future after the Chagos agreement | open |
| Generator | Hugo, extended, pinned to v0.165.0 | Installed locally; single binary; markdown with front matter, RSS, sitemap and code highlighting built in; official Pages workflow. The landing page is one hand-written template, so template language matters little | open |
| Hosting | GitHub Pages, deployed by GitHub Actions | Stays inside the repo's GitHub project; free; no extra accounts; custom domain and HTTPS supported | open |
| Repository | This repository, under `site/` | The user's requirement; one repo to fork and one PR flow | decided |
| Look | The deck's palette and type | Most recent, approved, and the deck itself will be linked from the site | open |
| Fonts | One self-hosted open sans with a feel close to Avenir Next, plus the system stack; system mono | Avenir Next only exists on macOS; no CDN requests | open |
| Post authors | Name, GitHub handle optional | Simple front matter | open |

## 4. Site structure

| URL | Page | Source |
| --- | --- | --- |
| `/` | Landing page | `site/layouts/index.html` |
| `/blog/` | Post list, newest first | `site/layouts/blog/list.html` |
| `/blog/<slug>/` | One post | `site/content/blog/<slug>/index.md` |
| `/blog/index.xml` | RSS feed | generated |
| `/deck/` | The presentation deck | copied from `docs/openrl-deck.html` at build time |
| `/sitemap.xml`, `/robots.txt` | | generated |

### 4.1 Landing page sections

Copy comes from the deck; each section names its source slide.

1. Hero. "OpenRL. A self-hosted, Kubernetes-native post-training API
   for fine-tuning LLMs." Buttons: GitHub, Get started. The wireframe
   landscape renders behind it (deck slides 1 and 4).
2. RL is a loop. The four-phase diagram, the band of infrastructure
   under each phase, the coupling point (slide 7).
3. Decouple infra from AI research. The researcher, API, OpenRL figure
   and the four calls (slide 9).
4. What OpenRL provides. The five features (slide 11).
5. The loop is user code. The Tinker SDK loop with the four calls
   highlighted (slide 10).
6. How it is built. The architecture figure and the GPU-sharing figure
   (slides 13 and 14).
7. Results. Three cards: the LoRA Without Regret rank sweep, the
   time-slicing duty cycle, delta weight sync. Each card carries its
   headline numbers and links to the post or source (slides 15 to 17).
8. Get started. The three install commands and the SDK snippet
   (slide 18).
9. Roadmap and community. The six focus areas, the CNCF application,
   links to the repo, issues, ROADMAP.md, the deck and the blog
   (slides 19 and 20).
10. Footer. Apache 2.0, GKE Labs, "not an officially supported Google
    product", links.

The figures are the deck's inline SVGs, reused as they are. The
captured X posts from the deck's "Why now" slide are not on the
landing page; they belong in a post if anywhere.

### 4.2 Blog

- One directory per post: `site/content/blog/<slug>/index.md` with
  images beside it. Hugo calls this a page bundle.
- Front matter:

  ```yaml
  ---
  title: "From your Mac to GKE: fine-tuning Gemma with OpenRL"
  date: 2026-08-19
  authors: ["Sunil Arora"]
  summary: "One training loop, run locally first, then against a GKE cluster."
  tags: ["tutorial", "gke"]
  draft: false
  ---
  ```

- `draft: true` posts build locally with `hugo server -D` and never
  publish.
- Code blocks use Hugo's built-in highlighter with CSS classes, styled
  to match the deck's terminal blocks.
- The list page shows title, date, authors and summary.

Seed posts, in order:

1. Migrate `docs/blog/from-mac-to-gke.md`. It needs front matter, its
   repo links updated from the old `google/open-rl` org, and its SVGs
   moved into the bundle. The original file is then replaced by a
   pointer to the site.
2. Concurrent RL in action: the LoRA Without Regret reproduction on
   Qwen3-8B, from `docs/experiments/lora-without-regret` on branch
   `experiment/lora-without-regret`.
3. Delta weights in RL, from the July 2026 note and design doc 003.
4. A short post on time-slicing that carries the numbers and links to
   the full write-up on the llm-d blog.

The Google Open Source Blog introduction is linked from the landing
page and the blog list, not duplicated.

## 5. Repository layout

```
site/
  hugo.toml                  baseURL, permalinks, highlighter, outputs
  content/
    _index.md                landing page front matter only
    blog/
      _index.md
      from-mac-to-gke/
        index.md
        arch.svg ...
  layouts/
    index.html               the landing page, hand-written
    _default/baseof.html     html shell, head, header, footer
    blog/list.html
    blog/single.html
    partials/head.html, header.html, footer.html, landscape.html
  assets/
    css/site.css             palette, type, layout; shared by all pages
    js/landscape.js          the height-field renderer from the deck
    fonts/                   the self-hosted sans, woff2
  static/
    CNAME                    openrl.dev
    favicon.svg
    img/og.png               a landscape still for link previews
    robots.txt
  README.md                  how to preview, how to add a post
.github/workflows/site.yml   build on PR, build and deploy on main
.gitignore                   site/public/, site/resources/_gen/, site/static/deck/
```

The untracked `site/index.html` draft from September 3 is superseded
and removed when the scaffold lands. Its OG and description metadata
carry over into `partials/head.html`.

## 6. Look and feel

Tokens, taken from the deck:

| Token | Value | Use |
| --- | --- | --- |
| paper | `#F2EEDE` | page background |
| ink | `#1A1A1A` | headings, body |
| ink-soft | `#33312B` | secondary text |
| ink-faint | `#85837A` | captions, rules |
| accent | `#1E6FCC` | links, the Tinker API |
| rust | `#B4470F` | the researcher's side |
| teal | `#0F8A7A` | the platform's side |
| tint | `#E9E4D0` | panels and bands |

Type: a self-hosted variable sans for headings and body, system mono
for code. The deck's sizes scale with slide width; the site uses a
fixed type scale with a 72ch measure for post bodies and a 1120px
maximum width for the landing page.

The landscape: the deck's canvas renderer draws the flat height field
behind the hero at page load, paused for `prefers-reduced-motion`. No
pinch animation on the site.

Code blocks: dark panel, light text, the same four colours the deck
uses for keywords, strings, comments and highlighted lines.

## 7. Build and deploy

- Hugo extended v0.165.0, pinned in the workflow and noted in
  `site/README.md`. Local preview: `hugo server -s site -D`.
  Build: `hugo -s site --minify`, output in `site/public/`.
- Workflow `site.yml`:
  - `pull_request` with `paths: [site/**, docs/openrl-deck.html,
    .github/workflows/site.yml]`: build only, upload `public/` as an
    artifact for review.
  - `push` to `main` with the same paths, plus `workflow_dispatch`:
    build, then deploy with `actions/configure-pages`,
    `actions/upload-pages-artifact`, `actions/deploy-pages`.
  - Permissions: `contents: read`, `pages: write`, `id-token: write`.
    Concurrency group `pages`, no cancellation of in-flight deploys.
  - Before the build, copy `docs/openrl-deck.html` to
    `site/static/deck/index.html`.
- Repository settings, done once by an admin: Pages source set to
  GitHub Actions; custom domain `openrl.dev`; "Enforce HTTPS" on after
  the certificate is issued.
- DNS for openrl.dev, at whichever provider hosts it:

  ```
  openrl.dev.      A     185.199.108.153
  openrl.dev.      A     185.199.109.153
  openrl.dev.      A     185.199.110.153
  openrl.dev.      A     185.199.111.153
  openrl.dev.      AAAA  2606:50c0:8000::153
  openrl.dev.      AAAA  2606:50c0:8001::153
  openrl.dev.      AAAA  2606:50c0:8002::153
  openrl.dev.      AAAA  2606:50c0:8003::153
  www.openrl.dev.  CNAME gke-labs.github.io.
  ```

  GitHub also asks for a `_github-pages-challenge-gke-labs` TXT record
  to verify the domain for the organization; the value comes from the
  organization's Pages settings.
- openrl.io: GitHub Pages serves one custom domain per site, so the
  redirect is configured at the .io domain's DNS or registrar. With
  Cloudflare it is one redirect rule to `https://openrl.dev$1`; most
  registrars offer domain forwarding with the same effect. If the .io
  DNS is on a provider without forwarding, the fallback is a second,
  tiny Pages repository whose only page is a meta refresh.

## 8. Authoring workflow

To publish a post:

1. Branch from `main`.
2. `mkdir site/content/blog/<slug>`, write `index.md` with front
   matter, put images beside it.
3. `hugo server -s site -D` and read it at `localhost:1313/blog/<slug>/`.
4. Open a PR. The build check runs; the artifact holds the rendered
   site if a reviewer wants to look without checking out.
5. Merge. The deploy runs and the post is live within a minute or two.

To change the landing page, edit `site/layouts/index.html` or
`site/assets/css/site.css` and follow the same PR path.

## 9. Phases

1. Scaffold PR. Hugo site, base templates, CSS tokens, the landing page
   drafted from the deck, the migrated from-mac-to-gke post, the
   build-only workflow, a README line linking the site. Reviewed from
   screenshots and the local server before it opens.
2. Admin. Pages enabled with the custom domain, DNS records added, the
   .io redirect configured. Then a one-line PR that turns on the deploy
   job. Verify HTTPS and the redirect.
3. Content PRs. The three remaining seed posts, the OG image, the deck
   at `/deck/`.
4. Later, as needed: dark mode, a docs section, a project logo.

## 10. Working in parallel with feature branches

Site work never touches the checkout used for feature work. Each
effort gets its own worktree, which is a second working directory
attached to the same repository:

```
# once
git worktree add ../open-rl-site -b website/scaffold upstream/main

# work there
cd ../open-rl-site
hugo server -s site -D
git push -u origin website/scaffold
gh pr create --repo gke-labs/open-rl --base main --head droot:website/scaffold

# when the PR has merged
cd ../open-rl && git worktree remove ../open-rl-site
```

Branches for the site are named `website/<topic>`. This design doc is
on `website/design-doc` in the worktree at `../open-rl-site`.

Because `site/` is a leaf directory, site PRs and feature PRs do not
conflict in the tree. The existing CI workflows run on every PR; the
heavier ones can add `paths-ignore: [site/**]` so a blog post does not
run the controller tests. That change is part of the scaffold PR only
if the reviewers of those workflows agree.

## 11. Alternatives considered

- Astro or Eleventy. Both fine; both add a Node dependency tree and a
  build pipeline for a site with two templates. Hugo is one binary.
- A small Python script rendering markdown into a template. The least
  machinery, but RSS, sitemap, drafts, list pages and highlighting all
  have to be rebuilt by hand, and Hugo gives them for free.
- Netlify, Cloudflare Pages or Firebase Hosting. Each adds redirects
  and previews per PR, which GitHub Pages lacks. Each also adds an
  account outside GitHub. Worth revisiting if PR previews become
  important.
- A separate repository for the site. Cleaner CI separation, but the
  user's requirement is one repository, and a subdirectory keeps the
  site with the code it describes.

## 12. Open questions

1. openrl.dev as canonical, and where the DNS for each domain is
   hosted.
2. Hugo and GitHub Pages, or a different pair.
3. Who has admin on gke-labs/open-rl for the Pages settings and the
   organization-level domain verification.
4. Which self-hosted sans. Candidates with an Avenir-like feel under
   the Open Font License: Figtree, Nunito Sans, Outfit.
5. Whether the from-mac-to-gke post is still current enough to migrate
   as the first post, or whether it should be refreshed first.
