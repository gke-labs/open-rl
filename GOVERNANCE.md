# OpenRL Project Governance

The OpenRL project is dedicated to providing a self-hosted, Tinker-compatible API for
fine-tuning language models on your own infrastructure. This governance explains how the
project is run.

- [Values](#values)
- [Maintainers](#maintainers)
- [Becoming a Maintainer](#becoming-a-maintainer)
- [Removing a Maintainer](#removing-a-maintainer)
- [Emeritus Maintainers](#emeritus-maintainers)
- [Meetings](#meetings)
- [Code of Conduct](#code-of-conduct)
- [Security Response Team](#security-response-team)
- [Voting](#voting)
- [When to Evolve This Governance](#when-to-evolve-this-governance)
- [Modifying this Charter](#modifying-this-charter)

## Values

The OpenRL project and its leadership embrace the following values:

* **Openness**: Communication and decision-making happens in the open and is discoverable
  for future reference. As much as possible, all discussions and work take place in public
  forums and open repositories.

* **Fairness**: All stakeholders have the opportunity to provide feedback and submit
  contributions, which will be considered on their merits.

* **Community over Product or Company**: Sustaining and growing our community takes priority
  over shipping code or sponsors' organizational goals. Each contributor participates in the
  project as an individual.

* **Vendor Neutrality**: The project direction and decisions are not controlled by any single
  organization. Maintainer selection, roadmap prioritization, and release decisions are made
  based on project merit, not employer affiliation.

* **Inclusivity**: We innovate through different perspectives and skill sets, which can only
  be accomplished in a welcoming and respectful environment.

* **Participation**: Responsibilities within the project are earned through participation,
  and there is a clear path up the contributor ladder into leadership positions.

## Maintainers

OpenRL Maintainers have write access to the [project GitHub repository](https://github.com/gke-labs/open-rl).
They can merge their own patches or patches from others. The current maintainers can be found
in [MAINTAINERS.md](MAINTAINERS.md). Maintainers collectively manage the project's resources
and contributors.

This privilege is granted with some expectation of responsibility: maintainers are people who
care about the OpenRL project and want to help it grow and improve. A maintainer is not just
someone who can make changes, but someone who has demonstrated their ability to collaborate
with the team, get the most knowledgeable people to review code and docs, contribute
high-quality code, and follow through to fix issues (in code or tests).

A maintainer is a contributor to the project's success and a citizen helping the project
succeed.

The collective team of all Maintainers is known as the Maintainer Council, which is the
governing body for the project.

### Becoming a Maintainer

To become a Maintainer you need to demonstrate the following:

* commitment to the project:
  * participate in discussions, contributions, code and documentation reviews for **3 months**
    or more,
  * perform reviews for **5** non-trivial pull requests,
  * contribute **5** non-trivial pull requests and have them merged,
* ability to write quality code and/or documentation,
* ability to collaborate with the team,
* understanding of how the team works (policies, processes for testing and code review, etc),
* understanding of the project's code base and coding and documentation style.

<!-- TODO: OpenRL does not yet have a public developer mailing list. Once one exists,
     nominations should be sent there instead of to a GitHub issue, and this section
     should be updated with the list address. -->

A new Maintainer must be proposed by an existing maintainer by opening a GitHub issue in the
project repository. A simple majority vote of existing Maintainers approves the application.
Maintainer nominations will be evaluated without prejudice to employer or demographics and
should consider the organizational diversity of the maintainer group.

Maintainers who are selected will be granted the necessary GitHub rights.

### Removing a Maintainer

Maintainers may resign at any time if they feel that they will not be able to continue
fulfilling their project duties.

Maintainers may also be removed after being inactive, failure to fulfill their Maintainer
responsibilities, violating the Code of Conduct, or other reasons. Inactivity is defined as a
period of very low or no activity in the project for **6 months** or more, with no definite
schedule to return to full Maintainer activity.

A Maintainer may be removed at any time by a 2/3 vote of the remaining maintainers.

### Emeritus Maintainers

Depending on the reason for removal or resignation, a Maintainer may be converted to Emeritus
status. Emeritus Maintainers are recognized for their past contributions and may still be
consulted on project matters, but do not have voting rights or merge access. Emeritus
Maintainers are listed in [MAINTAINERS.md](MAINTAINERS.md) under a separate Emeritus section.

An Emeritus Maintainer may be reinstated to active Maintainer status by a simple majority vote
of existing Maintainers, provided they meet the current Maintainer requirements and can commit
to ongoing participation.

## Meetings

<!-- TODO: OpenRL does not yet have a regular public developer meeting. Once one is
     scheduled, add the cadence, joining details, and notes/recording location here,
     and update CONTRIBUTING.md and the README to point to it. -->

Time zones permitting, Maintainers are expected to participate in the public developer
meeting. A regular public developer meeting has not yet been established for OpenRL; until
one is, project discussion happens in GitHub issues and pull requests.

Maintainers will also have closed meetings in order to discuss security reports or Code of
Conduct violations. Such meetings should be scheduled by any Maintainer on receipt of a
security issue or CoC report. All current Maintainers must be invited to such closed meetings,
except for any Maintainer who is accused of a CoC violation.

## Code of Conduct

[Code of Conduct](CODE_OF_CONDUCT.md) violations by community members will be discussed and
resolved privately by the Maintainer Council. If a Maintainer is directly involved in the
report, the remaining Maintainers will designate two Maintainers to resolve it, and the
Maintainer named in the report is recused from the discussion.

## Security Response Team

The Maintainers will appoint a Security Response Team to handle security reports. This
committee may simply consist of the Maintainer Council themselves. If this responsibility is
delegated, the Maintainers will appoint a team of at least two contributors to handle it. The
Maintainers will review who is assigned to this at least once a year.

The Security Response Team is responsible for handling all reports of security holes and
breaches according to the [security policy](SECURITY.md).

## Voting

While most business in OpenRL is conducted by
"[lazy consensus](https://community.apache.org/committers/lazyConsensus.html)", periodically
the Maintainers may need to vote on specific actions or changes. A vote can be taken in a
GitHub issue in the project repository, or privately among Maintainers for security or conduct
matters. Any Maintainer may demand a vote be taken.

Most votes require a simple majority of all Maintainers to succeed, except where otherwise
noted. Two-thirds majority votes mean at least two-thirds of all existing maintainers.

## When to Evolve This Governance

The Maintainer Council model works well for focused projects with a small, cohesive group of
contributors. As the project grows, watch for these signals that a governance transition may
be needed:

* **Decisions stall.** When the maintainer group is too large for lazy consensus to work, or
  when decisions affect subgroups differently, a delegation structure (working groups, SIGs)
  helps.
* **New contributors cannot find a path in.** If the only path to influence is "become a
  maintainer," the project needs intermediate roles (reviewer, approver). Projects with
  intermediate roles produce more diverse maintainer pools because they give external
  contributors a visible progression path.
* **A single organization dominates.** All current OpenRL maintainers are employed by Google.
  As the project attracts contributors from other organizations, the Maintainers should
  actively grow the maintainer group beyond a single employer and consider adding an
  org-balanced voting clause (for example, capping any one organization at 1/3 of the total
  votes on a decision).
* **Subprojects diverge.** When parts of the project develop their own contributor communities
  or release cadences, consider a federated subproject governance model.

These transitions are a sign of project growth, not governance failure.

## Modifying this Charter

Changes to this Governance and its supporting documents may be approved by a 2/3 vote of the
Maintainers.
