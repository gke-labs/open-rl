# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in OpenRL, please report it responsibly.
**Do not open a public GitHub issue.**

To report a vulnerability, use GitHub Private Vulnerability Reporting:

* **[Report a vulnerability](https://github.com/gke-labs/open-rl/security/advisories/new)**

<!-- TODO: Set up a project security email (e.g. openrl-security@googlegroups.com) and
     add it here as an alternative reporting channel for reporters who cannot or prefer
     not to use GitHub. -->

Please include in your report:

* Description of the vulnerability
* Steps to reproduce the issue
* Affected versions
* Any potential impact you have identified

## Response Timeline

The OpenRL security response team will acknowledge receipt of your report within
**3 business days** and will provide an estimated timeline for a fix within
**10 business days**.

The team will keep you informed of progress toward a fix and may ask for additional
information.

## Supported Versions

OpenRL has not yet cut a tagged release. Until the first release, only the `main` branch is
supported and security fixes land there.

| Version | Supported |
| ------- | --------- |
| `main`  | Yes       |

<!-- TODO: Replace this table with a real supported-versions matrix once the project
     starts cutting tagged releases. -->

## Disclosure Policy

When a security issue is confirmed, the OpenRL maintainers will:

1. Develop and test a fix
2. Assign a CVE identifier if appropriate
3. Release a patched version
4. Publish a security advisory via
   [GitHub Security Advisories](https://github.com/gke-labs/open-rl/security/advisories)

## Security Response Team

The security response team handles all reports of security vulnerabilities according to this
policy. See [GOVERNANCE.md](GOVERNANCE.md) for how the security response team is appointed and
maintained, and [MAINTAINERS.md](MAINTAINERS.md) for the current members.

## Security Practices

The project maintains the following security practices:

* Automated dependency updates via Dependabot
* Linting and build checks on every pull request

<!-- TODO: As the project matures, consider adding an OpenSSF Best Practices badge,
     signed releases, and a published SBOM. -->

## Disclaimer

This is not an officially supported Google product. This project is not eligible for the
Google Open Source Software Vulnerability Rewards Program.
