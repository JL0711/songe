# Security

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub: on
[sous-labs/songe](https://github.com/sous-labs/songe), open the **Security**
tab and choose **Report a vulnerability**. Do not open a public issue.

Include the Songe version (`songe --version`), your operating system, the
model server and agent you use, and the steps to reproduce. We will
acknowledge the report, keep you informed while we fix it, and credit you in
the release notes if you wish.

## Scope

Songe runs on your machine and is designed to stay there. Reports we
especially want:

- the local proxy or dashboard (127.0.0.1) accepting requests it should
  refuse, such as from another origin or another machine;
- Songe running tools, commands or code from recorded conversations (it must
  only read them);
- agent settings (opencode, pi) written or restored incorrectly, or backups
  lost;
- recorded conversations or learned versions leaving the machine, or being
  readable by other local users;
- `songe uninstall --purge` leaving data it said it would delete.

## Supported versions

Songe is in alpha. Fixes go into the latest release only.
