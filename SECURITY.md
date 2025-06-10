# Security Policy

This project follows a responsible disclosure process. If you discover a security issue, please email **security@example.com** with the details.

## Supported Versions

| Version | Supported |
| ------- | --------- |
| 1.0.x   | Yes |

## Reporting a Vulnerability

Please include a detailed description and steps to reproduce the issue. We will respond within one week.

## Draft Security Advisory

**Issue**: Potential path traversal when loading matrix files from the YAML configuration.

When the tool is executed with an untrusted configuration file, attackers may supply arbitrary file paths in the `file` field. This could allow reading unexpected files when `pandas.read_excel` is called.

**Status**: Investigating. A fix is planned to validate file paths before opening them.

