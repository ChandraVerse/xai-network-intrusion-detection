# Security Policy

## Reporting a Vulnerability

This is an open-source research project. If you discover a security vulnerability — in the code, dependencies, Docker configuration, or any other component — please **do not open a public GitHub issue**.

Instead, report it privately:

**Email:** [Open a private security advisory on GitHub](https://github.com/ChandraVerse/xai-network-intrusion-detection/security/advisories/new)

Please include:
- A clear description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (optional)

You will receive an acknowledgement within **72 hours**. We aim to release a fix within **14 days** for critical issues.

---

## Supported Versions

| Version | Supported |
|---|---|
| `main` (latest) | ✅ Yes |
| Older tags | ❌ No |

---

## Scope

In-scope for security reports:
- Remote code execution via model loading (pickle/joblib deserialization)
- Injection vulnerabilities in the Streamlit dashboard
- Dependency vulnerabilities in `requirements.txt`
- Dockerfile privilege escalation or image poisoning
- Sensitive data exposure in logs or outputs

Out-of-scope:
- Attacks requiring physical access to the machine running this tool
- Vulnerabilities in CICIDS-2017 dataset itself (report to UNB directly)
- Social engineering

---

## Dependency Security

We monitor dependencies using GitHub Dependabot. To check for known CVEs locally:

```bash
pip install safety
safety check -r requirements.txt
```

---

## Note on Model Artifacts

> ⚠️ Never load `.pkl` or `.h5` model files from untrusted sources.

`joblib`/`pickle` deserialisation can execute arbitrary code. Always verify SHA-256 checksums from `models/model_registry.yaml` before loading any model artifact:

```bash
python scripts/compute_checksums.py
```

This applies especially if you downloaded model files from a third-party mirror.

---

## Acknowledgements

We appreciate responsible disclosure. Reporters who follow this policy will be credited in the relevant release notes (if they wish).
