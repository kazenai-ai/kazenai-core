# Security

## Reporting
Report suspected vulnerabilities privately to the maintainer (Dhavan Shah / KazenAI contact channels on https://kazenai.com). Do not open public issues for unreleased security details.

## Capture defaults (SDK)
The `kazenai` SDK defaults to metadata-only capture (request/response bodies omitted). Secret redaction runs before sinks/logs. Enabling fuller capture requires explicit configuration and consent as documented in `docs/security/capture-and-retention.md` when present.

## Scope
This repository's license covers the SDK/schema code. Hosted FinOps/Lens and other private services are separate products with separate terms.
