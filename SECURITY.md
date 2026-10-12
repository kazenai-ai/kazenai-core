# Security

## Reporting
Report suspected vulnerabilities privately to **founder@kazenai.com**. Please do not open public issues for security problems; we will acknowledge your report and coordinate a fix and disclosure.

## Capture defaults (SDK)
The `kazenai` SDK defaults to metadata-only capture (request/response bodies omitted). Secret redaction runs before sinks/logs. Capturing request or response bodies requires explicit configuration.

## Scope
This repository's license covers the SDK code. Agent FinOps, Agent Lens and other hosted KazenAI services are separate products with separate terms.
