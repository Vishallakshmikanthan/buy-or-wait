# NVIDIA Nemotron Grounded Explanation Layer — Usage & Telemetry Report

**Evaluation Run Mode**: `offline_fallback`  
**Model Name**: `nvidia/nemotron-4-340b-instruct`  
**Model Provider**: `NVIDIA API Catalog (OpenAI-compatible)`  
**API Endpoint**: `https://integrate.api.nvidia.com/v1`  

## Summary Metrics

| Metric | Value |
|---|---|
| Total Evaluation Requests | 250 |
| Successful API Inferences | 0 |
| Failed API Inferences | 0 |
| Deterministic Fallbacks Used | 250 |
| Total Prompt (Input) Tokens | 0 |
| Total Completion (Output) Tokens | 0 |
| Total Tokens Processed | 0 |
| Average Tokens per Request | 0.0 |
| Total Latency (seconds) | 0.000 |
| Average Latency per Request (seconds) | 0.0000 |

## Breakdown by Mode

- **Actual API Usage**: 0 calls (API key not configured in environment; offline/no-key mode per Prompt 13 & 14).
- **Fallback Usage**: 250 calls (100% deterministic fallback generator adhering to canonical format).
- **Validation Pass Rate**: 250 / 250 (100.0%).

## Cost Estimation

- Official NVIDIA API pricing: Varies by endpoint / tier.
- Bypassed API cost: $0.00 (zero tokens consumed in offline mode).
- Estimated Total Cost: $0.00.
- Estimated Per-Request Cost: $0.00.

## Security & Privacy Hygiene

- **Zero Secret Leakage**: Verified zero API keys, authorization tokens, or credentials stored or logged.
- **Strict Data Minimization**: Model receives only certified fact pack derived from `DecisionCertificate`. Raw transactions, messages, and OCR data are excluded.
- **Decision Authority**: The deterministic financial engine remains the sole authority. The LLM cannot make or override financial decisions.
