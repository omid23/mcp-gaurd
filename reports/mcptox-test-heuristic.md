# MCPTox test split, heuristic baseline, no intent

2172 cases · intent off · policy + model · thresholds deny ≥ 0.85, ask ≥ 0.5 · 2026-10-09T14:03:15Z

## tool_call

| backend | attacks | passed (ASR) | denied | asked | benign | allowed | asked | denied | p50 ms | p95 ms | $/1k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| heuristic | 839 | 62.9% | 4.3% | 32.8% | 937 | 86.3% | 13.7% | 0.0% | 0.3 | 0.7 | 0.0000 |

Attacks stopped by category (heuristic):

- Service Disruption: 19% of 143
- Privacy Leakage: 34% of 139
- Information Manipulation: 21% of 136
- Infrastructure Damage: 71% of 109
- Data Tampering: 18% of 96
- Code Injection: 77% of 91
- Financial Loss: 10% of 51
- Credential Leakage: 68% of 34
- Instruction Tampering: 58% of 24
- Message Hijacking: 38% of 8
- Other: 0% of 8

## tool_description

| backend | attacks | passed (ASR) | denied | asked | benign | allowed | asked | denied | p50 ms | p95 ms | $/1k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| heuristic | 212 | 12.7% | 87.3% | 0.0% | 184 | 100.0% | 0.0% | 0.0% | 0.6 | 1.9 | 0.0000 |

Attacks stopped by category (heuristic):

- Information Manipulation: 81% of 42
- Privacy Leakage: 90% of 39
- Service Disruption: 97% of 35
- Infrastructure Damage: 97% of 29
- Code Injection: 93% of 15
- Data Tampering: 57% of 14
- Credential Leakage: 92% of 13
- Financial Loss: 100% of 12
- Instruction Tampering: 38% of 8
- Message Hijacking: 100% of 3
- Other: 100% of 2

