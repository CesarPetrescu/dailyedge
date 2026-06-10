#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/evidence_cache.py --v2 --universe "${DAILYEDGE_UNIVERSE:-liquid_ai_semis}" > /tmp/dailyedge_evidence_v2.out
python3 - <<'PY'
import json
from pathlib import Path
E=json.loads(Path('output/evidence_v2.json').read_text())
rows={r['setup']: r for r in E.get('rows', [])}
print('DailyEdge evidence v2 refreshed')
print('universe:', E.get('universe'), 'tickers:', len(E.get('ticker_list', [])), 'ok:', E.get('data_quality', {}).get('ok_count'), 'failed:', E.get('data_quality', {}).get('failed_count'))
for setup in ['ema_cross', 'breakout_20_rr3']:
    r=rows.get(setup)
    if not r:
        continue
    bm=r.get('bootstrap_margin', {})
    print(f"{setup}: n={r.get('n')} E_cons={float(r.get('E_cons',0)):+.2f} base={float(r.get('baseline_E_cons',0)):+.2f} Pgt={float(bm.get('p_setup_gt_base',0)):.2f} gap={float(bm.get('E_cons_gap',0)):+.2f} beats={r.get('beats_baseline_cons')}")
PY
