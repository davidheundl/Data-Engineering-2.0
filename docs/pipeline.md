# EVADE / DiscoGeM Level-1 Pipeline

Dieses Dokument beschreibt die aktuelle Implementierung der Pipeline — von den
Rohdaten bis zu den Analyse-Artefakten. Es konzentriert sich auf das Was/Wie
(Inputs, Outputs, Schemas, Algorithmus pro Stage). Die *Motivation* der
einzelnen Stages ist hier nicht das Thema; die EVADE-Idee selbst ist im
Theorie-Teil deiner Präsentation erklärt.

## Übersicht

Die Pipeline läuft fünfstufig, asynchron, mit Zwischenartefakten auf der Platte
(JSONL-Dateien). Jede Stage liest die JSONL der Vorgänger und schreibt die
ihrige — kein gemeinsamer Speicher, keine Datenbank.

```
DiscoGeM CSVs
       │
       ▼  Stage 1: Prep                  (src/prep.py)
   items.jsonl
       │
       ▼  Stage 2: Generate              (src/generate.py)
   generations.jsonl ───► costs.csv
       │
       ▼  Stage 3: Validate              (src/validate.py)
   validations.jsonl ───► costs.csv
       │
       ▼  Stage 4: Aggregate / Compare   (src/aggregate.py)
   distributions_aggregate.jsonl  (Variante A)
   distributions.jsonl            (Variante B, comparative)
   comparative_results.jsonl
   metrics_aggregate.json | metrics_compare.json
       │
       ▼  Stage 5: Analyze
   analysis/, analysis_aggregate/, analysis_compare/
   analysis_detailed/   (via scripts/analyze_results.py)
```

Alle Ausgaben landen in `results/{run_id}/`, wobei
`run_id = {UTC-timestamp}_{config-name}_{git-short-hash}` — z.B.
`20260607T081000Z_level1_experiment_7108c04`.

## Entry Points

**Orchestrator: [scripts/run_level1.py](../scripts/run_level1.py).** Ruft
die fünf Stage-Funktionen sequenziell auf. CLI-Argumente:

- `--config` (required): Pfad zur YAML-Config.
- `--stages` (default `prep,generate,validate,aggregate,compare,analyze`):
  Komma-separierte Liste der zu fahrenden Stages — für Re-Runs einzelner
  Stages.
- `--run-id`: in ein bestehendes Run-Verzeichnis hinein resumen (Prep wird
  dann übersprungen, items.jsonl wird wiederverwendet).
- `--items-file`: Textdatei mit Item-IDs (eine pro Zeile) zum Überspringen
  des stratified Sampling — nützlich, wenn du einen exakten Item-Satz
  reproduzieren willst.

**Standalone-Analyse: [scripts/analyze_results.py](../scripts/analyze_results.py).**
Liest ein fertiges Run-Verzeichnis und schreibt vertiefte Plots/Statistiken
nach `analysis_detailed/`. Sektionen A–F (siehe Stage 5).

## Stage 1: Prep

**Modul:** [src/prep.py](../src/prep.py)

**Inputs:**
- `DiscoGeM 1.0_items/DiscoGeM1.0.wide.csv` — eine Zeile pro Item, mit
  `arg1`, `arg2`, `genre`, `split`, `reflabel` (nur bei Wikipedia bzw.
  PDTB-Items gesetzt).
- `DiscoGeM 1.0_labels/DiscoGeMcorpus_fulldataset.csv` — eine Zeile pro
  Annotator-Annotation; relevant ist die Spalte `lev1_conn2` (PDTB
  Level-1-Sense).

**Schritte:**

1. Genre-Mapping auf drei Klassen (`Europarl`, `Lit`, `Wiki`).
2. Drop von Annotationen mit `lev1_conn2 == NA` oder Multi-Wert-Annotationen
   ("expansion,contingency"). Letzteres ist selten und ambiguous.
3. Pro Item: Liste der Level-1-Senses aller Annotatoren.
4. Aufbau der Crowd-Verteilung:

   ```python
   def _build_crowd_distribution(senses):
       counts = Counter(senses)
       total = sum(counts.values())
       return {s: c / total for s, c in counts.items()}
   ```

5. **Stratified Sampling** nach Konfiguration: pro Genre wird eine Quote
   gesetzt (`sampling.genre_split`), 50/50 aufgeteilt zwischen
   *high-* und *low-agreement* Items (Schwelle 0.5 auf die größte
   Sense-Wahrscheinlichkeit). Seed steuert die Auswahl reproduzierbar.

**Output: `items.jsonl`.** Pro Zeile ein `PrepItem`
([src/schemas.py](../src/schemas.py)):

| Feld | Bedeutung |
|---|---|
| `item_id` | DiscoGeM-Item-ID |
| `genre` | Europarl / Lit / Wiki |
| `arg1`, `arg2` | volle Argumenttexte |
| `arg1_singlesentence`, `arg2_singlesentence` | reduzierte Single-Sentence-Form |
| `annotator_step2_senses` | Liste der L1-Senses pro Annotator |
| `n_valid_annotations` | Anzahl Annotatoren (typ. 9–11) |
| `crowd_sense_distribution` | normalisierte Senseverteilung über alle Annotatoren |
| `candidate_senses` | sortierte Sense-Liste aus der Verteilung (ohne `norel`) |
| `majority_single_sense` | wahrscheinlichster Sense |
| `crowd_agreement_score` | maximale Sense-Wahrscheinlichkeit |
| `stratification_bin` | `high` (≥0.5) oder `low` (<0.5) |
| `wikipedia_reference_labels` | editorisches Gold-Label (nur für Wiki/PDTB), Level-2-Tokens als Liste |

## Stage 2: Generate

**Modul:** [src/generate.py](../src/generate.py)

**Inputs:** `items.jsonl` plus die Sense-Definitionen unter
`prompts/pdtb_sense_definitions.json` (PDTB-3.0-Definition + kanonisches
Beispiel pro L1-Sense).

**Was passiert:** Für jedes Tripel
*(Item, Sense ∈ {temporal, contingency, comparison, expansion}, Generator-Modell)*
wird ein LLM-Call abgesetzt. Der Prompt fordert *alle* unterschiedlichen
Begründungen, warum die implizite Discourse-Relation zwischen Arg1 und Arg2
genau dieser Sense sein könnte — kein Argument zweimal in anderen Worten.

Wenn das Modell genuin nicht begründen kann, soll es eine leere Liste
zurückgeben — das ist eine **Abstention** und wird so erfasst (nicht als
Fehler).

Output-Schema des Modells (strikt JSON):

```json
{ "explanations": ["…", "…", "…"] }
```

Geparst durch `_parse_generation_response_l1`, das sowohl das obige
Plural-Schema als auch das alte Single-Format toleriert.

**Idempotenz:** Vor dem Call wird geprüft, ob das Tripel
*(item_id, candidate_sense, generator_model)* schon in der bisherigen
`generations.jsonl` steht — wenn ja, wird übersprungen. Abgebrochene Runs
können also einfach noch einmal angestoßen werden, ohne dass Tokens doppelt
gezahlt werden.

**LLM-Dispatch:** über [src/llm_client.py](../src/llm_client.py). Provider
werden anhand des Modell-IDs `"provider:model"` ausgewählt; die SDKs
(`openai`, `anthropic`, `mistralai`) werden lazy initialisiert. Pro Provider
sitzt ein `asyncio.Semaphore`; Mistral ist auf 1 begrenzt (Free-Tier-Limit),
die anderen auf 4.

**Retry / Rate-Limit:** Tenacity, exponentielles Backoff:

```python
async for attempt in AsyncRetrying(
    stop=stop_after_attempt(self.max_retries),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(LLMError),
    reraise=True,
):
    ...
```

`LLMError` deckt 429er, 5xx, Timeouts. `FatalLLMError` (z.B. 401/403)
unterdrückt den Retry und propagiert sofort.

**Output: `generations.jsonl`** (`GenerationRecord`):

- `generation_id` (UUID), `item_id`, `candidate_sense`, `generator_model`
- `explanations: list[str]` (leer wenn `abstained`)
- `abstained: bool`, `abstention_reason: str | None`
- Kosten/Telemetrie: `input_tokens`, `output_tokens`, `cost_usd`,
  `latency_ms`, `timestamp`
- `raw_response` (Original-JSON-String)

Zusätzlich wird pro Call eine Zeile in `costs.csv` angefügt:
`stage,provider,model,item_id,input_tokens,output_tokens,cost_usd,latency_ms,timestamp`.

## Stage 3: Validate

**Modul:** [src/validate.py](../src/validate.py)

**Inputs:** `generations.jsonl`, `items.jsonl`, Sense-Definitionen.

**Cross-Validation:** Jede Explanation wird von Validatoren bewertet, die
**nicht** das gleiche Modell wie der Generator sind. Im Code:

```python
for validator_model in config.models.validators:
    if validator_model == gen.generator_model:
        continue
```

Damit kann kein Modell sich selbst bewerten. Der Validator-Prompt fragt nach
einer integer-Bewertung 0–10 (0–2 schwach, 3–5 mittelmäßig, 6–7 ok, 8–10
stark) ohne weiteren Text. Das Score-Parsing ist defensiv:

```python
INT_RE = r"\b(\d{1,2})\b"

def parse_validity_score(text):
    m = INT_RE.search(text or "")
    if not m: return 0.0, False
    raw = max(0, min(10, int(m.group(1))))
    return raw / 10.0, True
```

Erste Zahl im Text, geclampt auf [0,10], normiert auf [0,1]. Fehlschläge
landen in `parsing_errors_{run_id}.jsonl` und werden mit `parsing_success=False`
markiert (Score 0).

**Idempotenz:** wie Stage 2, hier auf
*(item_id, candidate_sense, explanation_text, validator_model)*.

**Output: `validations.jsonl`** (`ValidationRecord`):

- `validation_id`, `generation_id`, `item_id`, `candidate_sense`,
  `validator_model`, `explanation_text`
- `validity_score: float ∈ [0,1]`, `parsing_success: bool`,
  `raw_response: str`
- Kosten/Telemetrie wie oben.

## Stage 4: Aggregate & Compare

**Modul:** [src/aggregate.py](../src/aggregate.py)

Aus den Validator-Scores werden zwei *parallele* LLM-Verteilungen pro Item
gebaut — die "Variante A" und "Variante B" der Analyse.

### Per-Sense-Statistiken

Pro (Item, Sense) werden alle Explanations gesammelt; pro Explanation der
Mittelwert über alle Validator-Scores. Daraus:

| Feld in `SenseStats` | Bedeutung |
|---|---|
| `max_validity` | bester (höchster) Explanation-Mean |
| `mean_validity` | Mittel der Explanation-Means |
| `n_explanations` | Anzahl bewerteter Explanations |
| `n_generators_abstained` | wie viele Generatoren für diesen Sense abstainten |
| `validator_std` | mittlere Standardabweichung innerhalb der Explanation-Scores |
| `commission_score` | `(1 − max_validity) × (1 − validator_std)`, geclampt — hoch = niedrige Bestätigung *und* niedrige Validator-Einigkeit, also „verdächtig" |

### Variante A — Aggregate

Aus den `SenseStats` wird per Tau-Sweep (τ ∈ {0.1, 0.2, …, 0.9}) eine
Sense-Verteilung gebaut:

```python
def _build_llm_distribution(per_sense_stats, tau, temperature):
    validated = {s: st.mean_validity
                 for s, st in per_sense_stats.items()
                 if st.max_validity >= tau}
    if not validated: return {}
    m = max(validated.values())
    raw = {s: math.exp((v - m) / temperature)
           for s, v in validated.items()}
    Z = sum(raw.values())
    return {s: r / Z for s, r in raw.items()}
```

Senses ohne ausreichend „starke" Explanation (max_validity < τ) fallen raus;
über den Rest wird ein Softmax über `mean_validity` mit Temperatur (default
0.5) gefahren. Output: `distributions_aggregate.jsonl`.

### Variante B — Comparative

Hier wird zusätzlich ein *zweiter* LLM-Schritt gefahren: pro
(Item, Validator-Modell) bekommt das Modell die besten Explanations je Sense
gezeigt und soll 100 Punkte auf die vier Senses verteilen. Diese
„Comparative"-Verteilungen werden in `comparative_results.jsonl` gespeichert
und über alle Validatoren gemittelt — Ergebnis in `distributions.jsonl`.
Das ist näher an der direkten Frage „Welcher Sense ist es?" und tendiert zu
weniger flacher Verteilung als Variante A.

### KL-Divergenz

```python
def _kld(p, q, eps=1e-9):
    return sum(pv * math.log((pv+eps)/(qv+eps))
               for k in set(p)|set(q)
               for pv, qv in [(p.get(k,0), q.get(k,0))]
               if pv > 0)
```

Wird gegen die Crowd-Verteilung berechnet und pro τ in `metrics_*.json`
abgelegt.

**Output-Schema `DistributionRecord`:** `item_id`, `candidate_senses`,
`per_sense_stats` (Dict von `SenseStats`), `llm_label_distribution_per_tau`
(Dict τ → Sense-Verteilung).

## Stage 5: Analyze

Zwei Analyse-Pfade laufen über demselben Run.

### `src/analyze.py` (über run_level1.py aufgerufen)

Schreibt pro Variante (A/B) nach `analysis_aggregate/` bzw.
`analysis_compare/`:

- `kld_curve.png` — KLD vs τ
- `validation_overlap.png` — Precision/Recall der LLM-validierten Senses
  gegen Crowd-Mehrheit, über τ gesweept
- `commission_error_candidates.csv` — nach `commission_score` ranked
- `cross_family_signal.json` + Scatter — gibt es systematische Validator-
  Stringenz-Unterschiede?
- `per_genre_breakdown.json` — Metriken nach Genre
- `worked_example.json` — ein Item komplett durchgespielt (Inputs,
  Explanations, Validator-Scores, finale Verteilung)
- `crowd_vs_llm_scatter.png`

### `scripts/analyze_results.py`

Externe, tiefere Analyse, Output nach `analysis_detailed/`. Sektionen:

- **A — LLM vs Crowd**: per-item JSD-Bar, Crowd↔LLM-Scatter (Sense/Genre),
  Kalibrierungsdiagramm, Per-Sense-Bias, Top-1-Confusion.
- **B — Zwischen Kategorien**: JSD-Boxplot nach Genre, Sense×Genre-Heatmap,
  Confusion nach Genre, Agreement↔JSD-Scatter.
- **C — Zwischen LLMs**: Mean-Distribution pro Modell, JSD-Boxplot pro
  Modell, paarweise Top-1-Übereinstimmung, Validator-Stringenz vs
  Generator-Quality, Abstention-Heatmap.
- **D — Variante A vs B**: Scatter JSD_A vs JSD_B, Bar-Vergleich aggregierter
  Metriken.
- **E — Risk-Quadranten**: 2×2-Scatter pro (Item, Sense). Crowd-Achse
  *count-basiert* gebinnt (Singleton-Vote = low, ≥2 Votes = high), LLM-Achse
  per Q25-Quantil. Punkte mit `crowd_prob == 0` sind ausgefiltert
  (keine Crowd-Aussage → keine Risk-Aussage). Ausgabe:
  `risk_quadrants.png` + `risk_quadrants.csv`.
- **F — Wikipedia-Gold**: lädt `reflabel` aus DiscoGeM-Roh-CSV, mappt
  PDTB-3-Tokens auf L1 (`PDTB_REFLABEL_TO_L1` im Script), hebt die
  Gold-Punkte auf dem Quadranten-Plot hervor und schreibt Statistiken
  (Anteil safe, „concerning items"). Ausgabe: `risk_quadrants_wiki_gold.png`,
  `wiki_gold_quadrants.csv`, `wiki_gold_stats.json`.

## Konfiguration

YAML, geladen über [src/config.py](../src/config.py). Top-Level-Keys:

```yaml
name: level1_experiment
models:
  generators:  [openai:gpt-4o-mini, anthropic:claude-haiku-4-5-20251001, ...]
  validators:  [...]
sampling:
  n_items:      45
  genre_split:  {Europarl: 15, Lit: 15, Wiki: 15}
  seed:         42
pipeline:
  tau_start: 0.1, tau_stop: 0.9, tau_step: 0.1
  softmax_temperature: 0.5
  max_retries: 5
  concurrency_per_provider: 4
  temperature_generate: 0.3
  max_tokens_generate: 1500
data:
  wide_csv:          "DiscoGeM 1.0_items/DiscoGeM1.0.wide.csv"
  full_csv:          "DiscoGeM 1.0_labels/DiscoGeMcorpus_fulldataset.csv"
  sense_definitions: "prompts/pdtb_sense_definitions.json"
  results_dir:       "results"
```

Reproduzierbarkeit: Config und Sense-Definitionen werden in das
Run-Verzeichnis kopiert; der Git-Hash steckt im `run_id`; Sampling-RNG
ist geseeded.

## Verzeichnis eines Runs

```
results/20260607T081000Z_level1_experiment_7108c04/
├── config.yaml                       # Snapshot der Config
├── pdtb_sense_definitions.json       # Snapshot der Sense-Definitionen
├── items.jsonl                       # Stage 1: 45 PrepItems
├── generations.jsonl                 # Stage 2: 720 GenerationRecords
├── validations.jsonl                 # Stage 3: ~10k ValidationRecords
├── distributions_aggregate.jsonl     # Stage 4 — Variante A
├── distributions.jsonl               # Stage 4 — Variante B
├── comparative_results.jsonl         # Rohe Comparative-Outputs
├── costs.csv                         # Telemetrie aller LLM-Calls
├── metrics.json
├── metrics_aggregate.json
├── metrics_compare.json
├── analysis_aggregate/               # Plots & CSVs für Variante A
├── analysis_compare/                 # Plots & CSVs für Variante B
└── analysis_detailed/                # Output von scripts/analyze_results.py
    ├── risk_quadrants.png|.csv
    ├── risk_quadrants_wiki_gold.png
    ├── wiki_gold_quadrants.csv
    ├── wiki_gold_stats.json
    └── … (Sektionen A–D)
```

## Eigenschaften zusammengefasst

- **Sprache:** Python ≥ 3.11, vollständig async.
- **Design:** stage-basierte offline-Pipeline, Stages über JSONL entkoppelt.
- **Provider-Strategie:** `LLMClient.call()` als einheitliches Interface,
  Modell-IDs als `"provider:model"`, Dispatch auf provider-spezifische
  Methoden, uniforme `LLMResponse`.
- **Resilience:** Tenacity-Backoff (2–30s, 5 Versuche) für transiente
  Fehler, harter Fail bei Auth-Fehlern.
- **Idempotenz:** Stages skippen, was schon in den JSONLs steht — gleicher
  CLI-Befehl resumed nach Crashes ohne Doppelkosten.
- **Reproduzierbarkeit:** Run-ID mit Git-Hash, Config-Snapshot, gesetzter
  Sampling-Seed.
- **Telemetrie:** `costs.csv` pro Call, `metrics_*.json` pro Variante.
