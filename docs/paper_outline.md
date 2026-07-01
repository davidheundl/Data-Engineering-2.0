# Paper Outline — EVADE auf DiscoGeM Level-1

> Arbeits-Notizen für ein ~5-seitiges Paper. Bullet-Punkte, keine
> ausformulierten Absätze. Statt Sätzen stehen hier nur Gedanken,
> Begründungen, Anknüpfungspunkte aus unseren Ergebnissen.
>
> Struktur folgt der Vorgabe:
> Abstract → Introduction (WHY) → Preliminaries → Problem (WHAT) →
> Main Approach (HOW) → Evaluation → Conclusions & Future Work.

## Title (Optionen, später entscheiden)

- *Commission Errors in LLM Discourse-Sense Annotation:
  Cross-Validated EVADE on the DiscoGeM 1.0 Corpus*
- *Does the LLM Agree with Itself? Cross-Family Validation for
  Implicit Discourse Relations*
- *Where LLMs Hallucinate Discourse Senses: A Risk-Quadrant Analysis
  on DiscoGeM*

## 0. Abstract (~150 Wörter)

- Setup: Implicit-Discourse-Relation-Labeling, vier PDTB-L1-Senses,
  DiscoGeM 1.0 als crowd-annotiertes Korpus mit Soft-Labels.
- Methode: EVADE-Style (mehrere Begründungen pro Kandidaten-Sense
  statt Ranking), erweitert um **Cross-Family-Validation** (Validator
  ≠ Generator, vier Provider).
- Was wir testen: Übereinstimmung der LLM-Verteilungen mit der Crowd,
  plus Verhalten gegen das partielle Editor-Gold (nur Wikipedia).
- Hauptergebnisse (Platzhalter, konkrete Zahlen nachpflegen):
  - KLD vs Crowd ≈ 0.58 (Variante B, mittel über τ)
  - Inter-Model-Fleiss-κ = 0.36 (mäßige Übereinstimmung →
    Cross-Family-Signal valid)
  - Wikipedia-Gold: 16/16 (100%) in Variante B im *safe*-Quadranten,
    15/16 (94%) in Variante A
- Vier Contributions (1 Satz/Item):
  1. Cross-Family-Validation-Erweiterung zu EVADE
  2. Risk-Quadrant-Analyse statt globaler KLD
  3. Aggregate- vs Comparative-Modus-Vergleich
  4. Wikipedia-Editor-Gold als externes Sanity-Signal

---

## 1. Introduction — *WHY* (≈ ¾ Seite)

> Warum ist das wichtig? Was ist die Motivation? Welche Lücke?

- **Discourse Relations** sind ein Kerntask im NLP — implizite Relationen
  besonders schwer, weil keine Connectives, mehrere Lesarten oft
  legitim, Crowd-Annotation zeigt deshalb Soft-Distributions statt
  Single-Labels.
- **Aktuelle Praxis**: LLMs werden zunehmend für linguistische
  Annotation eingesetzt — günstig, schnell, skalierbar.
- **Aktuelle Challenges**:
  1. *Commission Errors* — LLMs geben plausibel klingende Begründungen
     auch für falsche Senses. Klassische Accuracy verfehlt das.
  2. *Self-Confirmation Bias* bei LLM-as-Judge: gleicher Anbieter
     bewertet sein eigenes Output zu mild.
  3. *Soft-Label-Korpora* werden in der Evaluation häufig kollabiert
     („majority sense") — nutzt das volle Crowd-Signal nicht aus.
- **Was wir liefern**: Pipeline, die (a) explizit Commission-Errors
  sucht via EVADE, (b) Cross-Family-Validation einbaut, (c) die volle
  Crowd-Verteilung statt nur Majority benutzt.
- **Warum jetzt**: das Original-EVADE-Paper (*Long, Fan, Zhou et al.*,
  Citation einfügen) nennt Cross-Family-Validation explizit als
  offenen Punkt. Wir setzen genau diese Empfehlung um.

## 2. Preliminaries (≈ ½ Seite)

> Konzepte, die der Reader braucht, bevor wir loslegen. Kurz, präzise,
> kein Tutorial.

- **PDTB-3 Sense Hierarchy**: vier Level-1-Senses (temporal,
  contingency, comparison, expansion), darunter Level-2/3. Wir
  benutzen **Level-1**.
- **DiscoGeM 1.0**: crowd-annotiertes Discourse-Relation-Korpus,
  ~10 Annotatoren pro Item, drei Genres (Literatur, Europarl, Wikipedia).
  Items haben *Soft-Labels* (Sense-Verteilung über Annotatoren).
  Für Wikipedia-Items existieren zusätzlich editorische
  **Reference-Labels** (`reflabel`) — partielles Gold, das wir als
  externes Signal nutzen.
- **EVADE (Original)**: Statt das LLM einen Sense aus Kandidaten
  auswählen zu lassen, fragt EVADE für *jeden* Kandidaten-Sense alle
  plausiblen Begründungen ab. Diese werden anschließend validiert,
  über eine Validity-Schwelle τ gefiltert und in eine Verteilung
  überführt. Output ist eine Sense-Verteilung, nicht ein Label.
- **Commission Error**: das LLM produziert eine überzeugende
  Begründung für einen falschen Sense — gefährlich, weil es im
  Datensatz „intelligent" aussieht.
- **Cross-Family-Validation**: Validator-Modell ≠ Generator-Modell,
  und zusätzlich aus *anderer Modellfamilie* (Provider) — verhindert
  Self-Echo durch shared training data.
- **Notation**:
  - Crowd-Distribution: `p = (p_temporal, p_contingency, p_comparison,
    p_expansion)`
  - LLM-Distribution: `q` analog, abgeleitet aus Validity-Scores
  - Metriken: KLD(p || q), JSD, Top-1-Match, Fleiss' κ, Pearson r

## 3. Problem — *WHAT* (≈ ¼ Seite)

> Problem-Statement und Forschungsfragen.

- **Problem-Statement**: Gegeben Discourse-Relations-Items mit
  Soft-Labels aus DiscoGeM, wie zuverlässig lässt sich eine
  cross-family-validierte EVADE-Pipeline einsetzen, um (a) die
  Crowd-Verteilung zu approximieren und (b) Items zu identifizieren,
  bei denen das LLM commission errors begeht?
- **Research Questions** (3 Stück, explizit nummeriert):
  1. *RQ1 — Cross-Family*: Liefert Cross-Family-Validation
     informative (nicht-tautologische) Signale auf DiscoGeM-L1?
  2. *RQ2 — Modell- und Genre-Unterschiede*: Gibt es systematische
     Unterschiede zwischen den vier Providern und zwischen den
     drei Genres (Literatur, Europarl, Wikipedia)?
  3. *RQ3 — Risk-Klassifikation*: Können wir Items zuverlässig in
     Risk-Quadranten einordnen, und stimmen die so identifizierten
     „safe"-Items mit dem editorisch bestätigten Wikipedia-Gold
     überein?
- **Hypothesen** (kurz, prüfbar):
  - H1: Cross-Family-Fleiss-κ liegt deutlich unter 1 — sonst wäre
    Cross-Family redundant. (Erwartung: 0.3–0.5)
  - H2: Literatur-Genre liefert die niedrigsten Commission-Scores
    (LLMs „erfinden" am leichtesten plausible literarische Begründungen);
    Europarl die höchsten.
  - H3: Wikipedia-Gold-Senses landen mit hoher Quote (≥ 90%) im
    *safe*-Quadranten.

## 4. Main Approach — *HOW* (≈ 1.25 Seiten, mit Diagramm)

> Wie lösen wir das Problem? Hauptkomponenten und ihre Funktion.

### 4.1 Pipeline-Überblick

- 5-Stage-Async-Pipeline, JSONL-entkoppelt, alle Outputs
  reproduzierbar unter `results/{run_id}/`.
- **Diagramm einfügen** (siehe `docs/pipeline.md`).
- Stages: Prep → Generate → Validate → Aggregate/Compare → Analyze.

### 4.2 Cross-Family-Validation (Kernbeitrag #1)

- Vier Provider gleichzeitig: OpenAI, Anthropic, Mistral, DeepSeek
  — jedes Modell ist *sowohl* Generator als auch Validator.
- Constraint im Code:
  ```python
  for validator_model in config.models.validators:
      if validator_model == gen.generator_model:
          continue   # niemals Self-Validation
  ```
- Rechtfertigung: Self-Validation hatte das Original-EVADE als
  Bias-Risiko genannt — vier disjunkte Trainingsregimes als
  natürliches Anti-Bias-Mittel.

### 4.3 Zwei Aggregations-Varianten (Kernbeitrag #2)

- **Variante A — Aggregate**: für jeden Sense die SenseStats
  (max_validity, mean_validity, validator_std). Tau-Sweep filtert
  Senses mit max_validity ≥ τ, Softmax über mean_validity.
  → Verteilungs-Sicht, behält Unsicherheit.
- **Variante B — Comparative**: zweiter LLM-Schritt — Validator
  sieht die besten Explanations pro Sense und verteilt 100 Punkte
  auf die vier Senses. Über Validatoren gemittelt.
  → Näher an direkter Klassifikation, schärfere Verteilungen.
- Beide werden parallel gerechnet, beide Ergebnisse berichtet —
  wir nehmen *keine* Variante als „die Richtige" an.

### 4.4 Risk-Quadrant-Analyse (Kernbeitrag #3)

- Per (Item, Sense) zwei Achsen:
  - Crowd-Achse (X): **count-basiert** gebinnt. Singleton-Vote
    = low, ≥ 2 Votes = high. Begründung: 1/9 = 0.111 vs 1/10 = 0.10
    sind semantisch beide „eine Person hat das gewählt" — ein
    Probability-Quantil-Cutoff klassifiziert sie inkonsistent.
  - LLM-Achse (Y): Q25-Cutoff über LLM-Probability.
- Vier Quadranten:
  - `high_risk` (low/low) — beide unsicher
  - `llm_overconfident` (crowd low, llm high) — *Commission-Error-
    Kandidaten*
  - `llm_underconfident` (crowd high, llm low)
  - `safe` (high/high)
- Punkte mit `crowd_prob == 0` ausgefiltert (keine Crowd-Aussage →
  keine Risk-Aussage).

### 4.5 Wikipedia-Gold-Anchoring (Kernbeitrag #4)

- DiscoGeM liefert für Wiki-Items editorische Reference-Labels
  (`reflabel`). PDTB-3-Level-2/3-Tokens → mit fester Map
  `PDTB_REFLABEL_TO_L1` (24 Tokens, in `scripts/analyze_results.py`)
  auf L1-Senses.
- Gold-Punkte werden auf dem Risk-Quadranten überlagert — externer,
  von der Crowd unabhängiger Sanity-Check.

### 4.6 Engineering-Details (Reliability + Reproducibility)

> Kurz, aber drin — gehört zur „How" und ist für Reviewer wichtig.

- **Exponential Backoff** via tenacity:
  `wait_exponential(multiplier=1, min=2, max=30)`, 5 Versuche, nur
  auf `LLMError`-Klasse (429er, 5xx, Timeout). `FatalLLMError`
  (Auth) bricht sofort ab.
  - Motivation: Mistral-Free-Tier-429er waren ohne Backoff
    Show-Stopper.
- **Per-Provider-Semaphores** (Mistral=1, Rest=4) — Concurrency-
  Begrenzung.
- **Idempotenz**: jede Stage skippt Tripel, die schon in der
  JSONL des Outputs stehen — Re-Runs nach API-Crashes ohne
  Doppelkosten.
- **Reproduzierbarkeit**: Run-ID = `{utc-ts}_{config}_{git-hash}`,
  Config + Sense-Definitionen als Snapshot im Run-Verzeichnis,
  gesetzter Sampling-Seed (42).

## 5. Evaluation (≈ 1.5 Seiten — größter Block)

> Metriken, Baselines, Hauptresultate. Nach den drei RQs strukturiert.

### 5.1 Experimental Setup

- **Daten**: 45 Items, stratified (15 je Genre × Agreement-High/Low).
- **Modelle** (Tabelle):

  | Rolle | Modell | API-Preis (in/out per 1M) |
  |---|---|---|
  | Gen + Val | openai:gpt-4o-mini | $0.15 / $0.60 |
  | Gen + Val | anthropic:claude-haiku-4-5 | $1.00 / $5.00 |
  | Gen + Val | mistral:mistral-small-latest | $0.20 / $0.60 |
  | Gen + Val | deepseek:deepseek-chat | $0.27 / $1.10 |

- **Hyperparameter**: τ ∈ {0.1, …, 0.9} step 0.1, Softmax-T = 0.5,
  Generate-T = 0.3, max-tokens-generate = 1500.
- **Metriken**:
  - vs Crowd: **KLD**, **JSD**, **MAE**, **RMSE**, **Top-1-Accuracy**
  - intern: **Fleiss' κ** (Modelle), **Validator-Std**,
    **Commission-Score**
  - vs Editor-Gold: Quadranten-Verteilung
- **Baselines**:
  - (a) **Single-Model Top-1**: bestes Einzelmodell ohne Validation
    (Mean-Crowd-JSD aus `per_model_jsd_boxplot.png`).
  - (b) **Variante A vs B** als interner Vergleich.
  - (c) Optional / wenn Budget reicht — **Cross-Family-Off-Ablation**:
    Validator = Generator erlaubt. Würde RQ1 direkt prüfen.
- **Total Cost** für den 45-Item-Run: **$2.44**, davon Anthropic
  $1.68, DeepSeek $0.39, Mistral $0.26, OpenAI $0.21.

### 5.2 RQ1 — Cross-Family-Signal

- **Fleiss' κ über Modelle = 0.36** (mäßig) — bestätigt H1: die vier
  Modelle disagreen substanziell, der Cross-Family-Constraint
  produziert nicht-redundante Information.
- Inter-Model-Agreement-Plot (`inter_model_agreement.png`) zeigt
  paarweise Top-1-Übereinstimmung — größte Distanz zwischen
  Mistral und Claude.
- Validator-Stringency-Plot (`validator_generator_quality.png`):
  ist *ein* Validator systematisch milder/strenger? Beobachtung
  einfügen.
- *(falls Ablation gemacht)*: Cross-Family-Off liefert höhere mean
  Validity → bestätigt Self-Confirmation-Bias.

### 5.3 RQ2 — Modell- und Genre-Unterschiede

- **Pro Modell**:
  - Mean-Distribution pro Modell (`per_model_mean_distribution.png`):
    Welches Modell überrepräsentiert welchen Sense?
  - Per-Model-JSD-Boxplot: welches Modell ist näher an der Crowd?
- **Pro Genre** (Commission-Scores aus `metrics.json`):
  - Lit: **0.255** (niedrigster)
  - Wiki: **0.325**
  - Europarl: **0.393** (höchster)
  - Bestätigt H2 (Literatur leicht, Europarl schwer).
- Agreement↔JSD-Scatter (`agreement_vs_jsd_scatter.png`): niedrigere
  Crowd-Einigkeit korreliert mit höherer LLM-Crowd-Divergenz
  (Spearman ρ einfügen).

### 5.4 RQ3 — Risk-Klassifikation + Wiki-Gold-Anchor

- **Quadranten-Verteilung** (126 Punkte nach Crowd-Prob-> 0-Filter):

  | Quadrant | Variante A | Variante B |
  |---|---|---|
  | high_risk | 18 | 19 |
  | llm_overconfident | 31 | 30 |
  | llm_underconfident | 14 | 13 |
  | safe | 63 | 64 |

  - ~25% sind `llm_overconfident` — genau die Klasse, die EVADE
    explizit zu erkennen versucht.

- **Wiki-Gold-Anchoring** (16 Gold-(Item, Sense)-Punkte):
  - Variante A: 15 *safe*, 1 *llm_underconfident* (wiki_089) → 94%
  - Variante B: 16/16 *safe* → 100%
  - **Bestätigt H3**.
  - Caveat: nur 16 Punkte — Sanity-Check, kein statistischer Test.

- Plot: `risk_quadrants_wiki_gold.png` (im Paper zeigen).

### 5.5 Variante-A-vs-B-Befund

- B liefert schärfere Verteilungen (niedrigere mittlere Entropie),
  bessere Top-1-Accuracy.
- A behält mehr Unsicherheit, robuster bei flachen Crowd-Verteilungen.
- → keine universelle Gewinnerin; Empfehlung:
  **B fürs Labeling, A für Uncertainty-Quantification**.

### 5.6 Kosten-Effizienz (kurz)

- 45 Items, ~10.7k Calls, ~5.2M Tokens, **$2.44 total** — die Pipeline
  ist mit modernen Cheap-Tier-Modellen tragbar selbst für Pilot-
  Experimente. Skalierung auf 500 Items: ≈ $25 geschätzt.

## 6. Conclusions & Future Work (≈ ½ Seite)

> Lessons Learned + Was wäre die nächste Iteration?

### Lessons Learned

- **Cross-Family-Validation funktioniert technisch** (κ = 0.36) und
  ist günstig — die Provider-Mischung ist Wert für sich, nicht nur
  Risk-Hedge.
- **Risk-Quadranten machen Commission-Error sichtbar**: ~25% der
  Punkte sind `llm_overconfident` — wären in einer reinen
  Accuracy-Metrik unsichtbar.
- **Variante A vs B** sind komplementär, nicht konkurrierend.
- **Wikipedia-Gold-Anchor** ist klein aber wertvoll: 100% safe in
  Variante B → die *safe*-Klassifikation ist konsistent mit
  Editor-Annotation.
- **Engineering-Lesson**: Idempotente JSONL-Pipelines + Tenacity-
  Backoff sind nicht „nice to have" — ohne sie wäre der Run an
  Mistral-429ern komplett gescheitert.

### Future Work

- **Scale-up**: N = 200–500 Items, mit größeren Modellen (Claude
  Sonnet, GPT-4o Full) als zusätzliche Validatoren.
- **Cross-Family-Ablation**: gleicher Validator = Generator
  *erlauben*, dieselben Items rechnen, A/B-Vergleich. Das wäre der
  Knockdown-Beweis für den Wert des Constraints.
- **L2-Senses**: feinere Klassifikation, aber dann braucht es mehr
  Items pro Sense (Power-Issue).
- **Active Human-in-the-Loop**: `high_risk`-Items automatisch zur
  manuellen Annotation routen — kostenoptimiertes Labeling.
- **Editor-Gold über Wiki hinaus**: PDTB-3-Items oder weitere
  Subsets in DiscoGeM, die partielles Gold haben.
- **Other tasks**: Coreference, Argument Mining — gleiche Pipeline-
  Struktur, anderer Sense-Set.

---

## TODO vor dem Schreiben

- [ ] Original-EVADE-Paper-Citation (Long/Fan/Zhou et al. — exakte
      Referenz)
- [ ] DiscoGeM-Paper (Scholman et al. 2022)
- [ ] PDTB-3-Citation (Webber/Prasad et al. 2019)
- [ ] Aus `analysis_detailed/aggregate_metrics.json` exakte Zahlen
      ziehen → in Abstract + Result-Bullets einsetzen (KLD/JSD/Top-1
      pro Variante)
- [ ] Plot-Shortlist fürs Paper (max 5, Rest in Appendix):
      Risk-Quadranten, Wiki-Gold, KLD-vs-τ, Per-Model-JSD-Boxplot,
      Agreement↔JSD-Scatter
- [ ] Entscheiden: Variante A vs B als „ablation" oder als „two
      methods" framen
- [ ] *Empfohlen*: Cross-Family-Off-Ablation laufen lassen
      (~$2–3 zusätzlich) — würde RQ1 direkt beweisen statt nur
      indirekt
