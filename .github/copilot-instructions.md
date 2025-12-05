# GitHub Copilot Instructions for TSAB Project

## Project Snapshot
- MOEA/D for association rule mining on diabetes data, built on `pymoo` with custom diploid genomes, adaptive operators (1/5 rule), strict validation, and cached metrics.

## Run It
- `python main.py` → CLI lists configs in `config/*.json`; `config/general/base_config.json` merges dataset paths automatically.
- Dependencies: numpy, pandas, pymoo, matplotlib, seaborn; `main.py` checks before running.
- Outputs land in `results/<experiment>/exp_<id>/` with `config_snapshot.json`, per-gen `populations/` and `pareto/`, aggregated `discarded/`, `stats/evolution_stats.csv`, optional `plots/`, and deduped `final_pareto.csv` plus `final_pareto_historical.csv`.

## Data Prep
- Process raw CSV + metadata with `src/preprocessing.py` (filters, rediscretization, one-hot decode, integer encoding, distribution reports) then `src/calculate_supports.py` to write `supports.json` (and `sample_supports.json`).
- To speed runs, create `data/processed/sample/*` via `src/sampling.py` (stratified two-level sample, validation report) and set `use_sampling` in config.

## Architecture Flow
- `main.py` → `Orchestrator`: loads config+base paths, picks full vs sample data, allocates `results/.../exp_XXX`, configures logging, snapshots config.
- `Orchestrator.run` instantiates `MOEAD_ARM`, `ARMCallback`, runs optimization, saves deduped Pareto (by flattened genome), consolidates historical Pareto, then triggers `VisualizationManager`.
- `MOEAD_ARM` wires `ARMMetrics`, `ARMValidator`, `DiscardedRulesLogger`, `ARMProblem`, operators, decomposition, termination, and runs `AdaptiveMOEAD`.

## Representation & Operators
- Genome `X`: length `2*num_genes` (roles then values). Roles: 0 ignore, 1 antecedent, 2 consequent. `initialize` favors sparsity (60% ignore); always call `repair` after edits to keep role/value consistent.
- `ARMSampling` builds the initial population with validation and duplicate shielding; after max attempts it fills remaining slots with duplicates to satisfy size.
- `DiploidNPointCrossover` swaps aligned role/value blocks.
- `ARMMutation` supports `extension`, `contraction`, `replacement`, with validation retries (`repair_attempts`) and duplicate retries (`duplicate_attempts`); failed attempts are logged via `DiscardedRulesLogger`.
- `AdaptiveMOEAD` overrides `_next` to perform mating/eval/update manually, caps neighbor replacements (`replacement_size`), and adapts mutation/crossover probabilities each gen using the 1/5 success rule.

## Validation & Metrics
- `ARMValidator` checks non-empty/disjoint antecedent/consequent, item count bounds, exclusions (`fixed_consequents`, `forbidden_pairs`), then metric thresholds; alias map maps `kappa` → `k_measure`.
- `ARMMetrics` caches by rule signature; singletons pull from `supports.json`, joints from the DataFrame. Metrics returning `None` (e.g., zero support) or validation failures get penalized with `F=2.0` in `ARMProblem`; objectives are negated because pymoo minimizes.

## Configuration Hotspots
- Experiment configs (`config/escenario_*.json`) set objectives, operator probabilities (min/max for adapters), neighborhood size/replacements, logging interval, termination (optional early stop ftol/period), and max init attempts.
- `constraints.metric_thresholds` must use canonical or alias keys; update `ARMMetrics.alias_map` and validator thresholds together when adding metrics.

## Outputs & Debugging
- `callback.py` logs per-interval stats (objective min/mean/max, HV, adaptive probs), saves populations/Pareto with decoded rules and encoded genome, and differential discarded logs. `DiscardedRulesLogger.save` aggregates reasons/metrics sorted by frequency.
- Plots (`visualization.py`) cover metric evolution, hypervolume, discarded reasons, and Pareto fronts (2D/3D/parallel) using stats and pareto CSVs.

## Developer Tips
- Keep `metadata.feature_order` aligned with encoded CSV columns; misalignment breaks decoding/support lookup.
- When changing genome structure or operators, preserve diploid layout and call `repair` after mutations/crossovers to avoid invalid roles/values.
- Hypervolume reference point assumes objectives normalized to [-1,0] in minimization space; be mindful if adding new metrics with different ranges.