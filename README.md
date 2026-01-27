# edca
Early Design Embodied Cost and Carbon Tool


┌─────────────────────────────────────────────────────┐
│                     USER INPUTS                     │
├─────────────────────────────────────────────────────┤
│ setup/control_file.yaml                             │
│ setup/control_file.csv                              │
│ setup/control_file.txt                              │
│  • building type / location                         │
│  • floor count, heights                             │
│  • grid spans (1W / 2W / irregular / polygon)       │
│  • unit system                                      │
│  • depth limits                                     │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│                INPUT PARSING & VALIDATION           │
├─────────────────────────────────────────────────────┤
│ • Parse control file (YAML preferred)               │
│ • Validate with Pydantic schema                     │
│ • Convert all units → SI                            │
│ • Expand floor-by-floor programs                    │
│ • Resolve load program keys                         │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│               PRESET DATABASE (READ ONLY)           │
├─────────────────────────────────────────────────────┤
│ data/presets/csvs/                                  │
│  • occupancies.csv                                  │
│  • materials.csv                                    │
│  • systems.csv (→ systems_catalog)                  │
│                                                     │
│ data/presets/yaml/                                  │
│  • loads.yaml                                       │
│  • version.yaml                                     │
│                                                     │
│ data/epds/                                          │
│  • EPD PDFs / references                            │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│              LOAD RESOLUTION & COMBINATIONS          │
├─────────────────────────────────────────────────────┤
│ • Match occupancy → load case                       │
│ • Apply EC / ASCE factors                           │
│ • Compute floor-specific load envelopes             │
│ • Output canonical load objects                     │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│           SYSTEMS CATALOG FILTERING (FAST)           │
├─────────────────────────────────────────────────────┤
│ • Filter by:                                        │
│    - span behavior (1W / 2W)                         │
│    - max span ≥ required                            │
│    - depth limits                                   │
│    - load compatibility                             │
│    - material flags                                 │
│                                                     │
│ → returns FEASIBLE system rows                      │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│            QUANTITY & ASSEMBLY DEFINITION            │
├─────────────────────────────────────────────────────┤
│ • Extract per-m² material volumes                   │
│ • Resolve material IDs → properties                 │
│ • Apply static rules (rebar tables, etc.)            │
│ • Build "assembly" objects                          │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│             CARBON & COST CALCULATION                │
├─────────────────────────────────────────────────────┤
│ • A1–A3 (materials)                                 │
│ • A4 (transport)                                    │
│ • A5 (construction)                                 │
│ • Cost per m²                                       │
│ • Store full material breakdown                     │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│          SPAN SWEEP & OPTIMIZATION (OPTIONAL)        │
├─────────────────────────────────────────────────────┤
│ • Increase span in increments (e.g. 0.5 m)          │
│ • Re-evaluate quantities & carbon                   │
│ • Apply penalties (beam/column growth)              │
│ • Store all candidate rows                          │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│               GROUPING & RANKING                     │
├─────────────────────────────────────────────────────┤
│ • Group by typology / material                      │
│ • Select minimum-carbon span per group              │
│ • Rank by carbon (tie-break cost)                   │
│ • Produce "optimized set"                           │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│                 CODE CHECKS                          │
├─────────────────────────────────────────────────────┤
│ • Floor checks                                      │
│ • Beam checks                                       │
│ • Column checks                                     │
│ • Pass/fail with reasons                            │
│ • Optional re-run with constrained set              │
└───────────────┬─────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────┐
│                 OUTPUTS & REPORTING                  │
├─────────────────────────────────────────────────────┤
│ • Ranked systems                                    │
│ • Carbon & cost tables                              │
│ • Span optimization summary                         │
│ • Material takeoffs                                 │
│ • Charts & figures                                  │
│ • CSV / JSON / Parquet exports                      │
└─────────────────────────────────────────────────────┘
