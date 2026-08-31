# Metadata completion checklist

Complete `sample_metadata.tsv` before any formal statistical analysis.

Required fields:

- `library_id`: exact directory/library name
- `group`: Y, OC, or OT
- `age_months`: 4 or 10 under the current animal design
- `treatment`: vehicle or MRJP1
- `dose_mg_kg`: 0 or 200
- `batch`: library preparation or sequencing batch; do not invent one
- `estrous_stage`: terminal stage if recorded; otherwise keep `unknown`
- `pool_mouse_ids`: all mouse IDs contributing to that library

If batch is perfectly confounded with group, it cannot be statistically
separated from biology and must not be treated as a removable batch effect.
