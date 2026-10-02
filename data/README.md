# Data

## cpt_query_validation_dataset.csv

145 physician-authored cohort queries with included and excluded CPT codes.
Queries were written by a licensed physician and reviewed independently by a graduate researcher.
Queries with more than 10 ground-truth codes are removed at load time, leaving 127.

## web_cpt_codes.xlsx (not included)

The CPT corpus is not redistributed because CPT descriptions are copyrighted by the
American Medical Association. Obtain the CPT code set under an appropriate license and
save it here as `web_cpt_codes.xlsx` with the columns:

| column | example |
|---|---|
| `code` | `27447` |
| `system` | `CPT4` (other rows are ignored) |
| `display` | `Arthroplasty, knee, condyle and plateau; medial AND lateral compartments ...` |
