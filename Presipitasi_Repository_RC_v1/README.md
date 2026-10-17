# Data-Center Water-Demand Exposure under Low-Flow Stress — Reproducibility Release Candidate

This repository is a **release-candidate package** prepared for public deposition of the analysis
scripts and shareable derived outputs supporting the manuscript targeted to *Jurnal Presipitasi*.

## Important status

**No public repository is claimed by the manuscript until this package has been uploaded to a
persistent repository (for example GitHub with a tagged release and/or Zenodo) and the resulting
persistent URL/DOI has been checked.**

## Contents

- `src/` — analysis and quality-control scripts available in the release candidate.
- `derived/` — shareable derived output currently available in this working package.
- `figures/` — shareable figure outputs.
- `metadata/` — GloFAS access date, DOI, and data-use terms.
- Raw GloFAS monthly files are intentionally not included; retrieve them from the official EWDS
  source and follow the applicable CEMS-FLOODS licence.

## Reproduction boundary

The manuscript analysis uses GloFAS v5 Historical for January 2019–July 2026, with a validated
969-cell spatial mask and a regional spatial-mean discharge indicator. The repository should be
augmented before publication with the final acquisition manifest(s), extraction metadata, complete
shareable result tables, environment specification, and a tagged release corresponding exactly to
the submitted manuscript.

## Suggested persistent-release workflow

1. Create a public repository.
2. Upload this package contents.
3. Add the final acquisition manifests and all shareable derived result tables that can legally be
   redistributed.
4. Add `requirements.txt` or an equivalent locked environment specification.
5. Tag the exact release used for submission.
6. If using Zenodo, archive that release and record the DOI.
7. Open the public URL and verify that the files, README, licence notes, and release/DOI resolve.
8. Only then replace the manuscript's repository placeholder with the verified persistent URL/DOI.

## Source dataset

GloFAS v5 Historical, Copernicus Emergency Management Service Early Warning Data Store.
DOI: 10.24381/cds.a4fdd6b9.

See `metadata/DATASET_ACCESS_AND_TERMS.md` for access date and licensing notes.
