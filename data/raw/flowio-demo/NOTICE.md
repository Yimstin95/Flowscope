# Source & license

`pbmc_13color_facsaria.fcs` (originally `100715.fcs`) is bundled test data from
the [FlowIO](https://github.com/whitews/FlowIO) Python library
(`data/fcs_files/100715.fcs` on the `master` branch), the same library
FlowScope uses to parse FCS files.

- **Instrument:** BD FACSAria
- **Panel:** 13-color — KI67, CD3, CD28, CD45RO, CD8, CD4, CD57, viability
  dye/CD14, CCR5, CD19, CD27, CCR7, CD127
- **Events:** 65,016
- **Compensation:** embedded spillover matrix (no external CSV needed)
- **License:** BSD 3-Clause (FlowIO's license), copyright Scott White — permits
  redistribution, which is why this file is committed directly to the repo
  rather than git-ignored like the larger FR-FCM-ZZEB dataset.

Chosen as a second example dataset because it downloads reliably with a plain
`curl`/`git clone` — no CAPTCHA, no login wall, no expired-certificate issue —
unlike flowrepository.org (see the main README's Data Source section).
