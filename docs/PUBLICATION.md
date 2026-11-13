# Publication scope

This folder is a standalone scientific code and results selection. It contains
simulation equations, model implementations, original arrays and trained weights,
training/evaluation runners, methods, regression tests and numerical audits.
The author's GitHub account is `jogecodes`.

Numerical arrays and checkpoint bytes retain their original hashes. Changes to
portable runners and scientific metadata are identified in the provenance
inventory. Full predictions remain intact, including invalid and nonfinite
states; scientific metrics never silently replace them with shorter forecasts.

Excluded material includes the web application and all of its playback exports,
exporters, UI tests, manifests, translations and hosting files. Also excluded
are private Git history, credentials, caches, internal handoff notes, intermediate
development backup ZIP files and regenerable result animations/plots/logs.
Small source-only archives from the original scientific studies are retained.

This is not a complete backup of every historical development artifact. Preserve
a separate archive of the original workspace before deleting it. Selected
scientific datasets/checkpoints are intentionally versionable; `.gitignore`
does not blanket-ignore `results/`, `.npz` or `.pt`. Fresh outputs belong under
`reproductions/`.

The preparation audit records what was checked. The checksum inventory detects
changes to the supplied selection; it is not a signed authenticity guarantee.
Publication target: https://github.com/jogecodes/chaos-library.
This scientific selection has a fresh local Git history; remote creation and upload require an authenticated GitHub connection.
