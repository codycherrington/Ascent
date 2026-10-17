# Ascent docs

Project documentation. Tracked docs are listed first; the local-only files are gitignored and a fresh clone will not have them.

## Contents

- [goal.md](goal.md) - the first goal (fluent English on TinyStories), non-goals and success criteria.
- [architecture.md](architecture.md) - technical reference for how the code base is structured: pipeline, files, model, trainer, run records, tests.
- [run-reports.md](run-reports.md) - how to read a run's `report.html` / `report.md`: sections, automatic flags and thresholds, text stats, regenerating.
- [tokenizer.md](tokenizer.md) - byte-level BPE tokenizer rationale, usage and measured stats.
- [datasets/tinystories.md](datasets/tinystories.md) - TinyStories dataset card (source, license, filtering, counts).
- [../eval/README.md](../eval/README.md) - evaluation protocol; also [../eval/rubric.md](../eval/rubric.md) (fluency rubric) and `eval/prompts.txt` (20 fixed prompts).

Local only (gitignored):

- `solutions-log.md` - chronological log of problems, causes, solutions and decisions, kept so the project's story can be assembled later.
- `legacy-issues-export.md` - backup export of the old issues.

## Adding an entry to the solutions log

1. Open `solutions-log.md` and copy the template from the "Entry template" section.
2. Get the timestamp with `date '+%Y-%m-%d %H:%M %Z'`. Always include both date and time, with timezone.
3. Add the entry at the bottom of the log. Newest entries go last.
4. Fill in all fields. Mention related issue and PR numbers and commit hashes under **Links**.
5. When an entry's status changes, update its **Status** line and add a note rather than rewriting history.
