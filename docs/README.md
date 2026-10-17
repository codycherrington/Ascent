# Ascent docs

This folder holds project documentation that is not code. Right now that is one thing: a dated log of problems we ran into and the solutions we worked out, kept so the project's story can be assembled later.

## Contents

- `architecture.md` - technical reference for how the code base is structured: pipeline, files, model, trainer, run records, tests.
- `solutions-log.md` - chronological log of problems, causes, solutions and decisions. This is a local-only file: it is listed in `.gitignore` and not committed, and is kept in this folder for the project's story. A fresh clone will not have it.

## Adding an entry

1. Open `solutions-log.md` and copy the template from the "Entry template" section.
2. Get the timestamp with `date '+%Y-%m-%d %H:%M %Z'`. Always include both date and time, with timezone.
3. Add the entry at the bottom of the log. Newest entries go last.
4. Fill in all fields. Mention related issue and PR numbers and commit hashes under **Links**.
5. When an entry's status changes, update its **Status** line and add a note rather than rewriting history.
