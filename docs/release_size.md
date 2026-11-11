# Release size budget

The release verifier enforces a 90 MiB maximum for any individual tracked file and a 900 MiB total checkout budget. Run `python scripts/verify_release.py` to inspect the current values. Generated training outputs, execution checks and caches are excluded by `.gitignore`.

GitHub blocks regular Git files above 100 MiB and limits an individual push to 2 GiB. It recommends keeping repositories below 1 GB when possible. The supplied scientific artifacts fit within the project budget and use regular Git storage.

Sources are [GitHub file and repository size guidance](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github) and [the push limit documentation](https://docs.github.com/en/get-started/using-git/troubleshooting-the-2-gb-push-limit), checked on 11 November 2026.
