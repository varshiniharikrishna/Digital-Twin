# Security notes

## Credentials

Never commit a database password or a populated `.env` file. The application reads `DT_MYSQL_USER` and `DT_MYSQL_PASSWORD` from the environment that launches the app. Rotate any credential that was ever committed; removing it from the current files does not invalidate a leaked password.

## Purge a previously committed secret

History rewriting changes commit IDs and affects every collaborator. Coordinate first, back up the repository, and do not run this automatically. After rotating the credential, use `git-filter-repo` with a local replacement file containing the exact old secret (keep that file outside the repository and delete it after use):

```sh
python3 -m pip install git-filter-repo
git clone --mirror https://github.com/varshiniharikrishna/Digital-Twin.git Digital-Twin-clean.git
cd Digital-Twin-clean.git
git filter-repo --replace-text /secure/path/replacements.txt
git push --force --mirror origin
```

Replacement-file format for `git-filter-repo`: `literal:OLD_SECRET==>***REMOVED***`. If only a particular file/path should be removed, inspect `git filter-repo --help` and use `--path`/`--invert-paths` on a mirror clone. Rewriting the remote is not part of routine pushes. Notify collaborators to reclone after an approved rewrite.

## Application data

Use least-privilege database accounts, restrict database network access, and avoid putting patient identifiers or health details in logs. The bundled data are synthetic; clinical deployment requires a reviewed privacy, security, and regulatory plan.
