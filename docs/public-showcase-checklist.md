# Public showcase checklist

The folder `showcase/` is the only tree that should become a public repository. This file stays in the private repository.

The current GitHub repo, `AnashehOhanian/enterprise-rag-assistant`, is public and already contains application source. Make that repo private before pushing the estimator. Then create a new public repo from `showcase/` only.

## Keep private

- All Python: `app.py`, `safeguard_ui.py`, `safeguard_catalog.py`, `rag_engine.py`, `app_ui.py`
- Prompts and image-edit instructions inside `app.py`, including the client-chat instructions
- `shop_requests.jsonl` and `shop_requests/`, the shop copy of questions and customer photos
- Price bands, pounds-per-foot, style minimums, and takeoff formulas in `safeguard_catalog.py`
- Shop-drawing dimensions, cut rules, and the drawing page in `app.py`
- `.env`, `.env.example`, and `.streamlit/secrets.toml`
- `requirements.txt` and `requirements-rag.txt`
- The developer `README.md` at the repository root
- `templates/`, tests, and the local `venv/`
- Customer photos and any screenshot that shows a rate table, a dollar bracket, a key field, or a real property
- Git history of this repository if a secret was ever committed

## Safe to publish

Publish by copying `showcase/` into a new, empty public repository. Do not push this repository and then delete files. History would still hold them.

- `showcase/README.md`
- `showcase/.gitignore`
- `showcase/screenshots/` with interface captures you have checked by hand
- A one-line note that the source is available on request

A safe screenshot shows the project picker, an empty takeoff form, or the shop-drawing viewer with sample measurements and no dollar amounts.

## Before the public repo goes live

1. Make `enterprise-rag-assistant` private on GitHub. Check for public forks.
2. Confirm `.env` and `.streamlit/secrets.toml` are untracked: `git status` must not list them.
3. Create a new public repository, for example `safeguard-4u-estimator`.
4. Copy only the contents of `showcase/` into that new repo. The public root README is `showcase/README.md`.
5. Read the public README once more and confirm it has no dollar amounts, no prompt text, and no key.
6. Push the new repo. Leave this repo private.

```powershell
mkdir ..\safeguard-4u-estimator
Copy-Item -Recurse .\showcase\* ..\safeguard-4u-estimator\
cd ..\safeguard-4u-estimator
git init
git add README.md .gitignore screenshots
git status
git commit -m "Add a public overview of the Safeguard 4U estimator."
git remote add origin https://github.com/AnashehOhanian/safeguard-4u-estimator.git
git push -u origin main
```

Stop at `git status` if any `.py`, `.env`, or `secrets.toml` file is staged.
