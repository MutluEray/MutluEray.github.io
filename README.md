# Eray Mutlu – Research Portfolio

Professional academic / research portfolio website.

**Live site (once deployed):** https://MutluEray.github.io

## Structure

- `index.html` – Home / research statement + highlights
- `research.html` – Ongoing research (Ear-EEG drowsiness, XAI for ECG)
- `projects.html` – Technical projects (TinyML stethoscope, optical imaging)
- `publications.html` – Publication list
- `about.html` – Education, experience, skills, awards
- `contact.html` – Contact information and links

## How to deploy on GitHub Pages

1. Make sure the repository is named **exactly** `MutluEray.github.io` (or the username matching your GitHub account).
2. Push all files in this folder to the `main` (or `master`) branch of that repository.
3. Go to the repository **Settings → Pages**.
4. Under “Source”, select the branch `main` and folder `/ (root)`.
5. Save. The site will be available at `https://MutluEray.github.io` within a few minutes.

### Quick push commands (from this folder)

```bash
git init
git add .
git commit -m "Initial portfolio site"
git branch -M main
git remote add origin https://github.com/MutluEray/MutluEray.github.io.git
git push -u origin main
```

(If the repo already exists and is empty, the above works. If you already have a README, pull first or force carefully.)

## Customization notes

- Color accent is teal (`#0d9488`). Change in the Tailwind config inside each HTML file if desired.
- ORCID link is commented out in `contact.html` – uncomment and add the correct URL when ready.
- Add a professional photo later by placing an image in the root and linking it on the Home or About page.
- For a downloadable CV, place a PDF in the root and add a link on the About or Contact page.

## Tech

- Pure static HTML
- Tailwind CSS via CDN (no build step)
- Responsive, clean academic design
- Mobile-friendly navigation
