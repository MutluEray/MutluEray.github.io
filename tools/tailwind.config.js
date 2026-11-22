// Rebuild assets/tailwind.css after adding new Tailwind classes (run from the repo root):
//   npx tailwindcss@3 -c tools/tailwind.config.js -i tools/tailwind.input.css -o assets/tailwind.css --minify
module.exports = {
  content: ['./*.html', './projects/*.html'],
  theme: {
    extend: {
      colors: {
        accent: { 50: '#EAF4F2', 100: '#CFE6E2', 400: '#3E9A92', 500: '#0E7C76', 600: '#0B6B66', 700: '#08524E' },
        navy: { 800: '#22313B', 900: '#15232D' }
      },
      fontFamily: {
        sans: ['"IBM Plex Sans"', 'system-ui', 'sans-serif'],
        serif: ['Newsreader', 'Georgia', 'serif']
      }
    }
  }
};
