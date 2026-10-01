// Swap the poster for the YouTube player on click, so nothing loads from YouTube until then.
// Without JavaScript the poster is a plain link to the video.
document.querySelectorAll('a.video[data-video]').forEach((a) => {
  a.addEventListener('click', (e) => {
    e.preventDefault();
    const f = document.createElement('iframe');
    f.src = `https://www.youtube-nocookie.com/embed/${a.dataset.video}?autoplay=1&rel=0`;
    f.title = 'What is OpenRL?';
    f.allow = 'autoplay; encrypted-media; picture-in-picture; fullscreen';
    f.allowFullscreen = true;
    const box = document.createElement('div');
    box.className = 'video';
    box.appendChild(f);
    a.replaceWith(box);
  }, { once: true });
});
