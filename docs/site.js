const filterButtons = Array.from(document.querySelectorAll('[data-filter]'));
const searchInput = document.querySelector('[data-project-search]');
const resultCount = document.querySelector('[data-result-count]');
const cards = Array.from(document.querySelectorAll('[data-project-card]'));
const detailButtons = Array.from(document.querySelectorAll('[data-detail-toggle]'));

function updateCards() {
  const activeButton = document.querySelector('[data-filter].is-active');
  const activeFilter = activeButton ? activeButton.dataset.filter : 'all';
  const searchTerm = (searchInput?.value || '').trim().toLowerCase();
  let visibleCount = 0;

  cards.forEach((card) => {
    const category = card.dataset.category || '';
    const keywords = card.dataset.keywords || '';
    const matchesFilter = activeFilter === 'all' || category === activeFilter;
    const matchesSearch = !searchTerm || keywords.includes(searchTerm);
    const isVisible = matchesFilter && matchesSearch;

    card.hidden = !isVisible;
    if (isVisible) {
      visibleCount += 1;
    }
  });

  if (resultCount) {
    resultCount.textContent = `${visibleCount} project${visibleCount === 1 ? '' : 's'} shown`;
  }
}

filterButtons.forEach((button) => {
  button.addEventListener('click', () => {
    filterButtons.forEach((item) => item.classList.remove('is-active'));
    button.classList.add('is-active');
    updateCards();
  });
});

searchInput?.addEventListener('input', updateCards);

detailButtons.forEach((button) => {
  button.addEventListener('click', () => {
    const targetId = button.getAttribute('aria-controls');
    const panel = targetId ? document.getElementById(targetId) : null;
    if (!panel) {
      return;
    }

    const expanded = button.getAttribute('aria-expanded') === 'true';
    button.setAttribute('aria-expanded', String(!expanded));
    panel.hidden = expanded;
    button.textContent = expanded ? 'Quick facts' : 'Hide facts';
  });
});

updateCards();
