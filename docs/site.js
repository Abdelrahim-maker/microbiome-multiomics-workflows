const filterButtons = Array.from(document.querySelectorAll('[data-filter]'));
const searchInput = document.querySelector('[data-project-search]');
const resultCount = document.querySelector('[data-result-count]');
const cards = Array.from(document.querySelectorAll('[data-project-card]'));
const detailButtons = Array.from(document.querySelectorAll('[data-detail-toggle]'));
const featuredButtons = Array.from(document.querySelectorAll('[data-featured-key]'));
const revealItems = Array.from(document.querySelectorAll('.reveal'));

const featuredProjects = {
  shrub: {
    title: 'Ordination and Shrub-System Interpretation',
    summary: 'A treatment-focused ordination workflow showing how shrub-state context changes carbon-linked community patterns.',
    focus: 'Ecological separation',
    methods: 'Hellinger PCA + contrasts',
    value: 'Fast visual insight',
    detail: 'This project is a strong first stop for recruiters because it combines statistical structure, visual storytelling, and a biologically intuitive treatment narrative.',
    link: 'projects/ordination-visualization.html',
    image: 'assets/previews/GC_OM_vs_noOM_separate_shrub_groups.png',
    alt: 'Shrub-system ordination preview'
  },
  carbon: {
    title: 'Carbon Sequestration Storyline',
    summary: 'A manuscript-ready carbon response view connecting category-level shifts, gene-level follow-up, and interpretation-ready graphics.',
    focus: 'Carbon-function response',
    methods: 'Contrasts + figure synthesis',
    value: 'Narrative clarity',
    detail: 'This view highlights how technical outputs can be reframed into a coherent carbon story for reviewers who care about environmental relevance and communication quality.',
    link: 'projects/manuscript-story.html',
    image: 'assets/previews/GC_carbon_category_with_stats.png',
    alt: 'Carbon response figure preview'
  },
  delivery: {
    title: 'Reporting and Delivery Pipeline',
    summary: 'A reproducible reporting layer that turns statistical outputs into auditable tables, shareable summaries, and export-ready deliverables.',
    focus: 'Handoff readiness',
    methods: 'Automated reporting + exports',
    value: 'Operational polish',
    detail: 'This project shows the part many portfolios omit: how analysis becomes something a collaborator, supervisor, or hiring team can actually review and reuse.',
    link: 'projects/reporting-deliverables.html',
    image: 'assets/previews/GC_cross_level_effect_sizes.png',
    alt: 'Reporting summary preview'
  }
};

function setFeaturedProject(key) {
  const project = featuredProjects[key];
  if (!project) {
    return;
  }

  const title = document.querySelector('[data-featured-title]');
  const summary = document.querySelector('[data-featured-summary]');
  const focus = document.querySelector('[data-featured-focus]');
  const methods = document.querySelector('[data-featured-methods]');
  const value = document.querySelector('[data-featured-value]');
  const detail = document.querySelector('[data-featured-detail]');
  const link = document.querySelector('[data-featured-link]');
  const image = document.querySelector('[data-featured-image]');

  if (title) title.textContent = project.title;
  if (summary) summary.textContent = project.summary;
  if (focus) focus.textContent = project.focus;
  if (methods) methods.textContent = project.methods;
  if (value) value.textContent = project.value;
  if (detail) detail.textContent = project.detail;
  if (link) link.href = project.link;
  if (image) {
    image.src = project.image;
    image.alt = project.alt;
  }
}

function setupReveals() {
  if (!('IntersectionObserver' in window)) {
    revealItems.forEach((item) => item.classList.add('is-visible'));
    return;
  }

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.12 });

  revealItems.forEach((item) => observer.observe(item));
}

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

featuredButtons.forEach((button) => {
  button.addEventListener('click', () => {
    featuredButtons.forEach((item) => item.classList.remove('is-active'));
    button.classList.add('is-active');
    setFeaturedProject(button.dataset.featuredKey);
  });
});

updateCards();
setFeaturedProject('shrub');
setupReveals();
