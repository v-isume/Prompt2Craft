function copyBibTeX() {
  var bibTexElement = document.querySelector(".bibtex-section pre code");
  if (bibTexElement) {
    var bibTexText = bibTexElement.innerText;
    navigator.clipboard.writeText(bibTexText);
    alert("BibTeX citation copied to clipboard!");
  }
}

function toggleDarkMode() {
  document.body.classList.toggle("dark-mode");
  const nav = document.querySelector(".nav");
  if (nav) nav.classList.toggle("dark-mode");
}

window.onscroll = function () {
  const scrollUpBtn = document.getElementById("scrollUpBtn");
  if (scrollUpBtn) {
    if (
      document.body.scrollTop > 100 ||
      document.documentElement.scrollTop > 100
    ) {
      scrollUpBtn.style.display = "block";
    } else {
      scrollUpBtn.style.display = "none";
    }
  }
};

function scrollToTop() {
  window.scrollTo({ top: 0, behavior: "smooth" });
}

class Carousel {
  constructor(element, interval = 3000) {
    if (!element) return;
    this.container = element;
    this.track = element.querySelector(".carousel-track");
    this.slides = Array.from(element.querySelectorAll(".carousel-slide"));
    this.indicators = element.querySelector(".carousel-indicators");

    if (!this.track || this.slides.length === 0) return;

    this.currentIndex = 0;
    this.slidesPerView = parseInt(element.dataset.slidesPerView, 10) || 3;
    this.totalSlides = Math.ceil(this.slides.length / this.slidesPerView);
    this.interval = interval;
    this.autoPlayTimer = null;

    if (this.indicators) {
      this.createIndicators();
    }
    this.setupEventListeners();
    this.startAutoPlay();
    this.updateCarousel();
  }

  createIndicators() {
    this.indicators.innerHTML = "";
    for (let i = 0; i < this.totalSlides; i++) {
      const button = document.createElement("button");
      button.classList.add("indicator");
      if (i === 0) button.classList.add("active");
      button.addEventListener("click", () => {
        this.goToSlide(i);
      });
      this.indicators.appendChild(button);
    }
  }

  setupEventListeners() {
    const prevBtn = this.container.querySelector(".prev");
    const nextBtn = this.container.querySelector(".next");

    if (prevBtn) {
      prevBtn.addEventListener("click", (e) => {
        e.preventDefault();
        this.prevSlide();
      });
    }

    if (nextBtn) {
      nextBtn.addEventListener("click", (e) => {
        e.preventDefault();
        this.nextSlide();
      });
    }

    this.container.addEventListener("mouseenter", () => {
      this.stopAutoPlay();
    });

    this.container.addEventListener("mouseleave", () => {
      this.startAutoPlay();
    });

    document.addEventListener("keydown", (e) => {
      if (this.container.matches(":hover")) {
        if (e.key === "ArrowLeft") {
          this.prevSlide();
        } else if (e.key === "ArrowRight") {
          this.nextSlide();
        }
      }
    });

    // Touch support
    let touchStartX = 0;
    let touchEndX = 0;

    this.container.addEventListener(
      "touchstart",
      (e) => {
        touchStartX = e.changedTouches[0].screenX;
      },
      { passive: true }
    );

    this.container.addEventListener(
      "touchend",
      (e) => {
        touchEndX = e.changedTouches[0].screenX;
        const swipeThreshold = 50;
        const diff = touchStartX - touchEndX;
        if (Math.abs(diff) > swipeThreshold) {
          if (diff > 0) {
            this.nextSlide();
          } else {
            this.prevSlide();
          }
        }
      },
      { passive: true }
    );
  }

  updateCarousel() {
    const offset =
      -this.currentIndex *
      (100 / this.slidesPerView) *
      this.slidesPerView;
    this.track.style.transform = `translateX(${offset}%)`;

    if (this.indicators) {
      const indicators = Array.from(this.indicators.children);
      indicators.forEach((indicator, index) => {
        indicator.classList.toggle("active", index === this.currentIndex);
      });
    }
  }

  nextSlide() {
    this.currentIndex = (this.currentIndex + 1) % this.totalSlides;
    this.updateCarousel();
    this.resetAutoPlay();
  }

  prevSlide() {
    this.currentIndex =
      (this.currentIndex - 1 + this.totalSlides) % this.totalSlides;
    this.updateCarousel();
    this.resetAutoPlay();
  }

  goToSlide(index) {
    if (index !== this.currentIndex) {
      this.currentIndex = index;
      this.updateCarousel();
      this.resetAutoPlay();
    }
  }

  startAutoPlay() {
    if (this.autoPlayTimer) {
      clearInterval(this.autoPlayTimer);
    }
    this.autoPlayTimer = setInterval(() => {
      this.nextSlide();
    }, this.interval);
  }

  stopAutoPlay() {
    if (this.autoPlayTimer) {
      clearInterval(this.autoPlayTimer);
      this.autoPlayTimer = null;
    }
  }

  resetAutoPlay() {
    this.stopAutoPlay();
    this.startAutoPlay();
  }
}

// YAML Dynamic Loader Logic
async function loadYAMLConfig() {
  const yamlApp = document.getElementById("yaml-app");
  if (!yamlApp) return; // If manually using HTML edit fallback method

  try {
    const response = await fetch("./data.yaml");
    if (!response.ok) {
      throw new Error(`Failed to load data.yaml: ${response.statusText}`);
    }
    const yamlText = await response.text();
    const data = jsyaml.load(yamlText);

    if (!data) return;

    window.appSettings = data.settings || {};

    renderNav(data.sections);
    renderHeader(data);
    renderButtons(data.buttons, data.show_bibtex_button, data.bibtex_button_icon, data.bibtex_button_color);
    renderSections(data.sections, data.settings);
    renderFooter(data.footer);
    applySettings(data.settings);

    initCarousels();
  } catch (error) {
    console.error("Error loading YAML config:", error);
  }
}

function applySettings(settings) {
  if (!settings) return;

  // Custom Max Width for Page Content Container and Nav Content
  const maxWidth = (settings.max_width && settings.max_width.trim()) ? settings.max_width.trim() : "1024px";
  document.querySelectorAll(".container, .nav-content").forEach((elem) => {
    elem.style.maxWidth = maxWidth;
  });

  // Top Nav Shadow
  const navElem = document.getElementById("nav");
  if (navElem) {
    if (settings.nav_shadow === true) {
      navElem.style.boxShadow = "0 2px 10px rgba(0, 0, 0, 0.08)";
    } else {
      navElem.style.boxShadow = "none";
    }
  }

  // Custom Nav Highlight Color
  if (settings.nav_highlight_color && settings.nav_highlight_color.trim()) {
    document.documentElement.style.setProperty("--nav-highlight-color", settings.nav_highlight_color.trim());
  } else {
    document.documentElement.style.removeProperty("--nav-highlight-color");
  }

  // Dark / Light mode toggle button
  const toggleBtn = document.querySelector(".toggle-btn");
  if (toggleBtn) {
    if (settings.theme_toggle_text) {
      toggleBtn.innerText = settings.theme_toggle_text;
    }
    if (settings.theme_toggle_color) {
      toggleBtn.style.backgroundColor = settings.theme_toggle_color;
    }
  }

  // Scroll Up button
  const scrollUpBtn = document.getElementById("scrollUpBtn");
  if (scrollUpBtn) {
    if (settings.scroll_top_text) {
      scrollUpBtn.innerText = settings.scroll_top_text;
    }
    if (settings.scroll_top_color) {
      scrollUpBtn.style.backgroundColor = settings.scroll_top_color;
    }
  }

  // Scroll Highlight for Nav Links
  if (settings.nav_scroll_highlight === true) {
    initNavScrollHighlight();
  }
}

function initNavScrollHighlight() {
  const sections = document.querySelectorAll(".abstract, .content-section, .bibtex-section");
  const navLinks = document.querySelectorAll("#nav-links a");

  if (!sections.length || !navLinks.length) return;

  function highlightOnScroll() {
    let currentSectionId = "";
    const scrollPosition = window.scrollY + 120;

    sections.forEach((section) => {
      const sectionTop = section.offsetTop;
      const sectionHeight = section.offsetHeight;
      if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
        currentSectionId = section.getAttribute("id");
      }
    });

    if (window.scrollY < 150 && sections.length > 0) {
      currentSectionId = sections[0].getAttribute("id");
    }

    if ((window.innerHeight + window.scrollY) >= document.body.offsetHeight - 50) {
      if (sections.length > 0) {
        currentSectionId = sections[sections.length - 1].getAttribute("id");
      }
    }

    navLinks.forEach((link) => {
      const href = link.getAttribute("href");
      if (href === `#${currentSectionId}`) {
        link.classList.add("active");
      } else {
        link.classList.remove("active");
      }
    });
  }

  window.addEventListener("scroll", highlightOnScroll);
  highlightOnScroll();
}

function renderNav(sections) {
  const navLinksContainer = document.getElementById("nav-links");
  if (!navLinksContainer || !sections) return;

  navLinksContainer.innerHTML = sections
    .map(
      (sec) =>
        `<a href="#${sec.id}">${sec.nav_label || sec.title || sec.id}</a>`
    )
    .join("");
}

function renderHeader(data) {
  const headerElem = document.getElementById("header");
  if (!headerElem) return;

  if (data.paper_title) {
    document.title = data.paper_title;
  }

  let html = `<h1>${data.paper_title || "Paper Title"}</h1>`;
  html += `<div class="authors">`;

  // Authors
  if (data.authors && data.authors.length > 0) {
    const authorStr = data.authors
      .map((author) => {
        const nameText =
          author.url && author.url !== "#"
            ? `<a href="${author.url}" target="_blank">${author.name}</a>`
            : `<a href="#">${author.name}</a>`;
        const sup = author.superscript ? `<sup>${author.superscript}</sup>` : "";
        return `${nameText}${sup}`;
      })
      .join(", ");
    html += `<p>${authorStr}</p>`;
  }

  // Affiliations
  if (data.affiliations && data.affiliations.length > 0) {
    const affStr = data.affiliations
      .map((aff) => `<sup>${aff.index}</sup>${aff.name}`)
      .join("&nbsp;&nbsp; ");
    html += `<p>${affStr}</p>`;
  }

  // Emails
  if (data.emails && data.emails.length > 0) {
    const emailStr = data.emails
      .map((e) => `<sup>${e.index}</sup><a href="mailto:${e.email}">${e.email}</a>`)
      .join(", ");
    html += `<p>${emailStr}</p>`;
  }

  // Equal contribution
  if (data.equal_contribution_note) {
    html += `<p><em>${data.equal_contribution_note}</em></p>`;
  }

  // Conference badges
  if (data.conference_badges && data.conference_badges.length > 0) {
    data.conference_badges.forEach((badge) => {
      html += `<p>${badge}</p>`;
    });
  }

  // Extra Header Note
  if (data.header_note) {
    html += `<p class="header-note">${data.header_note}</p>`;
  }

  html += `</div>`;
  headerElem.innerHTML = html;
}

function renderButtons(buttons, showBibTex, bibtexIcon, bibtexColor) {
  const buttonsElem = document.getElementById("buttons");
  if (!buttonsElem) return;

  let html = "";

  if (buttons && buttons.length > 0) {
    html += buttons
      .map((btn) => {
        const iconStr = btn.icon ? `${btn.icon} ` : "";
        const bgStyle = btn.color ? ` style="background-color: ${btn.color};"` : "";
        return `<a href="${btn.url || "#"}" target="_blank"${bgStyle}>${iconStr}${btn.label}</a>`;
      })
      .join("");
  }

  if (showBibTex !== false) {
    const iconStr = bibtexIcon ? `${bibtexIcon} ` : "📜 ";
    const bgStyle = bibtexColor ? ` style="background-color: ${bibtexColor};"` : "";
    html += `<a href="#bibtex" target="_self"${bgStyle}>${iconStr}BibTeX</a>`;
  }

  buttonsElem.innerHTML = html;
}

function renderListHTML(listItems, listType) {
  if (!listItems || listItems.length === 0) return "";

  let tag = "ul";
  let styleAttr = "";

  const typeLower = (listType || "bullet").toLowerCase();

  if (typeLower === "number" || typeLower === "decimal" || typeLower === "numbered") {
    tag = "ol";
  } else if (typeLower === "check" || typeLower === "checkmark") {
    tag = "ul";
    styleAttr = ' style="list-style-type: \'✅ \';"';
  } else if (typeLower === "arrow") {
    tag = "ul";
    styleAttr = ' style="list-style-type: \'➡️ \';"';
  } else if (typeLower === "star") {
    tag = "ul";
    styleAttr = ' style="list-style-type: \'⭐ \';"';
  } else if (typeLower === "circle") {
    tag = "ul";
    styleAttr = ' style="list-style-type: circle;"';
  } else if (typeLower === "square") {
    tag = "ul";
    styleAttr = ' style="list-style-type: square;"';
  } else if (typeLower === "none") {
    tag = "ul";
    styleAttr = ' style="list-style-type: none; padding-left: 0;"';
  } else {
    tag = "ul";
  }

  let html = `<${tag}${styleAttr}>`;
  listItems.forEach((item) => {
    html += `<li>${item}</li>`;
  });
  html += `</${tag}>`;
  return html;
}

function renderSections(sections, settings) {
  const container = document.getElementById("sections-container");
  if (!container || !sections) return;

  let html = "";
  const copyBtnText = (settings && settings.copy_bibtex_text) || "Copy to Clipboard";
  const copyBtnIcon = (settings && typeof settings.copy_bibtex_icon !== "undefined") ? settings.copy_bibtex_icon : "📋";
  const copyBtnLabel = copyBtnIcon ? `${copyBtnIcon} ${copyBtnText}` : copyBtnText;
  const copyBtnStyle = (settings && settings.copy_bibtex_color)
    ? ` style="background-color: ${settings.copy_bibtex_color};"`
    : "";

  sections.forEach((sec) => {
    if (sec.id === "bibtex") {
      html += `
        <div class="bibtex-section" id="${sec.id}">
          <h2>${sec.title || "BibTeX"}</h2>
          <button class="bibtex-copy-button"${copyBtnStyle} onclick="copyBibTeX()">${copyBtnLabel}</button>
          <pre><code class="language-bibtex">${escapeHTML(sec.citation || "")}</code></pre>
        </div>
      `;
      return;
    }

    const sectionClass = sec.id === "abstract" ? "abstract" : "content-section";
    html += `<div class="${sectionClass}" id="${sec.id}">`;
    html += `<h2>${sec.title}</h2>`;

    if (sec.content) {
      html += `<p>${sec.content}</p>`;
    }

    if (sec.list && sec.list.length > 0) {
      html += renderListHTML(sec.list, sec.list_type);
    }

    // Render modular components inside section
    if (sec.components && sec.components.length > 0) {
      sec.components.forEach((comp) => {
        html += renderComponent(comp);
      });
    }

    // Render subsections inside section
    if (sec.subsections && sec.subsections.length > 0) {
      sec.subsections.forEach((subsec, subIdx) => {
        const subId = subsec.id || `${sec.id}-sub-${subIdx + 1}`;
        html += `<div class="subsection" id="${subId}">`;
        if (subsec.title) {
          html += `<h3>${subsec.title}</h3>`;
        }
        if (subsec.content) {
          html += `<p>${subsec.content}</p>`;
        }
        if (subsec.list && subsec.list.length > 0) {
          html += renderListHTML(subsec.list, subsec.list_type);
        }
        if (subsec.components && subsec.components.length > 0) {
          subsec.components.forEach((comp) => {
            html += renderComponent(comp);
          });
        }
        html += `</div>`;
      });
    }

    html += `</div>`;
  });

  container.innerHTML = html;
}

function renderComponent(comp) {
  if (!comp || !comp.type) return "";

  if (comp.type === "list" || comp.type === "text_block") {
    let compHtml = '<div class="content-block">';
    if (comp.title) compHtml += `<h3>${comp.title}</h3>`;
    if (comp.content) compHtml += `<p>${comp.content}</p>`;
    if (comp.list && comp.list.length > 0) {
      compHtml += renderListHTML(comp.list, comp.list_type);
    }
    compHtml += '</div>';
    return compHtml;
  }

  if (comp.type === "custom_html" || comp.type === "html") {
    let compHtml = '<div class="custom-html-block">';
    if (comp.title) compHtml += `<h3>${comp.title}</h3>`;
    if (comp.html) compHtml += comp.html;
    compHtml += '</div>';
    return compHtml;
  }

  if (comp.type === "youtube_video") {
    return `
      <div class="video-section">
        ${comp.title ? `<h3>${comp.title}</h3>` : ""}
        <div class="video-container">
          <iframe width="560" height="315" src="${comp.url}" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
        </div>
        ${comp.description ? `<p>${comp.description}</p>` : ""}
      </div>
    `;
  }

  if (comp.type === "youtube_playlist") {
    return `
      <div class="video-section">
        ${comp.title ? `<h3>${comp.title}</h3>` : ""}
        <div class="video-container">
          <iframe width="560" height="315" src="${comp.url}" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
        </div>
      </div>
    `;
  }

  if (comp.type === "image_carousel" && comp.images) {
    const slides = comp.images
      .map(
        (img, idx) =>
          `<div class="carousel-slide"><img src="${img}" alt="Slide ${idx + 1}" /></div>`
      )
      .join("");
    // Optional per-carousel layout: slides_per_view (default 3) and slide height (default 300px)
    const spv = parseInt(comp.slides_per_view, 10) || 3;
    const height = comp.height ? String(comp.height).trim() : "300px";
    return `
      ${comp.title ? `<h3>${comp.title}</h3>` : ""}
      <div class="carousel-container image-carousel" data-slides-per-view="${spv}" style="--slides-per-view: ${spv}; --slide-height: ${height};">
        <div class="carousel-track">${slides}</div>
        <button class="carousel-button prev">←</button>
        <button class="carousel-button next">→</button>
        <div class="carousel-indicators"></div>
      </div>
    `;
  }

  if (comp.type === "video_carousel" && comp.videos) {
    const slides = comp.videos
      .map(
        (video) =>
          `<div class="carousel-slide"><iframe width="560" height="315" src="${video}" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe></div>`
      )
      .join("");
    return `
      ${comp.title ? `<h3>${comp.title}</h3>` : ""}
      <div class="carousel-container video-carousel">
        <div class="carousel-track">${slides}</div>
        <button class="carousel-button prev">←</button>
        <button class="carousel-button next">→</button>
        <div class="carousel-indicators"></div>
      </div>
    `;
  }

  return "";
}

function renderFooter(footerData) {
  const footerElem = document.getElementById("footer");
  if (!footerElem) return;

  let html = "";
  if (footerData && footerData.copyright) {
    html += `<p>${footerData.copyright}</p>`;
  }
  if (footerData && footerData.footer_note) {
    html += `<p class="footer-note">${footerData.footer_note}</p>`;
  }
  html += `<p>Website template free to borrow from <a href="https://github.com/indramal/iNdra-GitHub-Page-Template-For-Resarch" target="_blank">here</a>.</p>`;

  footerElem.innerHTML = html;
}

function initCarousels() {
  const carouselElems = document.querySelectorAll(".carousel-container");
  carouselElems.forEach((elem) => {
    new Carousel(elem);
  });
}

function escapeHTML(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

document.addEventListener("DOMContentLoaded", () => {
  // Reads USE_YAML_DATA variable set in index.html (defaults to true if not defined)
  const isYamlMode = typeof USE_YAML_DATA !== "undefined" ? USE_YAML_DATA : true;

  if (isYamlMode) {
    loadYAMLConfig();
  } else {
    // Hide YAML skeleton containers if using manual HTML edit method
    const yamlApp = document.getElementById("yaml-app");
    if (yamlApp) {
      yamlApp.style.display = "none";
    }
    const dynamicFooter = document.getElementById("footer");
    if (dynamicFooter) {
      dynamicFooter.style.display = "none";
    }
    // Initialize carousels for manual static HTML content
    initCarousels();
  }
});