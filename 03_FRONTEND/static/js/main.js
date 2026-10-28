// main.js — sidebar toggle + theme toggle (light is default; toggles to dark)

// Applied immediately (before DOMContentLoaded) so there's no flash of the
// wrong theme on navigation between pages.
(function () {
  const saved = localStorage.getItem("pace-theme");
  if (saved === "dark" || saved === "light") {
    document.documentElement.setAttribute("data-theme", saved);
  }
})();

document.addEventListener("DOMContentLoaded", function () {
  const sidebar = document.getElementById("sidebar");
  const sidebarToggle = document.getElementById("sidebarToggle");
  if (sidebarToggle) {
    sidebarToggle.addEventListener("click", () => sidebar.classList.toggle("open"));
  }

  // Desktop sidebar collapse (persisted across pages/sessions)
  const collapseToggle = document.getElementById("collapseToggle");
  if (collapseToggle && sidebar) {
    if (localStorage.getItem("rp-sidebar-collapsed") === "1") {
      sidebar.classList.add("collapsed");
    }
    collapseToggle.addEventListener("click", () => {
      const collapsed = sidebar.classList.toggle("collapsed");
      localStorage.setItem("rp-sidebar-collapsed", collapsed ? "1" : "0");
    });
  }

  const themeToggle = document.getElementById("themeToggle");
  const html = document.documentElement;
  function syncThemeIcon() {
    if (!themeToggle) return;
    const light = html.getAttribute("data-theme") === "light";
    themeToggle.innerHTML = light ? '<i class="fa-solid fa-sun"></i>' : '<i class="fa-solid fa-moon"></i>';
    themeToggle.title = light ? "Toggle dark mode" : "Toggle light mode";
  }
  syncThemeIcon();

  // Count-up animation for KPI numbers (.kpi-value / .kpi-number). Parses
  // the existing server-rendered text so no values are invented — only
  // how they appear replays from 0, keeping prefix/suffix (₹, %, commas).
  document.querySelectorAll(".kpi-value, .kpi-number").forEach(function (el) {
    const raw = el.textContent.trim();
    const match = raw.match(/^([^\d\-]*)([\d,]+\.?\d*)(.*)$/);
    if (!match) return;
    const [, prefix, numStr, suffix] = match;
    const target = parseFloat(numStr.replace(/,/g, ""));
    if (isNaN(target)) return;
    const decimals = (numStr.split(".")[1] || "").length;
    const duration = 900;
    const start = performance.now();
    function tick(now) {
      const p = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      const val = target * eased;
      el.textContent = prefix + val.toLocaleString(undefined, {
        minimumFractionDigits: decimals, maximumFractionDigits: decimals,
      }) + suffix;
      if (p < 1) requestAnimationFrame(tick);
      else el.textContent = raw;
    }
    requestAnimationFrame(tick);
  });

  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      const nowLight = html.getAttribute("data-theme") === "light";
      const next = nowLight ? "dark" : "light";
      html.setAttribute("data-theme", next);
      localStorage.setItem("pace-theme", next);
      syncThemeIcon();
      // Chart.js already rendered with the old palette — refresh the ones
      // that read Chart.defaults.color at construction time.
      if (window.Chart) {
        Chart.defaults.color = next === "light" ? "#565F78" : "#A6AEC4";
        Chart.defaults.borderColor = next === "light" ? "rgba(227,230,240,0.8)" : "rgba(38,44,61,0.8)";
        Object.values(Chart.instances || {}).forEach(c => c.update());
      }
    });
  }
});

// Shared Chart.js palette matching the Aurora Intelligence design tokens
const CHART_COLORS = ["#8B6BFF", "#2DD4E8", "#34D399", "#FBBF24", "#FB7185", "#6D4FE0", "#A78BFA"];
window.CHART_COLORS = CHART_COLORS;

function isLightTheme() { return document.documentElement.getAttribute("data-theme") === "light"; }

if (typeof Chart !== "undefined") {
  Chart.defaults.font.family = "'Inter', sans-serif";
  Chart.defaults.color = isLightTheme() ? "#565F78" : "#A6AEC4";
  Chart.defaults.borderColor = isLightTheme() ? "rgba(227,230,240,0.8)" : "rgba(38,44,61,0.8)";
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.boxWidth = 8;
  Chart.defaults.plugins.legend.labels.padding = 14;
  Chart.defaults.elements.line.borderWidth = 2.5;
  Chart.defaults.elements.bar.borderRadius = 6;
  Chart.defaults.elements.point.radius = 3;
  Chart.defaults.elements.point.hoverRadius = 5;
}
