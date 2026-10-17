/* Lab Scan AI: тема, drag-and-drop, масштаб скана, горячие клавиши. Без зависимостей. */
(function () {
  "use strict";

  // ---- тема: светлая / тёмная, запоминается в localStorage ----
  var root = document.documentElement;
  var toggle = document.getElementById("theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var dark = root.dataset.theme === "dark" ||
        (!root.dataset.theme && window.matchMedia("(prefers-color-scheme: dark)").matches);
      root.dataset.theme = dark ? "light" : "dark";
      try { localStorage.setItem("labscan-theme", root.dataset.theme); } catch (e) { /* приватный режим */ }
    });
  }

  // ---- загрузка: перетаскивание файла и отправка без лишнего клика ----
  var dz = document.getElementById("dropzone");
  if (dz) {
    var input = dz.querySelector("input[type=file]");
    var submit = function () { dz.classList.add("busy"); dz.requestSubmit ? dz.requestSubmit() : dz.submit(); };
    ["dragenter", "dragover"].forEach(function (ev) {
      dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.add("over"); });
    });
    ["dragleave", "drop"].forEach(function (ev) {
      dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.remove("over"); });
    });
    dz.addEventListener("drop", function (e) {
      if (e.dataTransfer && e.dataTransfer.files.length) { input.files = e.dataTransfer.files; submit(); }
    });
    input.addEventListener("change", function () { if (input.files.length) submit(); });
  }
  document.querySelectorAll("form.sample").forEach(function (f) {
    f.addEventListener("submit", function () { f.classList.add("busy"); });
  });

  // ---- проверка: масштаб скана ----
  var stage = document.getElementById("scan-stage");
  var zoomVal = document.getElementById("zoom-val");
  var zoom = 1;
  function setZoom(z) {
    zoom = Math.min(4, Math.max(0.5, Math.round(z * 20) / 20));
    stage.style.setProperty("--zoom", zoom);
    if (zoomVal) zoomVal.textContent = Math.round(zoom * 100) + "%";
  }
  if (stage) {
    document.querySelectorAll("[data-zoom]").forEach(function (b) {
      b.addEventListener("click", function () {
        var k = b.dataset.zoom;
        setZoom(k === "+" ? zoom + 0.25 : k === "-" ? zoom - 0.25 : 1);
      });
    });
    stage.addEventListener("wheel", function (e) {
      if (!e.ctrlKey) return;
      e.preventDefault();
      setZoom(zoom + (e.deltaY < 0 ? 0.1 : -0.1));
    }, { passive: false });
    stage.addEventListener("dblclick", function () { setZoom(zoom === 1 ? 2 : 1); });
  }

  // ---- проверка: горячие клавиши ----
  var form = document.getElementById("review-form");
  if (form) {
    document.addEventListener("keydown", function (e) {
      var tag = (e.target.tagName || "").toLowerCase();
      var typing = tag === "input" || tag === "select" || tag === "textarea";
      if (e.key === "Enter" && !e.shiftKey && !e.altKey && (e.ctrlKey || e.metaKey || !typing || tag === "select")) {
        e.preventDefault();
        var btn = form.querySelector("button[type=submit]");
        if (btn) btn.click();
        return;
      }
      if (typing || !stage) return;
      if (e.key === "+" || e.key === "=") { setZoom(zoom + 0.25); e.preventDefault(); }
      else if (e.key === "-" || e.key === "_") { setZoom(zoom - 0.25); e.preventDefault(); }
      else if (e.key === "0") { setZoom(1); e.preventDefault(); }
    });
    // Первое поле, требующее внимания, получает фокус: Tab идёт по строкам сверху вниз.
    var first = form.querySelector("tr.need input.num");
    if (first) first.focus({ preventScroll: true });
  }
})();
