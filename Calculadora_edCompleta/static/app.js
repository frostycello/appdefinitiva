
function fillAndFocus(formSelector, values) {
  const form = document.querySelector(formSelector);
  if (!form) return;
  Object.entries(values).forEach(([name, val]) => {
    const el = form.querySelector(`[name="${name}"]`);
    if (el) el.value = val;
  });
  const first = form.querySelector("input, textarea, select");
  if (first) first.focus();
}

document.addEventListener("click", (e) => {
  const btn = e.target.closest("[data-fill]");
  if (!btn) return;
  e.preventDefault();
  try {
    const payload = JSON.parse(btn.getAttribute("data-fill"));
    fillAndFocus(payload.form, payload.values);
  } catch (_) {}
});
