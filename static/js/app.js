/**
 * app.js — CarMūlya frontend controller
 * Handles: metadata → UI population, form interaction, API call, result rendering
 */

"use strict";

// ─────────────────────────────────────────────
// 0. BOOT
// ─────────────────────────────────────────────
const meta = JSON.parse(document.getElementById("meta-json").textContent);

// ─────────────────────────────────────────────
// 1. HELPERS
// ─────────────────────────────────────────────

/** Format a number into Indian numbering system (₹4,25,000) */
function formatINR(amount) {
  const n = Math.round(amount);
  const s = n.toString();
  if (s.length <= 3) return "₹" + s;
  const last3 = s.slice(-3);
  const rest   = s.slice(0, -3);
  return "₹" + rest.replace(/\B(?=(\d{2})+(?!\d))/g, ",") + "," + last3;
}

/** Populate a <select> with options */
function populateSelect(selectEl, items, placeholder) {
  selectEl.innerHTML = `<option value="">${placeholder}</option>`;
  items.forEach(item => {
    const opt = document.createElement("option");
    opt.value = item;
    opt.textContent = item;
    selectEl.appendChild(opt);
  });
}

/** Show / hide an element */
function toggle(el, show) {
  el.hidden = !show;
  if (show) el.classList.add("fade-in");
}

// ─────────────────────────────────────────────
// 2. DOM REFS
// ─────────────────────────────────────────────
const brandSel    = document.getElementById("brand");
const modelSel    = document.getElementById("model");
const citySel     = document.getElementById("city");
const yearSlider  = document.getElementById("year");
const yearDisplay = document.getElementById("year-display");
const kmSlider    = document.getElementById("km_driven");
const kmDisplay   = document.getElementById("km-display");
const ownerInput  = document.getElementById("owners");
const fuelInput   = document.getElementById("fuel_type");
const transInput  = document.getElementById("transmission");
const predictBtn  = document.getElementById("predictBtn");
const resultCard  = document.getElementById("resultCard");
const errorCard   = document.getElementById("errorCard");
const errorMsg    = document.getElementById("errorMsg");

// ─────────────────────────────────────────────
// 3. POPULATE DROPDOWNS FROM METADATA
// ─────────────────────────────────────────────
populateSelect(brandSel, Object.keys(meta.car_models), "Select brand…");
populateSelect(citySel,  meta.cities,                  "Select city…");

// When brand changes → populate model dropdown
brandSel.addEventListener("change", () => {
  const brand  = brandSel.value;
  const models = brand ? Object.keys(meta.car_models[brand]) : [];
  populateSelect(modelSel, models, "Select model…");
  modelSel.disabled = models.length === 0;
  clearError("brand");
});

modelSel.addEventListener("change", () => clearError("model"));
citySel.addEventListener("change",  () => clearError("city"));

// ─────────────────────────────────────────────
// 4. SLIDER LIVE UPDATES
// ─────────────────────────────────────────────
yearSlider.addEventListener("input", () => {
  yearDisplay.textContent = yearSlider.value;
});

kmSlider.addEventListener("input", () => {
  const km = parseInt(kmSlider.value, 10);
  kmDisplay.textContent = km.toLocaleString("en-IN") + " km";
});

// ─────────────────────────────────────────────
// 5. PILL BUTTONS (Fuel & Transmission)
// ─────────────────────────────────────────────
function setupPillGroup(groupId, hiddenInputId, errorId) {
  const group = document.getElementById(groupId);
  const input = document.getElementById(hiddenInputId);
  group.querySelectorAll(".pill").forEach(pill => {
    pill.addEventListener("click", () => {
      group.querySelectorAll(".pill").forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      input.value = pill.dataset.value;
      clearError(errorId);
    });
  });
}

setupPillGroup("fuel-pills",  "fuel_type",    "fuel_type");
setupPillGroup("trans-pills", "transmission", "transmission");

// ─────────────────────────────────────────────
// 6. OWNER DOT SELECTOR
// ─────────────────────────────────────────────
const ownerDots = document.querySelectorAll(".dot");
ownerDots.forEach(dot => {
  dot.addEventListener("click", () => {
    ownerDots.forEach(d => d.classList.remove("active"));
    dot.classList.add("active");
    ownerInput.value = dot.dataset.val;
    const n = parseInt(dot.dataset.val, 10);
    document.getElementById("owners-display").textContent =
      n === 1 ? "1 owner" : `${n} owners`;
  });
});

// ─────────────────────────────────────────────
// 7. VALIDATION
// ─────────────────────────────────────────────
function clearError(field) {
  const el = document.getElementById(`err-${field}`);
  if (el) el.textContent = "";
}

function showError(field, msg) {
  const el = document.getElementById(`err-${field}`);
  if (el) el.textContent = msg;
}

function validateForm() {
  let valid = true;
  if (!brandSel.value)   { showError("brand",        "Select a brand");        valid = false; }
  if (!modelSel.value)   { showError("model",        "Select a model");        valid = false; }
  if (!citySel.value)    { showError("city",         "Select a city");         valid = false; }
  if (!fuelInput.value)  { showError("fuel_type",    "Choose a fuel type");    valid = false; }
  if (!transInput.value) { showError("transmission", "Choose transmission");   valid = false; }
  return valid;
}

// ─────────────────────────────────────────────
// 8. PREDICT
// ─────────────────────────────────────────────
predictBtn.addEventListener("click", async () => {
  // Hide stale results
  toggle(resultCard, false);
  toggle(errorCard,  false);

  if (!validateForm()) return;

  // Loading state
  const btnText   = predictBtn.querySelector(".btn-text");
  const btnLoader = predictBtn.querySelector(".btn-loader");
  btnText.hidden   = true;
  btnLoader.hidden = false;
  predictBtn.disabled = true;

  const payload = {
    brand:        brandSel.value,
    model:        modelSel.value,
    year:         parseInt(yearSlider.value, 10),
    fuel_type:    fuelInput.value,
    transmission: transInput.value,
    km_driven:    parseInt(kmSlider.value, 10),
    owners:       parseInt(ownerInput.value, 10),
    city:         citySel.value,
  };

  try {
    const resp = await fetch("/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });
    const data = await resp.json();

    if (!resp.ok) {
      throw new Error(data.error || `Server error (${resp.status})`);
    }

    renderResult(data, payload);
  } catch (err) {
    errorMsg.textContent = err.message || "Network error — please try again.";
    toggle(errorCard, true);
  } finally {
    btnText.hidden   = false;
    btnLoader.hidden = true;
    predictBtn.disabled = false;
  }
});

// ─────────────────────────────────────────────
// 9. RENDER RESULT
// ─────────────────────────────────────────────
function renderResult(data, payload) {
  // Price
  document.getElementById("resultPrice").textContent = formatINR(data.price_inr);
  document.getElementById("resultRange").textContent =
    `Range: ${formatINR(data.range_low_inr)} – ${formatINR(data.range_high_inr)}`;
  document.getElementById("resultCarName").textContent =
    `${payload.year} ${payload.brand} ${payload.model} · ${payload.fuel_type} · ${payload.transmission}`;

  // Gauge needle (score 0–100 mapped to 0–100% width)
  const needle = document.getElementById("gaugeNeedle");
  // score 100 = far left (Great Deal), score 0 = far right (Overpriced)
  const pct = (100 - data.deal_score); // invert so left = great
  needle.style.left = `${Math.max(2, Math.min(98, pct))}%`;

  // Deal badge
  const badge  = document.getElementById("dealBadge");
  badge.textContent = data.deal_rating;
  badge.className   = "deal-badge";
  if (data.deal_rating === "Great Deal") badge.classList.add("badge-great");
  else if (data.deal_rating === "Fair Deal") badge.classList.add("badge-fair");
  else badge.classList.add("badge-over");

  // Breakdown pills
  const bd = document.getElementById("breakdown");
  bd.innerHTML = "";
  const pills = [
    ["Brand",        payload.brand],
    ["Model",        payload.model],
    ["Year",         payload.year],
    ["Fuel",         payload.fuel_type],
    ["Transmission", payload.transmission],
    ["KM Driven",    parseInt(payload.km_driven).toLocaleString("en-IN") + " km"],
    ["Owners",       payload.owners],
    ["City",         payload.city],
    ["Deal Score",   `${data.deal_score}/100`],
  ];
  pills.forEach(([lbl, val]) => {
    const div = document.createElement("div");
    div.className = "breakdown-pill";
    div.innerHTML = `${lbl}<strong>${val}</strong>`;
    bd.appendChild(div);
  });

  toggle(resultCard, true);

  // Smooth scroll to results
  resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
}
