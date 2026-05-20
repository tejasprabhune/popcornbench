/* PopcornBench gh-pages front-end.
 *
 * Vanilla JS. Loads /runs.json, applies the filter dropdowns, runs
 * the requested sort, and renders into one or more tables. The
 * leaderboard (index.html) calls renderLeaderboard(); the run index
 * (runs.html) calls renderRunsIndex(). The two share filter state
 * so a level chosen on the leaderboard does not silently reset when
 * the user navigates to the run index in the same tab.
 */

const PopcornSite = (() => {
  let DATA = { runs: [], generated_at: null };
  let FILTER_STATE = { level: "", persona: "", scenario: "", model: "" };
  const SORT_STATE = new Map(); // table id -> {key, direction}

  const EM_DASH = "—";

  function fmtNumber(n, digits = 2) {
    if (n === null || n === undefined || !isFinite(n)) return EM_DASH;
    return Number(n).toFixed(digits);
  }
  function fmtPct(n, digits = 0) {
    if (n === null || n === undefined || !isFinite(n)) return EM_DASH;
    return (n * 100).toFixed(digits) + "%";
  }
  function fmtInt(n) {
    if (n === null || n === undefined) return EM_DASH;
    return String(n);
  }
  function fmtStr(s) {
    if (s === null || s === undefined || s === "") return EM_DASH;
    return s;
  }

  async function loadData() {
    if (DATA.runs.length) return DATA;
    const r = await fetch("runs.json", { cache: "no-store" });
    if (!r.ok) throw new Error("runs.json fetch failed: " + r.status);
    DATA = await r.json();
    return DATA;
  }

  function filteredRuns() {
    return DATA.runs.filter(r => {
      if (FILTER_STATE.level && String(r.level) !== FILTER_STATE.level) return false;
      if (FILTER_STATE.persona && r.persona !== FILTER_STATE.persona) return false;
      if (FILTER_STATE.scenario && r.scenario !== FILTER_STATE.scenario) return false;
      if (FILTER_STATE.model && r.model !== FILTER_STATE.model) return false;
      return true;
    });
  }

  function uniqueValues(key) {
    const seen = new Set();
    for (const r of DATA.runs) {
      const v = r[key];
      if (v !== null && v !== undefined && v !== "") seen.add(String(v));
    }
    return Array.from(seen).sort((a, b) => {
      const na = Number(a), nb = Number(b);
      if (!isNaN(na) && !isNaN(nb)) return na - nb;
      return a.localeCompare(b);
    });
  }

  function buildFilters(container, keys) {
    for (const sel of container.querySelectorAll("select[data-filter]")) {
      const key = sel.dataset.filter;
      if (keys && !keys.includes(key)) continue;
      const current = sel.value;
      sel.innerHTML = "";
      const optAll = document.createElement("option");
      optAll.value = "";
      optAll.textContent = "all";
      sel.appendChild(optAll);
      for (const v of uniqueValues(key)) {
        const o = document.createElement("option");
        o.value = v; o.textContent = v;
        sel.appendChild(o);
      }
      sel.value = FILTER_STATE[key] || current || "";
      sel.onchange = (ev) => {
        FILTER_STATE[key] = ev.target.value;
        if (sel.closest("body").contains(document.getElementById("by-model"))) {
          renderLeaderboard();
        } else {
          renderRunsIndex();
        }
      };
    }
  }

  function aggregate(runs, groupKeys) {
    const groups = new Map();
    for (const r of runs) {
      const groupId = groupKeys.map(k => String(r[k] ?? "")).join("|");
      let g = groups.get(groupId);
      if (!g) {
        g = { runs: 0, passed: 0, speedups: [], cost: 0 };
        for (const k of groupKeys) g[k] = r[k];
        groups.set(groupId, g);
      }
      g.runs += 1;
      if (r.outcome === "passed") g.passed += 1;
      if (r.outcome === "passed" && typeof r.speedup === "number") {
        g.speedups.push(r.speedup);
      }
      if (typeof r.cost_gpu_seconds === "number") g.cost += r.cost_gpu_seconds;
    }
    const out = [];
    for (const g of groups.values()) {
      g.success_rate = g.runs ? g.passed / g.runs : 0;
      g.mean_speedup = g.speedups.length
        ? g.speedups.reduce((a, b) => a + b, 0) / g.speedups.length
        : null;
      g.cost_gpu_seconds = g.cost;
      out.push(g);
    }
    return out;
  }

  function sortRows(rows, tableId, defaultKey, defaultDir = "desc") {
    let st = SORT_STATE.get(tableId);
    if (!st) {
      st = { key: defaultKey, direction: defaultDir };
      SORT_STATE.set(tableId, st);
    }
    rows.sort((a, b) => {
      const av = a[st.key], bv = b[st.key];
      if (av === null || av === undefined) return 1;
      if (bv === null || bv === undefined) return -1;
      if (av === bv) return 0;
      const cmp = (typeof av === "number" && typeof bv === "number")
        ? av - bv
        : String(av).localeCompare(String(bv));
      return st.direction === "asc" ? cmp : -cmp;
    });
    return st;
  }

  function bindHeaderSort(table, defaultKey, defaultDir, rerender) {
    const ths = table.querySelectorAll("thead th[data-sortable='true']");
    let st = SORT_STATE.get(table.id) || { key: defaultKey, direction: defaultDir };
    SORT_STATE.set(table.id, st);
    for (const th of ths) {
      th.onclick = () => {
        const key = th.dataset.key;
        const cur = SORT_STATE.get(table.id);
        if (cur.key === key) {
          cur.direction = cur.direction === "asc" ? "desc" : "asc";
        } else {
          cur.key = key;
          cur.direction = "desc";
        }
        rerender();
      };
      const cur = SORT_STATE.get(table.id);
      th.classList.remove("sort-asc", "sort-desc");
      if (th.dataset.key === cur.key) {
        th.classList.add(cur.direction === "asc" ? "sort-asc" : "sort-desc");
      }
    }
  }

  function renderLeaderboard() {
    loadData().then(() => {
      const filters = document.getElementById("filters");
      if (filters) buildFilters(filters, ["level", "persona", "scenario"]);

      const visibleRuns = filteredRuns();

      // By-model table.
      const byModel = aggregate(visibleRuns, ["model"]);
      const t1 = document.getElementById("by-model");
      bindHeaderSort(t1, "success_rate", "desc", renderLeaderboard);
      const st1 = sortRows(byModel, "by-model", "success_rate", "desc");
      const peakSpeedup = Math.max(0, ...byModel.map(g => g.mean_speedup || 0));
      const tbody1 = t1.querySelector("tbody");
      tbody1.innerHTML = "";
      if (!byModel.length) {
        tbody1.innerHTML = '<tr><td colspan="5" class="empty">No runs match the current filter.</td></tr>';
      }
      for (const g of byModel) {
        const tr = document.createElement("tr");
        if (g.mean_speedup && g.mean_speedup === peakSpeedup) tr.classList.add("peak");
        tr.innerHTML = `
          <td>${fmtStr(g.model)}</td>
          <td class="numeric">${fmtInt(g.runs)}</td>
          <td class="numeric">${fmtPct(g.success_rate, 0)}</td>
          <td class="numeric">${fmtNumber(g.mean_speedup, 2)}</td>
          <td class="numeric">${fmtNumber(g.cost_gpu_seconds, 1)}</td>
        `;
        tbody1.appendChild(tr);
      }

      // By-model-and-level table.
      const byModelLevel = aggregate(visibleRuns, ["model", "level"]);
      const t2 = document.getElementById("by-model-level");
      bindHeaderSort(t2, "success_rate", "desc", renderLeaderboard);
      sortRows(byModelLevel, "by-model-level", "success_rate", "desc");
      const tbody2 = t2.querySelector("tbody");
      tbody2.innerHTML = "";
      if (!byModelLevel.length) {
        tbody2.innerHTML = '<tr><td colspan="6" class="empty">No runs match the current filter.</td></tr>';
      }
      for (const g of byModelLevel) {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${fmtStr(g.model)}</td>
          <td class="numeric">${fmtInt(g.level)}</td>
          <td class="numeric">${fmtInt(g.runs)}</td>
          <td class="numeric">${fmtPct(g.success_rate, 0)}</td>
          <td class="numeric">${fmtNumber(g.mean_speedup, 2)}</td>
          <td class="numeric">${fmtNumber(g.cost_gpu_seconds, 1)}</td>
        `;
        tbody2.appendChild(tr);
      }

      const gen = document.getElementById("generated-at");
      if (gen && DATA.generated_at) gen.textContent = "data: " + DATA.generated_at;
    }).catch(err => {
      const tbody = document.querySelector("#by-model tbody");
      if (tbody) tbody.innerHTML = `<tr><td colspan="5" class="empty">${err.message}</td></tr>`;
    });
  }

  function renderRunsIndex() {
    loadData().then(() => {
      const filters = document.getElementById("filters");
      if (filters) buildFilters(filters, ["level", "persona", "scenario", "model"]);

      const visibleRuns = filteredRuns().slice();
      const t = document.getElementById("runs");
      bindHeaderSort(t, "timestamp", "desc", renderRunsIndex);
      sortRows(visibleRuns, "runs", "timestamp", "desc");
      const tbody = t.querySelector("tbody");
      tbody.innerHTML = "";
      if (!visibleRuns.length) {
        tbody.innerHTML = '<tr><td colspan="10" class="empty">No runs match the current filter.</td></tr>';
      }
      for (const r of visibleRuns) {
        const tr = document.createElement("tr");
        const outcomeClass = "outcome-" + (r.outcome || "incomplete");
        const probLabel = r.problem_name
          ? `${r.problem_id} ${r.problem_name}`
          : fmtInt(r.problem_id);
        tr.innerHTML = `
          <td>${fmtStr(r.timestamp)}</td>
          <td>${fmtStr(r.scenario)}</td>
          <td>${fmtStr(r.model)}</td>
          <td>${fmtStr(r.persona)}</td>
          <td class="numeric">${fmtInt(r.level)}</td>
          <td>${fmtStr(probLabel)}</td>
          <td class="${outcomeClass}">${fmtStr(r.outcome)}</td>
          <td class="numeric">${fmtNumber(r.speedup, 2)}</td>
          <td class="numeric">${fmtNumber(r.cost_gpu_seconds, 1)}</td>
          <td><a href="${r.viewer_path}">trace</a></td>
        `;
        tbody.appendChild(tr);
      }

      const gen = document.getElementById("generated-at");
      if (gen && DATA.generated_at) gen.textContent = "data: " + DATA.generated_at;
    }).catch(err => {
      const tbody = document.querySelector("#runs tbody");
      if (tbody) tbody.innerHTML = `<tr><td colspan="10" class="empty">${err.message}</td></tr>`;
    });
  }

  return { renderLeaderboard, renderRunsIndex };
})();
