import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import "./styles.css";
import {align, display, get, type History, type Records, type Run, type RunDetails} from "./api";
import {directoryPicker} from "./directories";

const palette = ["#8ba6ff", "#f2b36e", "#6fd4bd", "#d999ee", "#ed899b", "#a4cc72"];
const root = document.querySelector<HTMLDivElement>("#app")!;
root.innerHTML = `
  <header><div class="brand"><b>et</b><span>Experiment tracker<small>Local logs · read only</small></span></div>
    <div class="header-controls"><button id="choose-directory" disabled>Choose directory</button><label><input id="follow" type="checkbox" checked> Live</label><button id="refresh">Refresh</button></div></header>
  <div class="workspace"><aside><div class="section-label">SERVER DIRECTORY</div><div id="current-directory" class="muted"></div><div class="section-label">EXPERIMENTS</div><input id="search" placeholder="Search runs…" aria-label="Search runs"><div id="runs"></div></aside>
    <main><div id="empty" class="empty"><h1>Your experiments, on your machine.</h1><p>Select a run to inspect its logs. Select multiple runs to compare.</p></div>
      <section id="detail" hidden><div class="run-heading"><div><h1 id="title"></h1><p id="subtitle"></p></div><div id="freshness"></div></div>
        <nav><button data-tab="charts" class="active">Metrics</button><button data-tab="records">Records</button><button data-tab="config">Config & summary</button><button data-tab="spans">Spans</button></nav>
        <div class="toolbar"><label>Source <select id="file"></select></label><label>Axis <select id="axis"><option value="step">Step</option><option value="time">Wall time</option></select></label>
          <label>From step <input id="from" type="number" placeholder="Any"></label><label>To step <input id="to" type="number" placeholder="Exclusive"></label>
          <button id="apply">Apply</button><span class="muted">Comparisons align by the selected axis, not by ingestion order.</span></div>
        <div id="warnings" role="status"></div><div id="metrics"></div><div id="charts" class="charts"></div>
        <div id="records" class="tab-content" hidden><button id="download">Download this page</button><div id="table"></div><button id="older" hidden>Older records</button></div>
        <div id="config" class="tab-content" hidden></div>
        <div id="spans" class="tab-content" hidden><p>Recorded spans, including rotated segments. Times are seconds since the Unix epoch.</p><div id="span-table"></div></div>
      </section><div id="error" role="alert" hidden></div>
    </main></div>`;

function element<T extends HTMLElement>(id: string): T {
  return document.getElementById(id)! as T;
}

const selected = new Set<string>();
const enabledMetrics = new Set<string>();
let runs: Run[] = [];
let currentDetails: RunDetails | null = null;
let plots: uPlot[] = [];
let tab = "charts";
let request: AbortController | null = null;
let timerBusy = false;
let nextRecords: string | null = null;
let visibleRecords: Record<string, unknown>[] = [];
let sourceKey = "";
let metricsInitialized = false;
let serverRoot: string | undefined;
let runsRequest: AbortController | null = null;
let ready = false;

function text(parent: HTMLElement, content: string, className?: string): HTMLDivElement {
  const child = document.createElement("div");
  child.textContent = content;
  if (className) child.className = className;
  parent.append(child);
  return child;
}

function showError(error: unknown): void {
  if (error instanceof DOMException && error.name === "AbortError") return;
  const notice = element("error");
  notice.hidden = false;
  notice.textContent = error instanceof Error ? error.message : String(error);
}

function drawRuns(): void {
  const container = element("runs");
  container.replaceChildren();
  const filter = element<HTMLInputElement>("search").value.toLowerCase();
  for (const run of runs.filter(run => `${run.project}/${run.name}`.toLowerCase().includes(filter))) {
    const label = document.createElement("label");
    label.className = `run${selected.has(run.id) ? " selected" : ""}`;
    const check = document.createElement("input");
    check.type = "checkbox";
    check.checked = selected.has(run.id);
    check.onchange = () => {
      if (check.checked && selected.size >= 6) {
        check.checked = false;
        showError(new Error("Compare at most six runs at once."));
        return;
      }
      check.checked ? selected.add(run.id) : selected.delete(run.id);
      drawRuns();
      void refreshDetails().catch(showError);
    };
    label.append(check);
    const body = document.createElement("div");
    text(body, run.name, "run-name");
    text(body, `${run.project} · ${(run.bytes / (1024 * 1024)).toFixed(1)} MiB`, "muted");
    label.append(body);
    container.append(label);
  }
  if (!container.childElementCount) text(container, "No runs found under this root.", "muted");
}

function table(container: HTMLElement, rows: Record<string, unknown>[]): void {
  container.replaceChildren();
  if (!rows.length) {
    text(container, "No records in this range.", "muted");
    return;
  }
  const keys = [...new Set(rows.flatMap(row => Object.keys(row)))];
  const wrapper = document.createElement("div");
  wrapper.className = "table-scroll";
  const result = document.createElement("table");
  const head = result.createTHead().insertRow();
  for (const key of keys) {
    const cell = document.createElement("th");
    cell.textContent = key;
    head.append(cell);
  }
  const body = result.createTBody();
  for (const row of rows) {
    const tr = body.insertRow();
    for (const key of keys) tr.insertCell().textContent = key in row ? display(row[key]) : "—";
  }
  wrapper.append(result);
  container.append(wrapper);
}

function params(run: string) {
  return {
    root: serverRoot,
    run,
    file: element<HTMLSelectElement>("file").value,
    from_step: element<HTMLInputElement>("from").value || undefined,
    to_step: element<HTMLInputElement>("to").value || undefined,
  };
}

function drawMetrics(metrics: string[]): void {
  const container = element("metrics");
  container.replaceChildren();
  if (!metricsInitialized) {
    for (const metric of metrics.includes("loss") ? ["loss"] : metrics.slice(0, 4)) enabledMetrics.add(metric);
    metricsInitialized = true;
  }
  for (const metric of metrics) {
    const label = document.createElement("label");
    label.className = "metric";
    const check = document.createElement("input");
    check.type = "checkbox";
    check.checked = enabledMetrics.has(metric);
    check.onchange = () => {
      if (check.checked && enabledMetrics.size >= 12) {
        check.checked = false;
        showError(new Error("Display at most twelve metrics at once."));
        return;
      }
      check.checked ? enabledMetrics.add(metric) : enabledMetrics.delete(metric);
      void refreshTab().catch(showError);
    };
    label.append(check, document.createTextNode(metric));
    container.append(label);
  }
}

async function refreshDetails(): Promise<void> {
  request?.abort();
  request = new AbortController();
  const signal = request.signal;
  element("error").hidden = true;
  const active = [...selected][0];
  element("empty").hidden = !!active;
  element("detail").hidden = !active;
  if (!active) {
    currentDetails = null;
    plots.forEach(plot => plot.destroy());
    plots = [];
    return;
  }
  const run = runs.find(run => run.id === active);
  if (!run) throw new Error("This run is no longer available.");
  const fileSelect = element<HTMLSelectElement>("file");
  const previous = fileSelect.value;
  fileSelect.replaceChildren();
  const metricFiles = run.files.filter(file => file.kind === "metrics");
  const files = metricFiles.length ? metricFiles : run.files.filter(file => file.kind === "spans");
  for (const file of files) {
    const option = document.createElement("option");
    option.value = file.name;
    option.textContent = `${file.stream ?? "default"} · rank ${file.rank}`;
    fileSelect.append(option);
  }
  if (files.some(file => file.name === previous)) fileSelect.value = previous;
  const chartsButton = document.querySelector<HTMLButtonElement>('[data-tab="charts"]')!;
  chartsButton.disabled = !metricFiles.length;
  if (!metricFiles.length && tab === "charts") {
    tab = "spans";
    document.querySelectorAll<HTMLButtonElement>("[data-tab]").forEach(button => {
      button.classList.toggle("active", button.dataset.tab === tab);
    });
  }
  const newKey = `${active}/${fileSelect.value}`;
  if (newKey !== sourceKey) {
    enabledMetrics.clear();
    metricsInitialized = false;
    sourceKey = newKey;
    nextRecords = null;
    visibleRecords = [];
  }
  const details = await get<RunDetails>("/api/run", {root: serverRoot, run: active, file: fileSelect.value || undefined}, signal);
  if (signal.aborted) return;
  currentDetails = details;
  element("title").textContent = selected.size > 1 ? `${selected.size} runs compared` : run.name;
  element("subtitle").textContent = `${run.project} / ${run.name} · ${currentDetails.rows.toLocaleString()} steps`;
  element("freshness").textContent = `Last write ${new Date(run.modified * 1000).toLocaleString()}`;
  drawMetrics(currentDetails.metrics);
  await refreshTab(signal);
}

async function refreshTab(signal = request?.signal): Promise<void> {
  if (!currentDetails || !selected.size) return;
  element("error").hidden = true;
  for (const name of ["charts", "records", "config", "spans"]) element(name).hidden = name !== tab;
  element("metrics").hidden = tab !== "charts";
  const active = [...selected][0];
  const warnings = [...currentDetails.warnings];
  if (tab === "charts") {
    const metrics = [...enabledMetrics];
    if (!metrics.length) {
      plots.forEach(plot => plot.destroy());
      plots = [];
      const container = element("charts");
      container.replaceChildren();
      text(container, "Select metrics to draw their history.", "muted");
      element("warnings").textContent = warnings.join(" ");
      return;
    }
    const ids = [...selected];
    const histories = await Promise.all(ids.map(run => get<History>("/api/history", {
      ...params(run), metrics: JSON.stringify(metrics), axis: element<HTMLSelectElement>("axis").value, max_points: 1000,
    }, signal)));
    if (signal?.aborted) return;
    for (const history of histories) warnings.push(...history.warnings);
    plots.forEach(plot => plot.destroy());
    plots = [];
    const container = element("charts");
    container.replaceChildren();
    for (const metric of metrics) {
      const card = document.createElement("article");
      card.className = "chart-card";
      text(card, metric, "chart-title");
      const surface = document.createElement("div");
      card.append(surface);
      container.append(card);
      const data = align(histories, metric);
      const plot = new uPlot({
        width: Math.max(240, card.clientWidth - 32), height: 280,
        scales: {x: {time: element<HTMLSelectElement>("axis").value === "time"}},
        axes: [{stroke: "#929cac", grid: {stroke: "#28303c"}}, {stroke: "#929cac", grid: {stroke: "#28303c"}}],
        series: [{}, ...ids.map((id, index) => ({
          label: runs.find(run => run.id === id)?.name ?? id,
          stroke: palette[index % palette.length], width: 1.5, spanGaps: false,
        }))],
      }, [data.x, ...data.ys], surface);
      plots.push(plot);
      const count = histories.reduce((sum, history) => sum + (history.series.find(series => series.metric === metric)?.nonfinite ?? 0), 0);
      if (count) text(card, `${count} non-finite observations shown as gaps.`, "muted");
    }
  } else if (tab === "records") {
    await loadRecords(false, signal);
  } else if (tab === "config") {
    const container = element("config");
    container.replaceChildren();
    text(container, "Config", "section-label");
    const config = document.createElement("pre");
    config.textContent = JSON.stringify(currentDetails.config, null, 2);
    container.append(config);
    text(container, "Summary", "section-label");
    const summary = document.createElement("pre");
    summary.textContent = JSON.stringify(currentDetails.summary, null, 2);
    container.append(summary);
  } else {
    const run = runs.find(run => run.id === active)!;
    const source = run.files.find(file => file.name === element<HTMLSelectElement>("file").value);
    const spanFile = source?.kind === "spans" ? source : run.files.find(file => file.kind === "spans" && file.stream === source?.stream && file.rank === source?.rank);
    if (!spanFile) {
      table(element("span-table"), []);
    } else {
      const spans = await get<Records>("/api/records", {...params(active), file: spanFile.name, limit: 100}, signal);
      if (signal?.aborted) return;
      warnings.push(...spans.warnings);
      table(element("span-table"), spans.records);
    }
  }
  element("warnings").textContent = [...new Set(warnings)].join(" ");
}

async function loadRecords(older: boolean, signal = request?.signal): Promise<void> {
  const data = await get<Records>("/api/records", {
    ...params([...selected][0]), limit: 100, before: older ? nextRecords ?? undefined : undefined,
  }, signal);
  if (signal?.aborted) return;
  visibleRecords = older ? [...visibleRecords, ...data.records] : data.records;
  nextRecords = data.next;
  table(element("table"), visibleRecords);
  element("older").hidden = !nextRecords;
}

async function refresh(): Promise<void> {
  if (!ready) return;
  runsRequest?.abort();
  runsRequest = new AbortController();
  const signal = runsRequest.signal;
  const result = await get<Run[]>("/api/runs", {root: serverRoot}, signal);
  if (signal.aborted) return;
  runs = result;
  for (const id of selected) if (!runs.some(run => run.id === id)) selected.delete(id);
  drawRuns();
  await refreshDetails();
}

directoryPicker(element("choose-directory"), () => serverRoot!, async (path, signal) => {
  const result = await get<Run[]>("/api/runs", {root: path}, signal);
  if (signal.aborted) return;
  runsRequest?.abort();
  request?.abort();
  serverRoot = path;
  ready = true;
  runs = result;
  selected.clear();
  enabledMetrics.clear();
  currentDetails = null;
  sourceKey = "";
  metricsInitialized = false;
  nextRecords = null;
  visibleRecords = [];
  element<HTMLInputElement>("search").value = "";
  element<HTMLSelectElement>("file").replaceChildren();
  element<HTMLInputElement>("from").value = "";
  element<HTMLInputElement>("to").value = "";
  element("current-directory").textContent = path;
  const url = new URL(location.href);
  url.searchParams.set("root", path);
  window.history.replaceState(null, "", url);
  drawRuns();
  await refreshDetails();
});

for (const button of document.querySelectorAll<HTMLButtonElement>("[data-tab]")) {
  button.onclick = () => {
    tab = button.dataset.tab!;
    document.querySelectorAll("[data-tab]").forEach(node => node.classList.toggle("active", node === button));
    void refreshTab().catch(showError);
  };
}
element("search").oninput = drawRuns;
element("refresh").onclick = () => void refresh().catch(showError);
element("apply").onclick = () => void refreshTab().catch(showError);
element<HTMLSelectElement>("file").onchange = () => void refreshDetails().catch(showError);
element<HTMLSelectElement>("axis").onchange = () => void refreshTab().catch(showError);
element("older").onclick = () => void loadRecords(true).catch(showError);
element("download").onclick = () => {
  const blob = new Blob([visibleRecords.map(row => JSON.stringify(row)).join("\n") + "\n"], {type: "application/x-ndjson"});
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "records.jsonl";
  link.click();
  URL.revokeObjectURL(link.href);
};
new ResizeObserver(() => {
  for (const plot of plots) {
    const card = plot.root.parentElement?.parentElement;
    if (card) plot.setSize({width: Math.max(240, card.clientWidth - 32), height: 280});
  }
}).observe(element("charts"));
setInterval(() => {
  if (!element<HTMLInputElement>("follow").checked || timerBusy || document.hidden) return;
  timerBusy = true;
  void refresh().catch(showError).finally(() => {timerBusy = false;});
}, 3000);
async function start(): Promise<void> {
  const initial = await get<{path: string}>("/api/root");
  serverRoot = initial.path;
  element("current-directory").textContent = serverRoot;
  element<HTMLButtonElement>("choose-directory").disabled = false;
  const wanted = new URLSearchParams(window.location.search).get("root");
  if (wanted !== null) {
    const chosen = await get<{path: string}>("/api/root", {root: wanted});
    serverRoot = chosen.path;
    element("current-directory").textContent = serverRoot;
  }
  ready = true;
  await refresh();
}
void start().catch(showError);
