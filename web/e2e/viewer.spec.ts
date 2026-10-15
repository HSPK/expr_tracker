import {test, expect} from "@playwright/test";
import {spawn, type ChildProcess} from "node:child_process";
import {mkdir, mkdtemp, rm, writeFile} from "node:fs/promises";
import {tmpdir} from "node:os";
import {resolve, join} from "node:path";
import {createServer} from "node:net";

let process: ChildProcess;
let directory: string;
let address: string;
let alternate: string;

test.beforeAll(async () => {
  directory = await mkdtemp(join(tmpdir(), "et-viewer-browser-"));
  const run = join(directory, "demo", "training");
  await mkdir(join(run, "metrics.jsonl.parts"), {recursive: true});
  await writeFile(join(run, "metrics.jsonl"), '{"_step":0,"_time":1,"loss":1}\n{"_step":1,"_time":2,"loss":NaN}\n');
  await writeFile(join(run, "metrics.jsonl.parts", "00000001.jsonl"), '{"_step":2,"_time":3,"loss":0.25}\n');
  await writeFile(join(run, "config.json"), '{"learning_rate":0.001}');
  alternate = await mkdtemp(join(tmpdir(), "et-viewer-alternate-"));
  await mkdir(join(alternate, "other-run"));
  await writeFile(join(alternate, "other-run", "metrics.jsonl"), '{"_step":7,"_time":8,"accuracy":0.9}\n');
  const socket = createServer();
  const port = await new Promise<number>(resolve => socket.listen(0, "127.0.0.1", () => {
    const bound = socket.address();
    if (bound === null || typeof bound === "string") throw new Error("No reserved port");
    resolve(bound.port);
  }));
  await new Promise<void>((resolve, reject) => socket.close(error => error ? reject(error) : resolve()));
  address = `http://127.0.0.1:${port}`;
  process = spawn(resolve("../.venv/bin/python"), [
    "-c", "from expr_tracker.cli import main; main()", "serve",
    "--root", directory, "--port", String(port),
  ], {cwd: resolve(".."), stdio: ["ignore", "pipe", "pipe"]});
  let diagnostics = "";
  let lastError = "";
  process.stderr?.on("data", chunk => {diagnostics += chunk.toString();});
  const deadline = Date.now() + 15_000;
  while (true) {
    if (process.exitCode !== null) throw new Error(`Viewer exited: ${diagnostics}`);
    try {
      const response = await fetch(`${address}/api/runs`);
      if (response.ok) break;
    } catch (error) {
      if (!(error instanceof TypeError)) throw error;
      lastError = error.message;
    }
    if (Date.now() > deadline) throw new Error(`Viewer did not start: ${diagnostics} ${lastError}`);
    await new Promise(resolve => setTimeout(resolve, 50));
  }
});

test.afterAll(async () => {
  if (process && process.exitCode === null) {
    process.kill("SIGTERM");
    await new Promise<void>(resolve => process.once("exit", () => resolve()));
  }
  if (directory) await rm(directory, {recursive: true, force: true});
  if (alternate) await rm(alternate, {recursive: true, force: true});
});

test("opens real rotated logs, renders charts and inspects exact records", async ({page}) => {
  const errors: string[] = [];
  page.on("pageerror", error => errors.push(error.message));
  await page.goto(address);
  await page.getByLabel("Live", {exact: true}).uncheck();
  await page.locator(".run").filter({hasText: "training"}).getByRole("checkbox").check();
  await expect(page.locator(".chart-card")).toContainText("loss");
  await expect(page.locator(".uplot canvas").first()).toBeVisible();
  await expect(page.locator(".chart-card")).toContainText("1 non-finite observations");
  await page.getByRole("button", {name: "Records", exact: true}).click();
  await expect(page.locator("#table")).toContainText("NaN");
  await expect(page.locator("#table")).toContainText("0.25");
  await page.getByRole("button", {name: "Config & summary", exact: true}).click();
  await expect(page.locator("#config")).toContainText("learning_rate");
  expect(errors).toEqual([]);
});

test("browses server paths, switches roots, and keeps pages independent", async ({page, context}) => {
  await page.goto(address);
  await page.getByLabel("Live", {exact: true}).uncheck();
  await page.locator(".run").filter({hasText: "training"}).getByRole("checkbox").check();
  await expect(page.locator(".chart-card")).toContainText("loss");
  const other = await context.newPage();
  await other.goto(address);
  await expect(other.locator(".run")).toContainText("training");
  await page.getByRole("button", {name: "Choose directory", exact: true}).click();
  const dialog = page.getByRole("dialog");
  await expect(dialog.locator("#directory-list")).toContainText("demo");
  await dialog.getByRole("button", {name: "demo", exact: true}).click();
  await expect(dialog.locator("#browse-path")).toHaveText(join(directory, "demo"));
  await dialog.getByRole("button", {name: "Up", exact: true}).click();
  await expect(dialog.locator("#browse-path")).toHaveText(directory);
  await dialog.getByLabel("Server directory path").fill(join(alternate, "missing"));
  await dialog.getByRole("button", {name: "Go", exact: true}).click();
  await expect(dialog.locator("#directory-error")).toBeVisible();
  await expect(dialog.getByRole("button", {name: "Use this directory"})).toBeDisabled();
  await dialog.getByLabel("Server directory path").fill(alternate);
  await dialog.getByRole("button", {name: "Go", exact: true}).click();
  await expect(dialog.locator("#browse-path")).toHaveText(alternate);
  await dialog.getByRole("button", {name: "Use this directory"}).click();
  await expect(dialog).not.toBeVisible();
  await expect(page.locator("#current-directory")).toHaveText(alternate);
  await expect(page.locator("#detail")).not.toBeVisible();
  await expect(page.locator(".run")).toContainText("other-run");
  await page.locator(".run").getByRole("checkbox").check();
  await expect(page.locator(".chart-card")).toContainText("accuracy");
  await other.getByRole("button", {name: "Refresh", exact: true}).click();
  await expect(other.locator(".run")).toContainText("training");
  await page.reload();
  await expect(page.locator("#current-directory")).toHaveText(alternate);
  await expect(page.locator(".run")).toContainText("other-run");
});

test("cancelling directory selection leaves the active root unchanged", async ({page}) => {
  await page.goto(address);
  await page.getByRole("button", {name: "Choose directory", exact: true}).click();
  const dialog = page.getByRole("dialog");
  await expect(dialog.locator("#browse-path")).toHaveText(directory);
  await dialog.getByLabel("Server directory path").fill(alternate);
  await dialog.getByRole("button", {name: "Go", exact: true}).click();
  await expect(dialog.locator("#browse-path")).toHaveText(alternate);
  await dialog.getByRole("button", {name: "Cancel", exact: true}).click();
  await expect(page.locator("#current-directory")).toHaveText(directory);
  await expect(page.locator(".run")).toContainText("training");
});

test("a failed bookmarked path remains recoverable through the picker", async ({page}) => {
  await page.goto(`${address}/?root=${encodeURIComponent(join(alternate, "missing"))}`);
  await expect(page.locator("#error")).toContainText("Cannot access server directory");
  await page.getByRole("button", {name: "Choose directory", exact: true}).click();
  const dialog = page.getByRole("dialog");
  await expect(dialog.locator("#browse-path")).toHaveText(directory);
  await dialog.getByRole("button", {name: "Use this directory"}).click();
  await expect(page.locator(".run")).toContainText("training");
  await expect(page.locator("#error")).not.toBeVisible();
});
