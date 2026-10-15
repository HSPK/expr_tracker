import {get} from "./api";

interface Listing {
  path: string;
  parent: string | null;
  directories: {name: string; path: string}[];
}

export function directoryPicker(
  trigger: HTMLElement,
  currentPath: () => string,
  select: (path: string, signal: AbortSignal) => Promise<void>,
): void {
  const dialog = document.createElement("dialog");
  dialog.className = "directory-picker";
  dialog.setAttribute("aria-label", "Choose server directory");
  dialog.innerHTML = `
    <h1>Choose server directory</h1>
    <p>Browse the server filesystem, not files on your browser's computer.</p>
    <form><label for="server-path">Server directory path</label>
      <div class="path-controls"><input id="server-path" autocomplete="off" spellcheck="false"><button>Go</button></div></form>
    <div class="directory-navigation"><button id="parent-directory">Up</button><span id="browse-path"></span></div>
    <div id="directory-list"></div><div id="directory-error" role="alert" hidden></div>
    <div class="directory-actions"><button id="cancel-directory">Cancel</button><button id="select-directory">Use this directory</button></div>`;
  document.body.append(dialog);
  const input = dialog.querySelector<HTMLInputElement>("input")!;
  const list = dialog.querySelector<HTMLDivElement>("#directory-list")!;
  const error = dialog.querySelector<HTMLDivElement>("#directory-error")!;
  const up = dialog.querySelector<HTMLButtonElement>("#parent-directory")!;
  const confirm = dialog.querySelector<HTMLButtonElement>("#select-directory")!;
  let listing: Listing | null = null;
  let controller: AbortController | null = null;

  function failure(reason: unknown): void {
    if (reason instanceof DOMException && reason.name === "AbortError") return;
    error.hidden = false;
    error.textContent = reason instanceof Error ? reason.message : String(reason);
  }

  function busy(value: boolean): void {
    for (const button of dialog.querySelectorAll<HTMLButtonElement>("button:not(#cancel-directory)")) button.disabled = value;
    input.disabled = value;
    if (!value) {
      up.disabled = !listing?.parent;
      confirm.disabled = !listing;
    }
  }

  async function browse(path: string): Promise<void> {
    controller?.abort();
    controller = new AbortController();
    const signal = controller.signal;
    listing = null;
    list.replaceChildren();
    dialog.querySelector("#browse-path")!.textContent = "";
    busy(true);
    error.hidden = true;
    try {
      const result = await get<Listing>("/api/directories", {path}, signal);
      if (signal.aborted) return;
      listing = result;
      input.value = result.path;
      dialog.querySelector("#browse-path")!.textContent = result.path;
      list.replaceChildren();
      for (const directory of result.directories) {
        const button = document.createElement("button");
        button.textContent = directory.name;
        button.onclick = () => void browse(directory.path);
        list.append(button);
      }
      if (!result.directories.length) list.textContent = "No subdirectories.";
    } catch (reason) {
      failure(reason);
    } finally {
      if (!signal.aborted) busy(false);
    }
  }

  trigger.onclick = () => {
    listing = null;
    list.replaceChildren();
    dialog.showModal();
    void browse(currentPath());
  };
  dialog.querySelector("form")!.onsubmit = event => {
    event.preventDefault();
    void browse(input.value);
  };
  up.onclick = () => {if (listing?.parent) void browse(listing.parent);};
  confirm.onclick = async () => {
    if (!listing) return;
    controller?.abort();
    controller = new AbortController();
    const signal = controller.signal;
    busy(true);
    error.hidden = true;
    try {
      await select(listing.path, signal);
      if (!signal.aborted) dialog.close();
    } catch (reason) {
      failure(reason);
    } finally {
      if (!signal.aborted) busy(false);
    }
  };
  dialog.querySelector<HTMLButtonElement>("#cancel-directory")!.onclick = () => dialog.close();
  dialog.addEventListener("close", () => controller?.abort());
  dialog.addEventListener("cancel", () => controller?.abort());
}
