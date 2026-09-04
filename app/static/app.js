(() => {
  "use strict";

  const statusEl = document.getElementById("status");
  const statusLabel = document.getElementById("statusLabel");

  const archiveEl = document.getElementById("archive");
  const archiveToggle = document.getElementById("archiveToggle");
  const archiveToggleCount = document.getElementById("archiveToggleCount");
  const archiveCount = document.getElementById("archiveCount");
  const docList = document.getElementById("docList");
  const docEmpty = document.getElementById("docEmpty");

  const fileInput = document.getElementById("fileInput");
  const uploadZone = document.getElementById("uploadZone");
  const uploadHint = document.getElementById("uploadHint");
  const uploadProgress = document.getElementById("uploadProgress");
  const uploadFill = document.getElementById("uploadFill");
  const uploadStatus = document.getElementById("uploadStatus");

  const transcript = document.getElementById("transcript");
  const emptyState = document.getElementById("emptyState");
  const suggestions = document.getElementById("suggestions");
  const turnTemplate = document.getElementById("turnTemplate");

  const composerForm = document.getElementById("composerForm");
  const composerInput = document.getElementById("composerInput");
  const composerSend = document.getElementById("composerSend");

  const SUGGESTED_QUERIES = [
    "How many teams compete in the IPL?",
    "When was Chennai Super Kings founded?",
    "How many IPL titles has Mumbai Indians won?",
  ];

  let turnCount = 0;

  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    const kb = bytes / 1024;
    if (kb < 1024) return `${kb.toFixed(kb < 10 ? 1 : 0)} KB`;
    return `${(kb / 1024).toFixed(1)} MB`;
  }

  function pad(n) {
    return String(n).padStart(2, "0");
  }

  // --- Health check --------------------------------------------------

  async function checkHealth() {
    try {
      const res = await fetch("/health");
      if (!res.ok) throw new Error("unhealthy");
      statusEl.dataset.state = "online";
      statusLabel.textContent = "Online";
    } catch {
      statusEl.dataset.state = "offline";
      statusLabel.textContent = "Offline";
    }
  }

  // --- Archive ---------------------------------------------------------

  async function loadDocuments() {
    try {
      const res = await fetch("/documents");
      if (!res.ok) throw new Error("failed to load documents");
      const docs = await res.json();
      renderDocuments(docs);
    } catch (err) {
      console.error(err);
    }
  }

  function renderDocuments(docs) {
    docList.innerHTML = "";
    const count = docs.length;
    archiveCount.textContent = `${count} file${count === 1 ? "" : "s"}`;
    archiveToggleCount.textContent = String(count);
    docEmpty.hidden = count > 0;

    docs.forEach((doc, i) => {
      const li = document.createElement("li");
      li.className = "doc-list__row";
      li.style.animationDelay = `${i * 40}ms`;
      li.innerHTML = `
        <span class="doc-list__index">${pad(i + 1)}</span>
        <span class="doc-list__meta">
          <span class="doc-list__name" title="${doc.filename}">${doc.filename}</span>
          <span class="doc-list__size">${formatBytes(doc.size_bytes)}</span>
        </span>
      `;
      docList.appendChild(li);
    });
  }

  // --- Upload ------------------------------------------------------------

  function setUploadIdle() {
    uploadZone.hidden = false;
    uploadProgress.hidden = true;
    uploadHint.textContent = "Drop a PDF, or click to browse";
  }

  function uploadFile(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      uploadHint.textContent = "Only PDF files are supported";
      return;
    }

    uploadZone.hidden = true;
    uploadProgress.hidden = false;
    uploadFill.style.animationPlayState = "running";
    uploadStatus.dataset.tone = "";
    uploadStatus.textContent = `Uploading ${file.name}…`;

    const formData = new FormData();
    formData.append("file", file);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/upload");

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        const data = JSON.parse(xhr.responseText);
        uploadStatus.dataset.tone = "success";
        uploadStatus.textContent = `Added — ${data.chunks_added} chunks indexed`;
        loadDocuments();
      } else {
        let detail = "Upload failed";
        try {
          detail = JSON.parse(xhr.responseText).detail || detail;
        } catch {
          /* keep default */
        }
        uploadStatus.dataset.tone = "error";
        uploadStatus.textContent = detail;
      }
      setTimeout(setUploadIdle, 2600);
    };

    xhr.onerror = () => {
      uploadStatus.dataset.tone = "error";
      uploadStatus.textContent = "Upload failed — check your connection";
      setTimeout(setUploadIdle, 2600);
    };

    xhr.send(formData);
  }

  uploadZone.addEventListener("click", () => fileInput.click());
  uploadZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
  fileInput.addEventListener("change", () => uploadFile(fileInput.files[0]));

  ["dragenter", "dragover"].forEach((evt) =>
    uploadZone.addEventListener(evt, (e) => {
      e.preventDefault();
      uploadZone.dataset.active = "true";
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    uploadZone.addEventListener(evt, (e) => {
      e.preventDefault();
      uploadZone.dataset.active = "false";
    })
  );
  uploadZone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    uploadFile(file);
  });

  // --- Mobile archive drawer ---------------------------------------------

  archiveToggle.addEventListener("click", () => {
    const open = archiveEl.dataset.open === "true";
    archiveEl.dataset.open = String(!open);
    archiveToggle.setAttribute("aria-expanded", String(!open));
  });

  // --- Suggestions ---------------------------------------------------------

  SUGGESTED_QUERIES.forEach((q) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "suggestion";
    btn.textContent = q;
    btn.addEventListener("click", () => {
      composerInput.value = q;
      composerInput.dispatchEvent(new Event("input"));
      composerInput.focus();
    });
    suggestions.appendChild(btn);
  });

  // --- Composer / chat -----------------------------------------------------

  function autoGrow() {
    composerInput.style.height = "auto";
    composerInput.style.height = `${Math.min(composerInput.scrollHeight, 240)}px`;
  }
  composerInput.addEventListener("input", autoGrow);

  composerInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      composerForm.requestSubmit();
    }
  });

  async function submitQuery(query) {
    emptyState.hidden = true;
    turnCount += 1;

    const node = turnTemplate.content.cloneNode(true);
    const turn = node.querySelector(".turn");
    turn.querySelector(".turn__index").textContent = `Q${pad(turnCount)}`;
    turn.querySelector(".turn__question").textContent = query;
    const body = turn.querySelector(".turn__body");
    const sourcesEl = turn.querySelector(".turn__sources");
    transcript.appendChild(node);
    turn.scrollIntoView({ behavior: "smooth", block: "start" });

    composerSend.disabled = true;

    try {
      const res = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      if (!res.ok) throw new Error(`Server responded ${res.status}`);

      const data = await res.json();
      body.innerHTML = "";
      const p = document.createElement("p");
      p.textContent = data.response;
      body.appendChild(p);

      if (Array.isArray(data.source) && data.source.length) {
        sourcesEl.hidden = false;
        data.source.forEach((src, i) => {
          const label = String(src).replace(/^\[\d+\]\s*/, "").trim();
          const tag = document.createElement("span");
          tag.className = "source-tag";
          const indexEl = document.createElement("span");
          indexEl.className = "source-tag__index";
          indexEl.textContent = `SRC ${pad(i + 1)}`;
          tag.appendChild(indexEl);
          if (label) tag.appendChild(document.createTextNode(label));
          sourcesEl.appendChild(tag);
        });
      }
    } catch (err) {
      body.dataset.tone = "error";
      body.innerHTML = "";
      const p = document.createElement("p");
      p.textContent = "Couldn't reach the archive. Check the server and try again.";
      body.appendChild(p);
      console.error(err);
    } finally {
      composerSend.disabled = false;
    }
  }

  composerForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const query = composerInput.value.trim();
    if (!query) return;
    composerInput.value = "";
    autoGrow();
    submitQuery(query);
  });

  // --- Init ---------------------------------------------------------------

  checkHealth();
  loadDocuments();
})();
