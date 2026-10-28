// Frontend Controller for Procurement Invoice & Contract Auditor Cockpit

let currentInvoiceId = null;
let currentAuditData = null;

document.addEventListener("DOMContentLoaded", () => {
  loadSamples();
  setupUploadAndDropzone();

  // Setup re-generate button
  document.getElementById("btn-re-synth").addEventListener("click", async () => {
    const btn = document.getElementById("btn-re-synth");
    btn.disabled = true;
    btn.textContent = "Generating...";
    try {
      const res = await fetch("/api/dataset/generate", { method: "POST" });
      await res.json();
      await loadSamples();
      appendBotMsg("✨ Generated fresh synthetic procurement suite with seeded discrepancies!");
    } catch (err) {
      alert("Error generating dataset: " + err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = "⚡ Generate Dataset";
    }
  });

  // Setup Chatbot
  document.getElementById("btn-send-chat").addEventListener("click", handleSendMessage);
  document.getElementById("chat-input").addEventListener("keypress", (e) => {
    if (e.key === "Enter") handleSendMessage();
  });
});

// --- Tab Switching ---
function switchView(viewName) {
  document.querySelectorAll(".nav-tab-btn").forEach(btn => btn.classList.remove("active"));
  document.querySelectorAll(".view-panel").forEach(panel => panel.classList.remove("active"));

  if (viewName === "audit") {
    document.querySelectorAll(".nav-tab-btn")[0].classList.add("active");
    document.getElementById("view-audit").classList.add("active");
  } else if (viewName === "erp") {
    document.querySelectorAll(".nav-tab-btn")[1].classList.add("active");
    document.getElementById("view-erp").classList.add("active");
    loadERPTables();
  } else if (viewName === "vectordb") {
    document.querySelectorAll(".nav-tab-btn")[2].classList.add("active");
    document.getElementById("view-vectordb").classList.add("active");
    loadVectorDBClauses();
  }
}

// --- Upload & Immediate Processing ---
function setupUploadAndDropzone() {
  const uploadInput = document.getElementById("pdf-upload-input");
  document.getElementById("btn-upload").addEventListener("click", () => uploadInput.click());

  uploadInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) processUploadedFile(file);
  });

  const dropzone = document.getElementById("quick-dropzone");
  if (dropzone) {
    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.style.borderColor = "var(--accent)";
    });
    dropzone.addEventListener("dragleave", (e) => {
      e.preventDefault();
      dropzone.style.borderColor = "rgba(99, 102, 241, 0.4)";
    });
    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.style.borderColor = "rgba(99, 102, 241, 0.4)";
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        processUploadedFile(e.dataTransfer.files[0]);
      }
    });
  }
}

async function processUploadedFile(file) {
  switchView("audit");
  const body = document.getElementById("audit-body");
  body.innerHTML = `
    <div class="empty-state">
      <div style="font-size: 1.5rem; margin-bottom: 8px;">⏳</div>
      <p style="font-size: 1.05rem; font-weight: 600;">Processing ${file.name} immediately...</p>
      <p style="font-size: 0.82rem; color: var(--text-muted); margin-top: 4px;">Running dynamic PDF extraction, deterministic 3-way matching, and LangGraph contract audit.</p>
    </div>
  `;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/api/audit/pdf", { method: "POST", body: formData });
    const data = await res.json();
    currentInvoiceId = data.invoice_id;
    currentAuditData = data;
    renderAuditResult(data);
    appendBotMsg(`Audited uploaded document **${file.name}**. Audit Decision: **${data.decision.toUpperCase()}**.`);
  } catch (err) {
    body.innerHTML = `<div class="empty-state" style="color: var(--danger);">Upload & Processing failed: ${err.message}</div>`;
  }
}

// --- Sample Loading ---
async function loadSamples() {
  const listEl = document.getElementById("sample-list");
  try {
    const res = await fetch("/api/dataset/samples");
    const data = await res.json();

    listEl.innerHTML = "";
    data.invoices.forEach((inv, idx) => {
      const item = document.createElement("div");
      item.className = "sample-item" + (idx === 0 ? " active" : "");
      item.onclick = () => selectSample(inv.id, item);

      const badgeClass = `badge-${inv.expected_decision}`;
      item.innerHTML = `
        <div class="sample-title">
          <span>${inv.id}</span>
          <span class="badge ${badgeClass}">${inv.expected_decision.replace("_", " ")}</span>
        </div>
        <div class="sample-desc">${inv.vendor} • ${inv.po_number}</div>
      `;
      listEl.appendChild(item);
    });

    if (data.invoices.length > 0) {
      selectSample(data.invoices[0].id, listEl.children[0]);
    }
  } catch (err) {
    listEl.innerHTML = `<div class="empty-state">Failed to load samples: ${err.message}</div>`;
  }
}

async function selectSample(invoiceId, el) {
  document.querySelectorAll(".sample-item").forEach(item => item.classList.remove("active"));
  if (el) el.classList.add("active");

  currentInvoiceId = invoiceId;
  const body = document.getElementById("audit-body");
  body.innerHTML = `<div class="empty-state">Auditing <b>${invoiceId}</b> with LangGraph Multi-Agents...</div>`;

  try {
    const res = await fetch(`/api/audit/sample/${invoiceId}`, { method: "POST" });
    const data = await res.json();
    currentAuditData = data;
    renderAuditResult(data);
  } catch (err) {
    body.innerHTML = `<div class="empty-state" style="color: var(--danger);">Audit failed: ${err.message}</div>`;
  }
}

function renderAuditResult(data) {
  const body = document.getElementById("audit-body");
  const badge = document.getElementById("active-inv-badge");
  const reviewBar = document.getElementById("review-bar");
  const reviewStatus = document.getElementById("review-status-label");

  badge.className = `badge badge-${data.decision}`;
  badge.textContent = `${data.invoice_id} • ${data.decision.replace("_", " ").toUpperCase()}`;

  let riskColor = "var(--success)";
  if (data.risk_score >= 80) riskColor = "var(--danger)";
  else if (data.risk_score >= 40) riskColor = "var(--warning)";

  let html = `
    <!-- Key Metrics Stats -->
    <div class="stats-row">
      <div class="stat-box">
        <span class="stat-label">Audit Decision</span>
        <span class="stat-value" style="color: ${riskColor}; text-transform: uppercase; font-size: 1.15rem;">
          ${data.decision.replace("_", " ")}
        </span>
      </div>
      <div class="stat-box">
        <span class="stat-label">Risk Exposure Score</span>
        <span class="stat-value" style="color: ${riskColor};">
          ${data.risk_score}<span style="font-size: 0.85rem; font-weight: 400; color: var(--text-muted);">/100</span>
        </span>
      </div>
      <div class="stat-box">
        <span class="stat-label">Discrepancy Variance</span>
        <span class="stat-value" style="color: ${data.financial_exposure > 0 ? 'var(--danger)' : 'var(--text-main)'};">
          $${parseFloat(data.financial_exposure).toFixed(2)}
        </span>
      </div>
    </div>
  `;

  // Explicit Human-in-the-Loop review guidance banner
  if (data.requires_human_review || data.human_in_loop_message) {
    html += `
      <div class="hitl-banner">
        <div class="hitl-icon">⚠️</div>
        <div class="hitl-content">
          <h4>Human-in-the-Loop Review Alert</h4>
          <p>${data.human_in_loop_message || "Discrepancy detected requiring manual auditor sign-off."}</p>
        </div>
      </div>
    `;
  }

  // Investigation Summary
  if (data.investigation_summary) {
    html += `
      <div>
        <div class="section-title">📋 Multi-Agent Investigation Summary</div>
        <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid var(--card-border); border-radius: 8px; padding: 12px 14px; font-size: 0.84rem; line-height: 1.45; white-space: pre-line;">
          ${data.investigation_summary}
        </div>
      </div>
    `;
  }

  // Reason Codes & Discrepancies
  html += `<div><div class="section-title">🚨 Findings & Reason Codes (${data.reasons.length})</div>`;
  if (data.reasons.length === 0) {
    html += `<div style="font-size: 0.84rem; color: var(--success); padding: 8px 0;">✓ Zero discrepancies found. Clean 3-way match.</div>`;
  } else {
    data.reasons.forEach(r => {
      const cls = r.severity.toLowerCase();
      html += `
        <div class="reason-item ${cls}">
          <div class="reason-code">${r.code} • ${r.severity.toUpperCase()}</div>
          <div>${r.message}</div>
        </div>
      `;
    });
  }
  html += `</div>`;

  // Attached Evidence & Contract Citations
  if (data.evidence_bundle && data.evidence_bundle.length > 0) {
    html += `<div><div class="section-title">📌 Attached Evidence & Contract Citations (${data.evidence_bundle.length})</div>`;
    data.evidence_bundle.forEach(e => {
      html += `
        <div class="evidence-item">
          <div class="evidence-src">${e.category}: ${e.source_document} (Ref: ${e.reference_id})</div>
          <div>${e.citation_text}</div>
        </div>
      `;
    });
    html += `</div>`;
  }

  body.innerHTML = html;

  reviewBar.style.display = "flex";
  if (data.human_decision) {
    reviewStatus.innerHTML = `Prior Human Review: <b>${data.human_decision.action.toUpperCase()}</b> by ${data.human_decision.reviewer_id} (${new Date(data.human_decision.timestamp).toLocaleTimeString()})`;
  } else {
    reviewStatus.innerHTML = `Status: Pending Human Review Decision`;
  }
}

async function submitReview(action) {
  if (!currentInvoiceId) return;

  const notes = document.getElementById("review-notes").value;
  const statusLabel = document.getElementById("review-status-label");
  statusLabel.textContent = "Recording decision in audit trail...";

  try {
    const res = await fetch("/api/review", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        invoice_id: currentInvoiceId,
        reviewer_id: "Senior_Auditor",
        action: action,
        notes: notes || "Auditor manual sign-off",
      }),
    });
    await res.json();
    statusLabel.innerHTML = `✓ Decision recorded: <b>${action.toUpperCase()}</b> at ${new Date().toLocaleTimeString()}`;
    appendBotMsg(`Human Review Decision saved for **${currentInvoiceId}**: **${action.toUpperCase()}**.`);
  } catch (err) {
    statusLabel.textContent = "Failed to record decision: " + err.message;
  }
}

// --- ERP Tables Loading ---
async function loadERPTables() {
  // 1. POs Table
  try {
    const resPo = await fetch("/api/erp/purchase-orders");
    const pos = await resPo.json();
    const poTbody = document.querySelector("#po-table tbody");
    poTbody.innerHTML = "";
    pos.forEach(row => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><b style="color: #a5b4fc;">${row.po_number}</b></td>
        <td>${row.vendor}</td>
        <td>#${row.line_number}</td>
        <td><code>${row.sku}</code></td>
        <td>${row.description}</td>
        <td>${row.quantity}</td>
        <td>$${row.unit_price}</td>
        <td><b>$${row.line_total}</b></td>
      `;
      poTbody.appendChild(tr);
    });
  } catch (e) {
    console.error("Failed to load POs:", e);
  }

  // 2. Goods Receipts Table
  try {
    const resGr = await fetch("/api/erp/goods-receipts");
    const grs = await resGr.json();
    const grTbody = document.querySelector("#gr-table tbody");
    grTbody.innerHTML = "";
    grs.forEach(row => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><b style="color: #34d399;">${row.receipt_id}</b></td>
        <td><b style="color: #a5b4fc;">${row.po_number}</b></td>
        <td>${row.receipt_date}</td>
        <td>#${row.line_number}</td>
        <td><code>${row.sku}</code></td>
        <td>${row.quantity_received}</td>
        <td><span class="badge badge-auto_approve">${row.status}</span></td>
      `;
      grTbody.appendChild(tr);
    });
  } catch (e) {
    console.error("Failed to load Goods Receipts:", e);
  }
}

// --- Vector DB Clauses Loading & Addition ---
async function loadVectorDBClauses() {
  try {
    const res = await fetch("/api/rag/clauses");
    const clauses = await res.json();
    document.getElementById("clause-count-badge").textContent = `${clauses.length} Clauses Indexed`;

    const tbody = document.querySelector("#clauses-table tbody");
    tbody.innerHTML = "";
    clauses.forEach(c => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><code>${c.clause_id}</code></td>
        <td><b>${c.vendor}</b></td>
        <td>${c.title}</td>
        <td>${c.token_count} tokens</td>
        <td>${c.prohibits_surcharges ? '<span class="badge badge-reject">PROHIBITED</span>' : '<span class="badge" style="background: rgba(255,255,255,0.05);">ALLOWED</span>'}</td>
        <td style="max-width: 320px; font-size: 0.78rem; color: var(--text-muted);">${c.content}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (e) {
    console.error("Failed to load clauses:", e);
  }
}

async function handleAddClause(e) {
  e.preventDefault();
  const feedback = document.getElementById("clause-addition-feedback");
  feedback.textContent = "Indexing clause into Vector DB...";

  const payload = {
    clause_id: document.getElementById("c-id").value.trim(),
    vendor: document.getElementById("c-vendor").value.trim(),
    title: document.getElementById("c-title").value.trim(),
    content: document.getElementById("c-content").value.trim(),
    prohibits_surcharges: document.getElementById("c-prohibits").value === "true",
  };

  try {
    const res = await fetch("/api/rag/clauses", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await res.json();
    feedback.innerHTML = `✓ Successfully indexed clause <b>${result.clause_details.clause_id}</b> (${result.clause_details.token_count} tokens embedded)!`;
    document.getElementById("add-clause-form").reset();
    await loadVectorDBClauses();
  } catch (err) {
    feedback.innerHTML = `<span style="color: var(--danger);">Failed to add clause: ${err.message}</span>`;
  }
}

// --- Chatbot Logic ---
async function handleSendMessage() {
  const input = document.getElementById("chat-input");
  const msg = input.value.trim();
  if (!msg) return;

  appendUserMsg(msg);
  input.value = "";

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: msg,
        invoice_id: currentInvoiceId,
      }),
    });
    const data = await res.json();
    appendBotMsg(data.reply);
  } catch (err) {
    appendBotMsg("Error querying auditor assistant: " + err.message);
  }
}

function sendChip(el) {
  document.getElementById("chat-input").value = el.textContent;
  handleSendMessage();
}

function appendUserMsg(text) {
  const container = document.getElementById("chat-messages");
  const div = document.createElement("div");
  div.className = "chat-msg user";
  div.textContent = text;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function appendBotMsg(text) {
  const container = document.getElementById("chat-messages");
  const div = document.createElement("div");
  div.className = "chat-msg bot";
  div.innerHTML = text.replace(/\*\*(.*?)\*\*/g, "<b>$1</b>").replace(/\n/g, "<br>");
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}
