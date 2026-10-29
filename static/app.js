// Frontend Controller for Session-Based Multi-Agent Procurement Auditor

let activeSessionId = null;
let sessionInvoices = [];
let sessionResults = {};
let currentInvoiceId = null;

// Selected files staging
const stagedFiles = {
  invoices: [],
  pos: null,
  receipts: null,
  contracts: null,
};

document.addEventListener("DOMContentLoaded", () => {
  setupChatbot();
  setupDragAndDrop();
  checkActiveSession();
});

// --- Screen Switching ---
function showScreen(screenId) {
  document.querySelectorAll(".screen").forEach(s => s.classList.remove("active"));
  const target = document.getElementById(screenId);
  if (target) target.classList.add("active");
}

function switchView(viewName) {
  document.querySelectorAll(".nav-tab-btn").forEach(btn => btn.classList.remove("active"));
  document.querySelectorAll(".view-panel").forEach(panel => panel.classList.remove("active"));

  if (viewName === "audit") {
    document.querySelectorAll(".nav-tab-btn")[0].classList.add("active");
    document.getElementById("view-audit").classList.add("active");
  } else if (viewName === "erp") {
    document.querySelectorAll(".nav-tab-btn")[1].classList.add("active");
    document.getElementById("view-erp").classList.add("active");
    loadSessionERPTables();
  } else if (viewName === "vectordb") {
    document.querySelectorAll(".nav-tab-btn")[2].classList.add("active");
    document.getElementById("view-vectordb").classList.add("active");
    loadSessionVectorDB();
  }
}

// --- Check Active Session on Page Load ---
async function checkActiveSession() {
  try {
    const res = await fetch("/api/session/current");
    const data = await res.json();
    if (data.session_id && data.status === "completed") {
      activeSessionId = data.session_id;
      // Load session results
      await restoreSessionData(activeSessionId);
    } else {
      // Start with clean upload portal (zero early tables/files)
      resetToUploadPortalUI();
    }
  } catch (e) {
    resetToUploadPortalUI();
  }
}

function resetToUploadPortalUI() {
  activeSessionId = null;
  sessionInvoices = [];
  sessionResults = {};
  currentInvoiceId = null;
  stagedFiles.invoices = [];
  stagedFiles.pos = null;
  stagedFiles.receipts = null;
  stagedFiles.contracts = null;

  document.getElementById("session-badge").style.display = "none";
  document.getElementById("main-nav-tabs").style.display = "none";
  document.getElementById("btn-new-session").style.display = "none";

  // Reset status text
  document.getElementById("status-invoices").textContent = "No files selected";
  document.getElementById("status-pos").textContent = "No file selected";
  document.getElementById("status-receipts").textContent = "No file selected";
  document.getElementById("status-contracts").textContent = "No file selected";

  document.querySelectorAll(".upload-card").forEach(c => c.classList.remove("has-file"));

  showScreen("screen-portal");
}

async function resetToUploadPortal() {
  try {
    await fetch("/api/session/reset", { method: "POST" });
  } catch (e) {}
  resetToUploadPortalUI();
}

// --- File Selection Handlers ---
function handleFileSelected(type, input) {
  const card = document.getElementById(`card-${type}`);
  const status = document.getElementById(`status-${type}`);

  if (type === "invoices") {
    stagedFiles.invoices = Array.from(input.files);
    if (stagedFiles.invoices.length > 0) {
      status.textContent = `✓ Selected ${stagedFiles.invoices.length} invoice(s). Uploading & starting audit workflow...`;
      card.classList.add("has-file");
      // Immediate auto-launch so nothing is stalled
      setTimeout(() => {
        startWorkflowExecution();
      }, 400);
    } else {
      status.textContent = "No files selected";
      card.classList.remove("has-file");
    }
  } else {
    const file = input.files[0];
    stagedFiles[type] = file || null;
    if (file) {
      status.textContent = `✓ Loaded: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
      card.classList.add("has-file");
    } else {
      status.textContent = "No file selected";
      card.classList.remove("has-file");
    }
  }
}

function setupDragAndDrop() {
  ["invoices", "pos", "receipts", "contracts"].forEach(type => {
    const card = document.getElementById(`card-${type}`);
    const input = document.getElementById(`input-${type}`);

    card.addEventListener("dragover", (e) => {
      e.preventDefault();
      card.classList.add("highlight");
    });
    card.addEventListener("dragleave", (e) => {
      e.preventDefault();
      card.classList.remove("highlight");
    });
    card.addEventListener("drop", (e) => {
      e.preventDefault();
      card.classList.remove("highlight");
      if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        input.files = e.dataTransfer.files;
        handleFileSelected(type, input);
      }
    });
  });
}

// --- Launch Workflow with User Uploaded Documents ---
async function startWorkflowExecution() {
  if (!stagedFiles.invoices || stagedFiles.invoices.length === 0) {
    alert("Please select or drop at least one invoice PDF document to audit.");
    return;
  }

  const formData = new FormData();
  for (const inv of stagedFiles.invoices) {
    formData.append("invoices", inv);
  }
  if (stagedFiles.pos) formData.append("purchase_orders", stagedFiles.pos);
  if (stagedFiles.receipts) formData.append("goods_receipts", stagedFiles.receipts);
  if (stagedFiles.contracts) formData.append("contracts", stagedFiles.contracts);

  executeWorkflowCall(formData);
}

// --- Launch Quick Test with Benchmark Sample Pack ---
async function launchWithSamplePack() {
  const formData = new FormData();
  formData.append("use_sample_pack", "true");
  executeWorkflowCall(formData);
}

// --- Stepper Execution Animation and API Call ---
async function executeWorkflowCall(formData) {
  showScreen("screen-stepper");
  resetStepperUI();

  // Animate initial step
  updateStepRow(1, "in_progress", "Chunking contract clauses & tokenizing for Vector DB in parallel...");

  try {
    // Artificial smooth step transition timer for high visual quality
    setTimeout(() => {
      updateStepRow(1, "completed", "Indexed clauses into session Vector RAG.");
      updateStepRow(2, "in_progress", "Extracting uploaded invoice PDFs in parallel threads...");
    }, 400);

    setTimeout(() => {
      updateStepRow(2, "completed", "Parallel coordinate extraction and line parsing complete.");
      updateStepRow(3, "in_progress", "Executing deterministic 3-way matching & contract RAG audit...");
    }, 900);

    const res = await fetch("/api/session/start", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server returned ${res.status}`);
    }

    const data = await res.json();
    activeSessionId = data.session_id;
    sessionInvoices = data.invoices;
    sessionResults = data.results;

    document.getElementById("stepper-session-id").textContent = activeSessionId;
    updateStepRow(3, "completed", data.steps[2].details || "3-Way reconciliation completed.");
    updateStepRow(4, "completed", data.steps[3].details || "Ledger composite history verified and saved.");
    
    document.getElementById("stepper-overall-status").className = "badge badge-auto_approve";
    document.getElementById("stepper-overall-status").textContent = "COMPLETED";

    const btnCockpit = document.getElementById("btn-view-cockpit");
    btnCockpit.style.display = "inline-flex";

    // Auto-transition to Cockpit after brief pause
    setTimeout(() => {
      transitionToCockpit();
    }, 800);

  } catch (err) {
    document.getElementById("stepper-overall-status").className = "badge badge-reject";
    document.getElementById("stepper-overall-status").textContent = "ERROR";
    updateStepRow(2, "completed", "Extraction failed.");
    updateStepRow(3, "completed", "Error: " + err.message);
    alert("Workflow failed: " + err.message);
  }
}

function resetStepperUI() {
  document.getElementById("stepper-session-id").textContent = "Creating session folder...";
  document.getElementById("stepper-overall-status").className = "badge badge-needs_review";
  document.getElementById("stepper-overall-status").textContent = "IN PROGRESS";
  document.getElementById("btn-view-cockpit").style.display = "none";

  for (let i = 1; i <= 4; i++) {
    const row = document.getElementById(`step-row-${i}`);
    row.className = "step-item";
    document.getElementById(`step-badge-${i}`).textContent = "PENDING";
    document.getElementById(`step-badge-${i}`).style.background = "rgba(255,255,255,0.06)";
  }
}

function updateStepRow(stepNum, status, details) {
  const row = document.getElementById(`step-row-${stepNum}`);
  const badge = document.getElementById(`step-badge-${stepNum}`);
  const desc = document.getElementById(`step-desc-${stepNum}`);

  row.className = `step-item ${status}`;
  badge.textContent = status.toUpperCase().replace("_", " ");
  if (status === "in_progress") {
    badge.className = "badge badge-needs_review";
  } else if (status === "completed") {
    badge.className = "badge badge-auto_approve";
  }
  if (details) desc.textContent = details;
}

// --- Transition to Active Cockpit ---
function transitionToCockpit() {
  showScreen("screen-cockpit");
  switchView("audit");

  // Show session headers
  const badge = document.getElementById("session-badge");
  badge.style.display = "inline-block";
  badge.textContent = `Session: ${activeSessionId}`;

  document.getElementById("main-nav-tabs").style.display = "flex";
  document.getElementById("btn-new-session").style.display = "inline-flex";

  renderSessionDocumentQueue();
  if (sessionInvoices.length > 0) {
    selectInvoice(sessionInvoices[0].id, 0);
  }
}

// --- Render Session Documents Queue ---
function renderSessionDocumentQueue() {
  const listEl = document.getElementById("sample-list");
  const queueBadge = document.getElementById("doc-queue-badge");

  queueBadge.textContent = `${sessionInvoices.length} Documents`;
  listEl.innerHTML = "";

  sessionInvoices.forEach((inv, idx) => {
    const item = document.createElement("div");
    item.className = "sample-item" + (idx === 0 ? " active" : "");
    item.onclick = () => selectInvoice(inv.id, idx);

    const badgeClass = `badge-${inv.decision || 'needs_review'}`;
    item.innerHTML = `
      <div class="sample-title">
        <span>${inv.id}</span>
        <span class="badge ${badgeClass}">${(inv.decision || 'review').replace("_", " ")}</span>
      </div>
      <div class="sample-desc">${inv.vendor || 'Unknown Supplier'} • PO: ${inv.po_number || 'N/A'}</div>
    `;
    listEl.appendChild(item);
  });
}

// --- Select and Inspect Invoice in Cockpit ---
function selectInvoice(invoiceId, idx) {
  currentInvoiceId = invoiceId;

  document.querySelectorAll(".sample-item").forEach((item, i) => {
    if (i === idx) item.classList.add("active");
    else item.classList.remove("active");
  });

  const auditData = sessionResults[invoiceId];
  if (auditData) {
    renderAuditResult(auditData);
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
        <span class="stat-value" style="color: ${parseFloat(data.financial_exposure) > 0 ? 'var(--danger)' : 'var(--text-main)'};">
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
  if (!currentInvoiceId || !activeSessionId) return;

  const notes = document.getElementById("review-notes").value;
  const statusLabel = document.getElementById("review-status-label");
  statusLabel.textContent = "Recording decision in session audit trail...";

  try {
    const res = await fetch(`/api/review?session_id=${activeSessionId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        invoice_id: currentInvoiceId,
        reviewer_id: "Senior_Auditor",
        action: action,
        notes: notes || "Auditor manual sign-off",
      }),
    });
    const resData = await res.json();
    statusLabel.innerHTML = `✓ Decision recorded: <b>${action.toUpperCase()}</b> at ${new Date().toLocaleTimeString()}`;
    if (sessionResults[currentInvoiceId]) {
      sessionResults[currentInvoiceId].human_decision = resData.decision;
    }
    appendBotMsg(`Human Review Decision saved for **${currentInvoiceId}**: **${action.toUpperCase()}**.`);
  } catch (err) {
    statusLabel.textContent = "Failed to record decision: " + err.message;
  }
}

// --- Session ERP Tables Loading ---
async function loadSessionERPTables() {
  const poTbody = document.querySelector("#po-table tbody");
  const grTbody = document.querySelector("#gr-table tbody");

  // 1. POs Table
  try {
    const resPo = await fetch(`/api/erp/purchase-orders?session_id=${activeSessionId || ''}`);
    const pos = await resPo.json();
    if (!pos || pos.length === 0) {
      poTbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--text-muted); padding: 30px;">No purchase orders in this session.</td></tr>`;
    } else {
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
    }
  } catch (e) {
    poTbody.innerHTML = `<tr><td colspan="8" style="text-align: center; color: var(--danger);">Failed to load POs: ${e.message}</td></tr>`;
  }

  // 2. Goods Receipts Table
  try {
    const resGr = await fetch(`/api/erp/goods-receipts?session_id=${activeSessionId || ''}`);
    const grs = await resGr.json();
    if (!grs || grs.length === 0) {
      grTbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-muted); padding: 30px;">No warehouse goods receipts in this session.</td></tr>`;
    } else {
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
    }
  } catch (e) {
    grTbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--danger);">Failed to load Goods Receipts: ${e.message}</td></tr>`;
  }
}

// --- Session Vector DB Loading & Addition ---
async function loadSessionVectorDB() {
  try {
    const res = await fetch(`/api/rag/clauses?session_id=${activeSessionId || ''}`);
    const clauses = await res.json();
    document.getElementById("clause-count-badge").textContent = `${clauses.length} Clauses Indexed`;

    const tbody = document.querySelector("#clauses-table tbody");
    if (!clauses || clauses.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted); padding: 30px;">No contract clauses indexed in this session.</td></tr>`;
      return;
    }

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
  feedback.textContent = "Indexing clause into session Vector DB...";

  const payload = {
    clause_id: document.getElementById("c-id").value.trim(),
    vendor: document.getElementById("c-vendor").value.trim(),
    title: document.getElementById("c-title").value.trim(),
    content: document.getElementById("c-content").value.trim(),
    prohibits_surcharges: document.getElementById("c-prohibits").value === "true",
  };

  try {
    const res = await fetch(`/api/rag/clauses?session_id=${activeSessionId || ''}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await res.json();
    feedback.innerHTML = `✓ Successfully indexed clause <b>${result.clause_details.clause_id}</b> (${result.clause_details.token_count} tokens embedded)!`;
    document.getElementById("add-clause-form").reset();
    await loadSessionVectorDB();
  } catch (err) {
    feedback.innerHTML = `<span style="color: var(--danger);">Failed to add clause: ${err.message}</span>`;
  }
}

// --- Chatbot Logic ---
function setupChatbot() {
  document.getElementById("btn-send-chat").addEventListener("click", handleSendMessage);
  document.getElementById("chat-input").addEventListener("keypress", (e) => {
    if (e.key === "Enter") handleSendMessage();
  });
}

async function handleSendMessage() {
  const input = document.getElementById("chat-input");
  const msg = input.value.trim();
  if (!msg) return;

  appendUserMsg(msg);
  input.value = "";

  try {
    const res = await fetch(`/api/chat?session_id=${activeSessionId || ''}`, {
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
