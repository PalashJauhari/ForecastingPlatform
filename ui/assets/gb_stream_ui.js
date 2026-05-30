/**
 * SSE clientside helpers for GaussianBlurr Dash UI.
 *
 * Server contract (POST /run/stream, POST /resume/stream):
 *   - {"type":"node",...} — LangGraph node finished (Orchestrator / RunTools / …).
 *   - {"type":"done",...} — same payload shape as POST /run (get_api_response).
 *   - {"type":"error","error":"..."} — stream aborted.
 */

window.dash_clientside = window.dash_clientside || {};
window.dash_clientside.gb_stream_ui = window.dash_clientside.gb_stream_ui || {};
window.dash_clientside.gb_stream_ui._inFlight = false;

function setComposerBusy(busy) {
  var btn = document.getElementById("btn-send");
  var input = document.getElementById("chat-input");
  var host = document.getElementById("gb-composer-host");
  if (btn) {
    btn.classList.toggle("gb-composer-busy", !!busy);
    btn.setAttribute("aria-busy", busy ? "true" : "false");
  }
  if (input) {
    input.classList.toggle("gb-composer-busy", !!busy);
    input.readOnly = !!busy;
  }
  if (host) host.classList.toggle("gb-composer-busy", !!busy);
}

function gbReadMessageText(messageText) {
  var raw = (messageText || "").trim();
  if (raw) return raw;
  var inputEl = document.getElementById("chat-input");
  if (inputEl && inputEl.value) return String(inputEl.value).trim();
  return "";
}

function truncate(s, maxLen) {
  if (!s) return "";
  if (s.length <= maxLen) return s;
  return s.slice(0, maxLen - 1) + "…";
}

function gbChatAreaScrollBottom() {
  var chatArea = document.getElementById("chat-area");
  if (chatArea) chatArea.scrollTop = chatArea.scrollHeight;
}

function gbProgressScrollBottom(progressEl) {
  if (!progressEl) return;
  progressEl.scrollTop = progressEl.scrollHeight;
}

function gbEnsureThinkingPanelHost() {
  var panel = document.getElementById("gb-thinking-panel");
  if (panel) return panel;
  var composer = document.getElementById("gb-composer-host");
  if (!composer || !composer.parentElement) return null;
  panel = document.createElement("div");
  panel.id = "gb-thinking-panel";
  panel.className = "gb-thinking-panel gb-thinking-panel--hidden";
  composer.parentElement.insertBefore(panel, composer);
  return panel;
}

/** Re-query progress container; rebuild card if Dash/React cleared the panel mid-stream. */
function gbResolveProgressEl(fallback) {
  var el = document.querySelector("#gb-thinking-panel .gb-inline-progress");
  if (el && el.isConnected) return el;
  if (fallback && fallback.isConnected) return fallback;
  if (window.dash_clientside.gb_stream_ui._inFlight) {
    return gbRebuildThinkingCard();
  }
  return fallback || null;
}

function gbBuildThinkingCardDOM() {
  var card = document.createElement("div");
  card.className = "gb-thinking-card";

  var bar = document.createElement("button");
  bar.type = "button";
  bar.className = "gb-thinking-bar";
  bar.setAttribute("aria-expanded", "true");

  var spinner = document.createElement("span");
  spinner.className = "gb-thinking-spinner";
  spinner.setAttribute("aria-hidden", "true");

  var label = document.createElement("span");
  label.className = "gb-thinking-label";
  label.textContent = "Thinking";

  var summary = document.createElement("span");
  summary.className = "gb-thinking-summary";
  summary.textContent = "Connecting…";

  var chevron = document.createElement("span");
  chevron.className = "gb-thinking-chevron";
  chevron.textContent = "▾";
  chevron.setAttribute("aria-hidden", "true");

  bar.appendChild(spinner);
  bar.appendChild(label);
  bar.appendChild(summary);
  bar.appendChild(chevron);

  bar.addEventListener("click", function () {
    var collapsed = card.classList.toggle("gb-thinking-card--collapsed");
    bar.setAttribute("aria-expanded", collapsed ? "false" : "true");
  });

  var body = document.createElement("div");
  body.className = "gb-thinking-body";

  var progressEl = document.createElement("div");
  progressEl.className = "gb-inline-progress";

  body.appendChild(progressEl);
  card.appendChild(bar);
  card.appendChild(body);

  return { card: card, progressEl: progressEl };
}

function gbRebuildThinkingCard() {
  var panel = gbEnsureThinkingPanelHost();
  if (!panel) return null;
  panel.className = "gb-thinking-panel";
  panel.innerHTML = "";
  var built = gbBuildThinkingCardDOM();
  panel.appendChild(built.card);
  return built.progressEl;
}

function gbUpdateThinkingSummary(boldText, restText) {
  var summary = document.querySelector(".gb-thinking-summary");
  if (!summary) return;
  var text = (boldText || "") + (restText || "");
  summary.textContent = truncate(text.trim(), 120) || "Working…";
}

function gbAppendProgressLine(progressEl, boldText, restText) {
  progressEl = gbResolveProgressEl(progressEl);
  if (!progressEl) return;
  var row = document.createElement("div");
  row.className = "gb-progress-line";
  if (boldText) {
    var s = document.createElement("strong");
    s.textContent = boldText;
    row.appendChild(s);
  }
  if (restText) row.appendChild(document.createTextNode(restText));
  progressEl.appendChild(row);
  gbProgressScrollBottom(progressEl);
  gbUpdateThinkingSummary(boldText, restText);
}

function gbCreateInlineUserBubble(chatArea, text) {
  if (!chatArea) return;
  var row = document.createElement("div");
  row.className = "gb-live-turn";
  row.style.display = "flex";
  row.style.justifyContent = "flex-end";
  row.style.marginBottom = "10px";

  var bubble = document.createElement("div");
  bubble.className = "gb-live-user-bubble";
  bubble.textContent = text;
  row.appendChild(bubble);
  chatArea.appendChild(row);
  gbChatAreaScrollBottom();
}

function gbOpenThinkingPanel(messageText) {
  var chatArea = document.getElementById("chat-area");
  var panel = gbEnsureThinkingPanelHost();
  if (!chatArea || !panel) return null;

  gbClearStreamUI();

  panel = gbEnsureThinkingPanelHost();
  if (!panel) return null;

  gbCreateInlineUserBubble(chatArea, messageText);

  panel.className = "gb-thinking-panel";
  panel.innerHTML = "";

  var built = gbBuildThinkingCardDOM();
  panel.appendChild(built.card);

  return built.progressEl;
}

function gbClearStreamUI() {
  var liveRows = document.querySelectorAll(".gb-live-turn");
  for (var i = 0; i < liveRows.length; i++) liveRows[i].remove();

  var panel = document.getElementById("gb-thinking-panel");
  if (panel) {
    panel.innerHTML = "";
    panel.className = "gb-thinking-panel gb-thinking-panel--hidden";
  }
}

function gbAppendProgressSub(progressEl, boldText, restText) {
  progressEl = gbResolveProgressEl(progressEl);
  if (!progressEl) return;
  var row = document.createElement("div");
  row.className = "gb-progress-line gb-progress-sub";
  if (boldText) {
    var s = document.createElement("strong");
    s.textContent = boldText;
    row.appendChild(s);
  }
  if (restText) {
    row.appendChild(document.createTextNode(restText ? " " + restText : ""));
  }
  progressEl.appendChild(row);
  gbProgressScrollBottom(progressEl);
  if (boldText) gbUpdateThinkingSummary(boldText, restText);
}

/** Node ids not shown in the progress panel (bookkeeping; server may already omit them). */
var GB_PROGRESS_SKIP_NODES = {
  ProfileSavedData_PostTools: true,
};

function gbAppendOrchestratorProgress(progressEl, ev) {
  var tc = Array.isArray(ev.tool_calls) ? ev.tool_calls : [];
  gbAppendProgressLine(progressEl, "Orchestrator", tc.length ? " — planned tools:" : "");
  for (var j = 0; j < tc.length; j++) {
    var t = tc[j] || {};
    var n = t.name || "?";
    gbAppendProgressSub(progressEl, n, "");
  }
  if (!tc.length && ev.had_tool_calls === false) {
    gbAppendProgressSub(progressEl, "", "(no tools)");
  }
  gbProgressScrollBottom(progressEl);
}

function gbAppendTodosUnderNode(progressEl, ev) {
  progressEl = gbResolveProgressEl(progressEl);
  if (!progressEl) return;
  var todos = Array.isArray(ev.todos) ? ev.todos : [];
  if (!todos.length) return;
  var ul = document.createElement("ul");
  ul.className = "gb-progress-todos";
  for (var i = 0; i < todos.length; i++) {
    var td = todos[i];
    var li = document.createElement("li");
    var st = (td && td.status) || "";
    var idp = td && td.id ? String(td.id) + " — " : "";
    var ct = truncate(idp + ((td && td.content) || ""), 500);
    li.textContent = (st ? "[" + String(st) + "] " : "") + ct;
    ul.appendChild(li);
  }
  progressEl.appendChild(ul);
  gbProgressScrollBottom(progressEl);
}

function gbAppendRunToolsProgress(progressEl, ev) {
  gbAppendProgressLine(progressEl, "RunTools", "");
  gbAppendTodosUnderNode(progressEl, ev);
  var tr = Array.isArray(ev.tool_results) ? ev.tool_results : [];
  for (var k = 0; k < tr.length; k++) {
    var x = tr[k];
    var name = (x && x.name) || "tool";
    gbAppendProgressSub(progressEl, name, "");
  }
  gbProgressScrollBottom(progressEl);
}

/** Turn one SSE envelope into `{ bold, rest }` for monospace progress rows (non-special nodes). */
function gbProgressBoldRest(ev) {
  if (!ev || typeof ev !== "object") return { bold: "event", rest: " — " + String(ev) };
  var node = ev.node || "";

  if (ev.type === "error") {
    return { bold: "error", rest: ev.error ? ": " + ev.error : "" };
  }
  if (ev.type === "done") {
    var doneLbl = ev.interrupted ? "done (interrupt)" : "done";
    var pv = truncate(ev.summary || "", 140);
    return { bold: doneLbl, rest: pv ? " — " + pv : "" };
  }

  var boldName = node || "node";

  if (node === "__interrupt__") {
    var ip = ev.interrupt_preview || "";
    return { bold: "__interrupt__", rest: ip ? " — " + truncate(ip, 200) : " — clarification" };
  }
  if (node === "ProfileSavedData") {
    var pe = ev.profile_entries != null ? " · " + ev.profile_entries + " files profiled" : "";
    return { bold: boldName, rest: pe };
  }
  if (node === "SummariseConversationalSummary" && ev.summary_preview) {
    return { bold: boldName, rest: " — " + truncate(ev.summary_preview, 180) };
  }

  var label = ev.label || "";
  return { bold: boldName, rest: label ? " — " + truncate(label, 120) : "" };
}

function gbAppendPlannerProgress(progressEl, ev) {
  gbAppendProgressLine(progressEl, "Planner", "");
  gbAppendTodosUnderNode(progressEl, ev);
}

function gbAppendStreamNode(progressEl, payload) {
  if (!payload || payload.type !== "node") return;
  var node = payload.node || "";
  if (GB_PROGRESS_SKIP_NODES[node]) return;
  if (node === "Orchestrator") return gbAppendOrchestratorProgress(progressEl, payload);
  if (node === "RunTools") return gbAppendRunToolsProgress(progressEl, payload);
  if (node === "Planner") return gbAppendPlannerProgress(progressEl, payload);
  var pr = gbProgressBoldRest(payload);
  gbAppendProgressLine(progressEl, pr.bold, pr.rest);
}

/**
 * Consume fetch Response SSE body until closed; pushes rows into progressEl.
 * Returns parsed done envelope or null.
 */
async function gbParseSSEStream(response, progressEl) {
  var reader = response.body.getReader();
  var decoder = new TextDecoder();
  var buffer = "";
  var donePayload = null;
  while (true) {
    var chunk = await reader.read();
    if (chunk.done) break;
    buffer += decoder.decode(chunk.value, { stream: true });
    var sep;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      var frame = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      var lines = frame.split("\n");
      for (var i = 0; i < lines.length; i++) {
        var line = lines[i];
        if (line.endsWith("\r")) line = line.slice(0, -1);
        if (!line.startsWith("data: ")) continue;
        var raw = line.slice(6);
        var payload;
        try {
          payload = JSON.parse(raw);
        } catch (e) {
          gbAppendProgressLine(progressEl, "sse", " — could not parse JSON frame");
          continue;
        }
        if (payload.type === "error") {
          var er = gbProgressBoldRest(payload);
          gbAppendProgressLine(progressEl, er.bold, er.rest);
          throw new Error(payload.error || "Stream error");
        }
        if (payload.type === "done") {
          donePayload = payload;
          gbAppendProgressLine(progressEl, "done", "");
          continue;
        }
        if (payload.type === "node") {
          gbAppendStreamNode(progressEl, payload);
          continue;
        }
        var pr = gbProgressBoldRest(payload);
        gbAppendProgressLine(progressEl, pr.bold, pr.rest);
      }
    }
  }
  return donePayload;
}

function gbResetSubmitLock() {
  window.dash_clientside.gb_stream_ui._inFlight = false;
  window.gbStreamSubmitBusy = false;
  setComposerBusy(false);
}

gbResetSubmitLock();

window.dash_clientside.gb_stream_ui.clear_stream_progress = function (_uploadGen) {
  gbClearStreamUI();
  gbResetSubmitLock();
  return "";
};

/**
 * Streams /run/stream or /resume/stream; mutates Dash ui-store chat JSON.
 */
window.dash_clientside.gb_stream_ui.submit_message_stream = async function (
  n_clicks,
  n_submit,
  messageText,
  store,
  apiBase
) {
  var nu = window.dash_clientside.no_update;

  var raw = gbReadMessageText(messageText);
  if (!raw) return [nu, nu];

  if (window.dash_clientside.gb_stream_ui._inFlight || window.gbStreamSubmitBusy || !store) {
    return [nu, nu];
  }

  window.dash_clientside.gb_stream_ui._inFlight = true;
  window.gbStreamSubmitBusy = true;
  setComposerBusy(true);

  try {
    var chat = {};
    Object.assign(chat, store);
    chat.agent_thinking = false;

    var messages = Array.isArray(chat.messages) ? chat.messages.slice() : [];
    chat.messages = messages;

    var awaitingResume = !!chat.awaiting_resume;
    var restorePendingQuestion = awaitingResume ? String(chat.pending_question || "") : "";
    chat.messages.push({ role: "user", content: raw });

    var base = (apiBase || "http://127.0.0.1:8000").replace(/\/$/, "");
    var sessionId = chat.session_id;
    var progressEl = gbOpenThinkingPanel(raw);
    if (!progressEl) {
      throw new Error("UI thinking panel is unavailable — hard-refresh the page and try again.");
    }
    gbAppendProgressLine(progressEl, "…", " — connecting");

    var endpoint = awaitingResume ? "/resume/stream" : "/run/stream";
    var payload = awaitingResume
      ? { session_id: sessionId, resume_value: raw }
      : { session_id: sessionId, query: raw };

    try {
      var r = await fetch(base + endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify(payload),
      });

      if (!r.ok) {
        var bodyText = await r.text();
        var errMsg = bodyText.slice(0, 800);
        try {
          var ej = JSON.parse(bodyText);
          if (ej.detail !== undefined)
            errMsg = typeof ej.detail === "string" ? ej.detail : JSON.stringify(ej.detail);
          else if (ej.error !== undefined) errMsg = String(ej.error);
        } catch (_e2) {}
        throw new Error(errMsg || r.statusText);
      }

      var donePayload = await gbParseSSEStream(r, progressEl);

      if (!donePayload) {
        gbClearStreamUI();
        chat.messages.push({
          role: "assistant",
          content: "Run finished without a final **done** event — check FastAPI logs or graph wiring.",
        });
        return [chat, ""];
      }

      if (donePayload.interrupted) {
        var q = donePayload.question || "Please clarify.";
        chat.awaiting_resume = true;
        chat.pending_question = q;
        gbClearStreamUI();
        chat.messages.push({
          role: "assistant",
          content: "I need a bit more information before I proceed:\n\n**" + q + "**",
        });
        return [chat, ""];
      }

      chat.awaiting_resume = false;
      chat.pending_question = "";

      var summary = donePayload.summary || "Done.";
      var images = donePayload.images || [];
      var amsg = { role: "assistant", content: summary, output_images: images };
      gbClearStreamUI();
      chat.messages.push(amsg);
      return [chat, ""];
    } catch (e) {
      var err = e && e.message ? String(e.message) : String(e);
      if (awaitingResume && restorePendingQuestion) {
        chat.awaiting_resume = true;
        chat.pending_question = restorePendingQuestion;
      }
      gbAppendProgressLine(progressEl, "error", " — " + truncate(err, 400));
      gbClearStreamUI();
      chat.messages.push({
        role: "assistant",
        content: "Something went wrong: " + truncate(err, 800),
      });
      return [chat, ""];
    }
  } catch (outerErr) {
    var outerMsg = outerErr && outerErr.message ? String(outerErr.message) : String(outerErr);
    var chatOuter = {};
    Object.assign(chatOuter, store || {});
    var outerMsgs = Array.isArray(chatOuter.messages) ? chatOuter.messages.slice() : [];
    chatOuter.messages = outerMsgs;
    gbClearStreamUI();
    chatOuter.messages.push({
      role: "assistant",
      content: "Something went wrong: " + truncate(outerMsg, 800),
    });
    return [chatOuter, ""];
  } finally {
    gbResetSubmitLock();
  }
};
