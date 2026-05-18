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

function truncate(s, maxLen) {
  if (!s) return "";
  if (s.length <= maxLen) return s;
  return s.slice(0, maxLen - 1) + "…";
}

function gbAppendProgressLine(progressEl, boldText, restText) {
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
  progressEl.scrollTop = progressEl.scrollHeight;
}

function gbAppendProgressSub(progressEl, boldText, restText) {
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
  progressEl.scrollTop = progressEl.scrollHeight;
}

/** Node ids not shown in the progress panel (bookkeeping; server may already omit them). */
var GB_PROGRESS_SKIP_NODES = {
  MergePrep: true,
  MergeTools: true,
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
  progressEl.scrollTop = progressEl.scrollHeight;
}

function gbAppendTodosUnderNode(progressEl, ev) {
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
  progressEl.scrollTop = progressEl.scrollHeight;
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
  if (node === "SelectOrchestratorSkills") {
    var cnt = ev.active_skill_count != null ? " (" + ev.active_skill_count + ")" : "";
    var sk =
      Array.isArray(ev.active_skills) && ev.active_skills.length ? " — " + ev.active_skills.join(", ") : "";
    return { bold: boldName, rest: cnt + truncate(sk, 200) };
  }
  if (node === "SelectPlannerSkills") {
    var pc =
      ev.active_planner_skill_count != null ? " (" + ev.active_planner_skill_count + ")" : "";
    var psk =
      Array.isArray(ev.active_planner_skills) && ev.active_planner_skills.length
        ? " — " + ev.active_planner_skills.join(", ")
        : "";
    return { bold: boldName, rest: pc + truncate(psk, 200) };
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

window.dash_clientside.gb_stream_ui.clear_stream_progress = function (_uploadGen) {
  var el = document.getElementById("gb-stream-progress");
  if (el) el.innerHTML = "";
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

  var raw = (messageText || "").trim();
  if (!raw) return [nu, nu];

  // Mutex until this handler resolves — Dash only commits ui-store afterward.
  if (window.gbStreamSubmitBusy || !store) return [nu, nu];
  window.gbStreamSubmitBusy = true;

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
    var progressEl = document.getElementById("gb-stream-progress");
    if (progressEl) progressEl.innerHTML = "";
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
      chat.messages.push(amsg);
      return [chat, ""];
    } catch (e) {
      var err = e && e.message ? String(e.message) : String(e);
      if (awaitingResume && restorePendingQuestion) {
        chat.awaiting_resume = true;
        chat.pending_question = restorePendingQuestion;
      }
      gbAppendProgressLine(progressEl, "error", " — " + truncate(err, 400));
      chat.messages.push({
        role: "assistant",
        content: "Something went wrong: " + truncate(err, 800),
      });
      return [chat, ""];
    }
  } finally {
    window.gbStreamSubmitBusy = false;
  }
};
