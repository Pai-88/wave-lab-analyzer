// Song identification relay for Wave Lab (Vercel serverless function, Node runtime).
//
// The browser records a few seconds of microphone audio and POSTs the raw clip here. This function adds the AudD token,
// which lives only in the AUDD_TOKEN environment variable on the server, forwards the clip to api.audd.io and returns
// AudD's JSON unchanged. Visitors never see or paste a token. Pasting a token in the page still works and bypasses this.
//
// Limits: only the page's own origins may call it, 1.5 MB per clip, 6 identifications per 10 minutes per IP (per warm
// instance, best effort). Keep the AudD account on its free tier or set a spending cap there.

const ALLOWED_ORIGINS = new Set([
  "https://paingheinhtet.com",
  "https://pai-88.github.io",
  "http://localhost:8000",
  "http://127.0.0.1:8000",
]);
const MAX_BYTES = 1.5 * 1024 * 1024;
const WINDOW_MS = 10 * 60 * 1000;
const PER_WINDOW = 6;
const hits = new Map();

function fail(res, code, message) {
  res.status(code).json({ status: "error", error: { error_code: code, error_message: message } });
}

module.exports = async (req, res) => {
  const origin = req.headers.origin || "";
  if (ALLOWED_ORIGINS.has(origin)) {
    res.setHeader("Access-Control-Allow-Origin", origin);
    res.setHeader("Vary", "Origin");
  }
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  res.setHeader("Access-Control-Max-Age", "86400");
  if (req.method === "OPTIONS") return res.status(204).end();
  if (req.method !== "POST") return fail(res, 405, "POST a recorded clip");
  if (!ALLOWED_ORIGINS.has(origin)) return fail(res, 403, "this relay only serves the Wave Lab page");

  const token = process.env.AUDD_TOKEN;
  if (!token) return fail(res, 503, "the relay has no AudD token yet; paste your own in the page");

  const ip = (req.headers["x-forwarded-for"] || "").split(",")[0].trim() || (req.socket && req.socket.remoteAddress) || "unknown";
  const now = Date.now();
  const recent = (hits.get(ip) || []).filter((t) => now - t < WINDOW_MS);
  if (recent.length >= PER_WINDOW) return fail(res, 429, "too many identifications from this address; try again in a few minutes");
  recent.push(now);
  hits.set(ip, recent);
  if (hits.size > 5000) hits.clear();

  const chunks = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > MAX_BYTES) return fail(res, 413, "clip too large");
    chunks.push(chunk);
  }
  if (size < 1000) return fail(res, 400, "clip too short");

  const clip = new Blob(chunks, { type: req.headers["content-type"] || "audio/webm" });
  const form = new FormData();
  form.append("api_token", token);
  form.append("file", clip, "clip.webm");
  form.append("return", "spotify");
  try {
    const upstream = await fetch("https://api.audd.io/", { method: "POST", body: form });
    const text = await upstream.text();
    res.setHeader("Content-Type", "application/json");
    return res.status(200).send(text);
  } catch (err) {
    return fail(res, 502, "AudD did not answer: " + String(err && err.message || err).slice(0, 120));
  }
};

module.exports.config = { api: { bodyParser: false } };
