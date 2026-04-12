import * as THREE from "https://unpkg.com/three@0.176.0/build/three.module.js";

const canvas = document.getElementById("bg");
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setSize(innerWidth, innerHeight);
camera.position.z = 6;

const geometry = new THREE.IcosahedronGeometry(2.2, 1);
const material = new THREE.MeshStandardMaterial({ color: 0x7c3aed, metalness: 0.5, roughness: 0.2, wireframe: true });
const mesh = new THREE.Mesh(geometry, material);
scene.add(mesh);

const light = new THREE.PointLight(0x06b6d4, 2.6);
light.position.set(4, 4, 6);
scene.add(light);

function animate() {
  requestAnimationFrame(animate);
  mesh.rotation.x += 0.003;
  mesh.rotation.y += 0.004;
  renderer.render(scene, camera);
}
animate();

addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

const form = document.getElementById("predict-form");
const result = document.getElementById("result");
const submitButton = document.getElementById("submit");
const apiInput = document.getElementById("api-url");
const textModelsInput = document.getElementById("text-models");
const audioModelsInput = document.getElementById("audio-models");
const realtimeModeInput = document.getElementById("realtime-mode");
const progressText = document.getElementById("progress-text");
const connectionStatus = document.getElementById("connection-status");

const apiUrlFromQuery = new URLSearchParams(location.search).get("apiUrl");
const apiUrlFromGlobalConfig = globalThis.APP_CONFIG?.apiUrl;
const DEFAULT_API_URL = apiUrlFromQuery || apiUrlFromGlobalConfig || "http://127.0.0.1:8000/predict";
// 120 attempts × 1 second = 2 minutes, aligned with backend default timeout.
const MAX_POLL_ATTEMPTS = 120;
const storedApiUrl = localStorage.getItem("apiUrl");
apiInput.value = storedApiUrl || apiInput.value || DEFAULT_API_URL;

function normalizeApiUrl(rawUrl) {
  const trimmed = rawUrl.trim();
  if (!trimmed) {
    throw new Error("Backend URL is required.");
  }

  const url = new URL(trimmed);
  if (!url.pathname || url.pathname === "/") {
    url.pathname = "/predict";
  } else if (!url.pathname.endsWith("/predict")) {
    url.pathname = `${url.pathname.replace(/\/$/, "")}/predict`;
  }
  return url.toString();
}

function apiRootFromPredictUrl(predictUrl) {
  const url = new URL(predictUrl);
  const pathname = url.pathname.endsWith("/predict")
    ? url.pathname.slice(0, -"/predict".length)
    : url.pathname;
  url.pathname = pathname || "/";
  return url.toString().replace(/\/$/, "");
}

function setConnectionStatus(type, message) {
  connectionStatus.className = `connection ${type}`;
  connectionStatus.textContent = message;
}

function setProgress(message, hidden = false) {
  progressText.textContent = message;
  progressText.classList.toggle("hidden", hidden);
}

function renderResult(data) {
  const hate = data.prediction === 1;
  const textRows = Object.entries(data.model_outputs?.text || {});
  const audioRows = Object.entries(data.model_outputs?.audio || {});
  const renderRows = (rows, modality) => rows.length
    ? `<h3>${modality} model outputs</h3><ul>${rows.map(([key, value]) =>
      `<li><b>${key}</b> → ${value.label} (${(value.hate_probability * 100).toFixed(2)}% hate)</li>`).join("")}</ul>`
    : "";

  result.classList.remove("hidden");
  result.innerHTML = `
    <h2>Prediction Result
      <span class="pill ${hate ? "hate" : "safe"}">${data.label}</span>
    </h2>
    <p><b>Confidence:</b> ${(data.confidence * 100).toFixed(2)}%</p>
    <p><b>Hate Probability:</b> ${(data.hate_probability * 100).toFixed(2)}%</p>
    <p><b>Fusion:</b> ${data.fusion_method}</p>
    <p><b>Used Text:</b> ${data.used_text ? "Yes" : "No"} | <b>Used Audio:</b> ${data.used_audio ? "Yes" : "No"}</p>
    <p><b>Language:</b> ${data.language}</p>
    ${renderRows(textRows, "Text")}
    ${renderRows(audioRows, "Audio")}
  `;
}

async function checkBackendHealth() {
  try {
    const apiUrl = normalizeApiUrl(apiInput.value);
    const root = apiRootFromPredictUrl(apiUrl);
    const response = await fetch(`${root}/health`);
    if (!response.ok) {
      throw new Error(`Health check failed (${response.status})`);
    }
    setConnectionStatus("ok", "Backend connected");
  } catch (error) {
    setConnectionStatus("error", "Backend unreachable");
  }
}

async function pollJob(jobUrl) {
  for (let pollAttempt = 0; pollAttempt < MAX_POLL_ATTEMPTS; pollAttempt += 1) {
    const response = await fetch(jobUrl);
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload?.detail || "Failed to read job status.");
    }
    if (payload.status === "completed") {
      return payload.result;
    }
    if (payload.status === "failed") {
      throw new Error(payload.error || "Prediction job failed.");
    }
    setProgress(`Job status: ${payload.status}...`);
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw new Error("Prediction job timed out on frontend.");
}

apiInput.addEventListener("change", () => {
  checkBackendHealth();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitButton.disabled = true;
  submitButton.textContent = "Predicting...";
  setProgress("Preparing request...");

  try {
    const apiUrl = normalizeApiUrl(apiInput.value);
    localStorage.setItem("apiUrl", apiUrl);
    apiInput.value = apiUrl;

    const payload = new FormData();
    payload.append("language", document.getElementById("language").value);
    const textValue = document.getElementById("text").value.trim();
    if (textValue) {
      payload.append("text", textValue);
    }
    payload.append("text_models", textModelsInput.value);
    payload.append("audio_models", audioModelsInput.value);

    const audioFile = document.getElementById("audio").files[0];
    if (audioFile) {
      payload.append("audio", audioFile);
    }
    if (!textValue && !audioFile) {
      throw new Error("Please provide text or audio input.");
    }

    let data;
    if (realtimeModeInput.checked) {
      const root = apiRootFromPredictUrl(apiUrl);
      const createJobResponse = await fetch(`${root}/predict/jobs`, { method: "POST", body: payload });
      const createPayload = await createJobResponse.json();
      if (!createJobResponse.ok) {
        throw new Error(createPayload?.detail || `Request failed (${createJobResponse.status})`);
      }
      const absoluteJobUrl = createPayload.status_url.startsWith("http")
        ? createPayload.status_url
        : `${root}${createPayload.status_url}`;
      setProgress("Job created. Waiting for completion...");
      data = await pollJob(absoluteJobUrl);
    } else {
      const response = await fetch(apiUrl, { method: "POST", body: payload });
      const contentType = response.headers.get("content-type") || "";
      data = contentType.includes("application/json") ? await response.json() : null;
      if (!response.ok) {
        const raw = data ? JSON.stringify(data) : await response.text();
        const reason = data?.detail || raw || `Request failed (${response.status})`;
        throw new Error(reason);
      }
    }
    if (!data) {
      throw new Error("Backend returned an invalid response.");
    }
    renderResult(data);
    setProgress("Prediction completed.");
    await checkBackendHealth();
  } catch (error) {
    const message = error instanceof TypeError
      ? "Could not reach backend API. Check backend URL, CORS, and server status."
      : error.message;
    result.classList.remove("hidden");
    result.innerHTML = `<h2>Error</h2><p>${message}</p>`;
    setProgress("Prediction failed. Update backend URL or retry.");
    setConnectionStatus("error", "Backend issue detected");
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Predict";
  }
});

checkBackendHealth();
