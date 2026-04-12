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

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitButton.disabled = true;
  submitButton.textContent = "Predicting...";

  try {
    const apiUrl = document.getElementById("api-url").value.trim();
    const payload = new FormData();
    payload.append("language", document.getElementById("language").value);
    payload.append("text", document.getElementById("text").value);

    const audioFile = document.getElementById("audio").files[0];
    if (audioFile) {
      payload.append("audio", audioFile);
    }

    const response = await fetch(apiUrl, { method: "POST", body: payload });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Prediction failed");
    }

    const hate = data.prediction === 1;
    result.classList.remove("hidden");
    result.innerHTML = `
      <h2>Prediction Result
        <span class="pill ${hate ? "hate" : "safe"}">${data.label}</span>
      </h2>
      <p><b>Confidence:</b> ${(data.confidence * 100).toFixed(2)}%</p>
      <p><b>Hate Probability:</b> ${(data.hate_probability * 100).toFixed(2)}%</p>
      <p><b>Fusion:</b> ${data.fusion_method}</p>
      <p><b>Used Audio:</b> ${data.used_audio ? "Yes" : "No"}</p>
      <p><b>Language:</b> ${data.language}</p>
    `;
  } catch (error) {
    result.classList.remove("hidden");
    result.innerHTML = `<h2>Error</h2><p>${error.message}</p>`;
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Predict";
  }
});
