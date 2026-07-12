const classNames = {
  0: "No DR",
  1: "Mild DR",
  2: "Moderate DR",
  3: "Severe DR",
  4: "Proliferative DR",
};

const fileInput = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");
const previewBox = document.getElementById("previewBox");
const previewImg = document.getElementById("previewImg");
const fileName = document.getElementById("fileName");
const fileMeta = document.getElementById("fileMeta");
const predictBtn = document.getElementById("predictBtn");
const loadingPanel = document.getElementById("loadingPanel");
const resultPanel = document.getElementById("resultPanel");

function percent(x) {
  return `${(x * 100).toFixed(2)}%`;
}

function safeNumber(x, digits = 4) {
  if (typeof x !== "number") return "-";
  return x.toFixed(digits);
}

async function checkHealth() {
  const healthText = document.getElementById("healthText");
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (res.ok && data.ready) {
      healthText.textContent = "Sẵn sàng";
    } else {
      healthText.textContent = "Chưa load model";
    }
  } catch (err) {
    healthText.textContent = "Không kết nối được";
  }
}

checkHealth();

function setPreview(file) {
  if (!file) return;

  const url = URL.createObjectURL(file);
  previewImg.src = url;
  fileName.textContent = file.name;
  fileMeta.textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
  previewBox.classList.remove("hidden");
}

fileInput.addEventListener("change", () => {
  setPreview(fileInput.files[0]);
});

["dragenter", "dragover"].forEach(eventName => {
  dropzone.addEventListener(eventName, e => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach(eventName => {
  dropzone.addEventListener(eventName, e => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", e => {
  const file = e.dataTransfer.files[0];
  if (!file) return;
  fileInput.files = e.dataTransfer.files;
  setPreview(file);
});

function renderWarnings(warnings) {
  const box = document.getElementById("warningBox");
  box.innerHTML = "";

  if (!warnings || warnings.length === 0) {
    const item = document.createElement("div");
    item.className = "warning-item";
    item.textContent = "Không có cảnh báo đặc biệt.";
    box.appendChild(item);
    return;
  }

  warnings.forEach(w => {
    const item = document.createElement("div");
    item.className = /cao|nghi|severe|class 3/i.test(w) ? "warning-item danger" : "warning-item";
    item.textContent = w;
    box.appendChild(item);
  });
}

function renderProbList(containerId, probs) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";

  probs.forEach((p, i) => {
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `
      <div class="prob-name">Class ${i} - ${classNames[i]}</div>
      <div class="prob-track">
        <div class="prob-fill" style="width:${Math.max(0, Math.min(100, p * 100))}%"></div>
      </div>
      <div class="prob-value">${percent(p)}</div>
    `;
    container.appendChild(row);
  });
}

function setupTabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  const contents = document.querySelectorAll(".tab-content");

  buttons.forEach(button => {
    button.addEventListener("click", () => {
      buttons.forEach(b => b.classList.remove("active"));
      contents.forEach(c => c.classList.remove("active"));

      button.classList.add("active");
      document.getElementById(button.dataset.tab).classList.add("active");
    });
  });
}

setupTabs();

predictBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) {
    alert("Vui lòng chọn ảnh trước.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  loadingPanel.classList.remove("hidden");
  resultPanel.classList.add("hidden");

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Dự đoán thất bại.");
    }

    document.getElementById("predictedClass").textContent =
      `Class ${data.predicted_class} - ${data.predicted_class_name}`;

    document.getElementById("confidence").textContent = percent(data.confidence);
    document.getElementById("uncertainty").textContent = percent(data.uncertainty);
    document.getElementById("severityScore").textContent = safeNumber(data.expected_severity_score, 4);

    renderWarnings(data.warnings);
    renderProbList("ensembleProbs", data.probabilities);
    renderProbList("denseProbs", data.model_outputs.densenet121);
    renderProbList("effProbs", data.model_outputs.efficientnetb3);

    document.getElementById("originalImage").src = data.original_image_url;
    document.getElementById("processedImage").src = data.processed_image_url;

    const heatmap = document.getElementById("heatmapImage");
    if (data.heatmap_url) {
      heatmap.src = data.heatmap_url;
      heatmap.style.display = "block";
    } else {
      heatmap.style.display = "none";
    }

    const reportLink = document.getElementById("reportLink");
    reportLink.href = data.report_url;

    resultPanel.classList.remove("hidden");
    resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    alert(err.message);
  } finally {
    loadingPanel.classList.add("hidden");
  }
});
