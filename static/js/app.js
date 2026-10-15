/**
 * VisionLens AI - Interactive Client Application
 */

document.addEventListener("DOMContentLoaded", () => {
    // State
    let currentModel = "mobilenet_v2";
    let chartInstance = null;
    let webcamStream = null;
    let liveInferenceInterval = null;
    let batchResultsData = [];
    let lastClassificationResult = null;

    // DOM Elements - Navigation & Header
    const tabButtons = document.querySelectorAll(".tab-btn");
    const tabPanes = document.querySelectorAll(".tab-pane");
    const modelSelect = document.getElementById("modelSelect");
    const systemStatus = document.getElementById("systemStatus");
    const deviceText = document.getElementById("deviceText");

    // DOM Elements - Single & URL
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const btnBrowse = document.getElementById("btnBrowse");
    const imageUrlInput = document.getElementById("imageUrlInput");
    const btnFetchUrl = document.getElementById("btnFetchUrl");
    const sampleChips = document.getElementById("sampleChips");
    const chkGradCAM = document.getElementById("chkGradCAM");

    // DOM Elements - Output Results
    const emptyState = document.getElementById("emptyState");
    const loadingState = document.getElementById("loadingState");
    const resultsContent = document.getElementById("resultsContent");
    const topPredictedLabel = document.getElementById("topPredictedLabel");
    const topConfidenceNum = document.getElementById("topConfidenceNum");
    const latencyVal = document.getElementById("latencyVal");
    const modelUsedName = document.getElementById("modelUsedName");
    const imgDimensions = document.getElementById("imgDimensions");
    const imgOriginalPreview = document.getElementById("imgOriginalPreview");
    const imgHeatmapPreview = document.getElementById("imgHeatmapPreview");
    const gradCamBox = document.getElementById("gradCamBox");
    const resultsTableBody = document.getElementById("resultsTableBody");
    const btnCopyJson = document.getElementById("btnCopyJson");

    // DOM Elements - Webcam
    const webcamVideo = document.getElementById("webcamVideo");
    const captureCanvas = document.getElementById("captureCanvas");
    const btnStartCamera = document.getElementById("btnStartCamera");
    const btnCapturePhoto = document.getElementById("btnCapturePhoto");
    const btnToggleLiveInference = document.getElementById("btnToggleLiveInference");
    const btnStopCamera = document.getElementById("btnStopCamera");
    const webcamLiveLabel = document.getElementById("webcamLiveLabel");
    const webcamLiveConf = document.getElementById("webcamLiveConf");
    const snapshotsStrip = document.getElementById("snapshotsStrip");

    // DOM Elements - Batch
    const batchFileInput = document.getElementById("batchFileInput");
    const btnSelectBatch = document.getElementById("btnSelectBatch");
    const btnExportBatchCsv = document.getElementById("btnExportBatchCsv");
    const batchEmpty = document.getElementById("batchEmpty");
    const batchLoading = document.getElementById("batchLoading");
    const batchTable = document.getElementById("batchTable");
    const batchTableBody = document.getElementById("batchTableBody");

    // ── 1. Initialize App & Check Server Health ──────────────────────────────
    async function initApp() {
        try {
            const healthRes = await fetch("/api/health");
            if (healthRes.ok) {
                const healthData = await healthRes.json();
                deviceText.innerHTML = `<i class="fa-solid fa-bolt"></i> PyTorch ${healthData.device}`;
            }
        } catch (e) {
            console.warn("Backend health check failed:", e);
        }

        loadSampleImages();
    }

    // ── 2. Tab Navigation ───────────────────────────────────────────────────
    tabButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetTab = btn.getAttribute("data-tab");
            tabButtons.forEach(b => b.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));
            btn.classList.add("active");
            document.getElementById(targetTab).classList.add("active");
        });
    });

    // Model Switcher
    modelSelect.addEventListener("change", (e) => {
        currentModel = e.target.value;
    });

    // ── 3. Load Sample Gallery Chips ─────────────────────────────────────────
    async function loadSampleImages() {
        try {
            const res = await fetch("/api/samples");
            const data = await res.json();
            if (data.samples && data.samples.length > 0) {
                sampleChips.innerHTML = "";
                data.samples.forEach(sample => {
                    const chip = document.createElement("button");
                    chip.className = "sample-chip-btn";
                    chip.innerHTML = `
                        <img src="${sample.url}" class="sample-chip-img" alt="${sample.title}">
                        <span>${sample.title}</span>
                    `;
                    chip.addEventListener("click", () => {
                        classifyFromUrl(window.location.origin + sample.url);
                    });
                    sampleChips.appendChild(chip);
                });
            } else {
                sampleChips.innerHTML = `<span class="hint-text">No samples found</span>`;
            }
        } catch (err) {
            sampleChips.innerHTML = `<span class="hint-text">Error loading samples</span>`;
        }
    }

    // ── 4. File Drag & Drop & Upload Handlers ────────────────────────────────
    btnBrowse.addEventListener("click", (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            classifyFromFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            classifyFromFile(e.target.files[0]);
        }
    });

    btnFetchUrl.addEventListener("click", () => {
        const url = imageUrlInput.value.trim();
        if (!url) {
            alert("Please paste a valid image URL.");
            return;
        }
        classifyFromUrl(url);
    });

    // ── 5. Main Inference API Handlers ──────────────────────────────────────
    function showLoading(msg = "Running Neural Forward Pass...") {
        emptyState.classList.add("hidden");
        resultsContent.classList.add("hidden");
        loadingState.classList.remove("hidden");
        document.getElementById("loadingText").innerText = msg;
    }

    function hideLoading() {
        loadingState.classList.add("hidden");
        resultsContent.classList.remove("hidden");
    }

    async function classifyFromFile(file) {
        showLoading(`Classifying ${file.name}...`);
        const formData = new FormData();
        formData.append("file", file);
        formData.append("model_id", currentModel);
        formData.append("generate_heatmap", chkGradCAM.checked);

        try {
            const resp = await fetch("/api/classify", {
                method: "POST",
                body: formData
            });
            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || "Classification failed");
            }
            const result = await resp.json();
            renderResults(result);
        } catch (err) {
            alert(`Error: ${err.message}`);
            emptyState.classList.remove("hidden");
            loadingState.classList.add("hidden");
        }
    }

    async function classifyFromUrl(url) {
        showLoading("Downloading & Classifying Image...");
        const formData = new FormData();
        formData.append("image_url", url);
        formData.append("model_id", currentModel);
        formData.append("generate_heatmap", chkGradCAM.checked);

        try {
            const resp = await fetch("/api/classify", {
                method: "POST",
                body: formData
            });
            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || "Classification failed");
            }
            const result = await resp.json();
            renderResults(result);
        } catch (err) {
            alert(`Error: ${err.message}`);
            emptyState.classList.remove("hidden");
            loadingState.classList.add("hidden");
        }
    }

    // ── 6. Render Results & Charts ──────────────────────────────────────────
    function renderResults(res) {
        hideLoading();
        lastClassificationResult = res;

        const topPred = res.predictions[0];
        topPredictedLabel.innerText = topPred.label;
        topConfidenceNum.innerText = `${topPred.confidence}%`;
        latencyVal.innerText = `${res.latency_ms} ms`;
        modelUsedName.innerText = res.model_info.name;
        imgDimensions.innerText = res.image_dimensions;

        // Set Images
        imgOriginalPreview.src = res.image_preview;
        if (res.heatmap_base64) {
            gradCamBox.style.display = "block";
            imgHeatmapPreview.src = `data:image/jpeg;base64,${res.heatmap_base64}`;
        } else {
            gradCamBox.style.display = "none";
        }

        // Table
        resultsTableBody.innerHTML = "";
        res.predictions.forEach((pred, i) => {
            const row = document.createElement("tr");
            row.innerHTML = `
                <td><strong>#${i + 1}</strong></td>
                <td><strong>${pred.label}</strong></td>
                <td><code style="color: #06b6d4;">#${pred.class_id}</code></td>
                <td><strong>${pred.confidence}%</strong></td>
                <td><span class="status-badge ${i === 0 ? 'status-top' : 'status-alt'}">${i === 0 ? 'Top Match' : 'Alternative'}</span></td>
            `;
            resultsTableBody.appendChild(row);
        });

        // Render Probability Chart
        renderChart(res.predictions);
    }

    function renderChart(predictions) {
        const labels = predictions.map(p => p.label);
        const dataValues = predictions.map(p => p.confidence);

        if (chartInstance) {
            chartInstance.destroy();
        }

        const ctx = document.getElementById("probChart").getContext("2d");
        chartInstance = new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    label: "Confidence %",
                    data: dataValues,
                    backgroundColor: [
                        "#10b981",
                        "#6366f1",
                        "#06b6d4",
                        "#f59e0b",
                        "#8b5cf6"
                    ],
                    borderRadius: 6,
                    borderWidth: 0
                }]
            },
            options: {
                indexAxis: "y",
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => `Confidence: ${ctx.parsed.x}%`
                        }
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "#94a3b8", callback: (val) => `${val}%` }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: "#f8fafc", font: { weight: "600", family: "Outfit" } }
                    }
                }
            }
        });
    }

    btnCopyJson.addEventListener("click", () => {
        if (lastClassificationResult) {
            navigator.clipboard.writeText(JSON.stringify(lastClassificationResult, null, 2));
            btnCopyJson.innerHTML = `<i class="fa-solid fa-check"></i> Copied!`;
            setTimeout(() => {
                btnCopyJson.innerHTML = `<i class="fa-solid fa-code"></i> Copy JSON`;
            }, 2000);
        }
    });

    // ── 7. Live AI Webcam Module ─────────────────────────────────────────────
    btnStartCamera.addEventListener("click", async () => {
        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
                audio: false
            });
            webcamVideo.srcObject = webcamStream;
            btnStartCamera.disabled = true;
            btnCapturePhoto.disabled = false;
            btnToggleLiveInference.disabled = false;
            btnStopCamera.disabled = false;
            webcamLiveLabel.innerText = "Camera Ready";
        } catch (err) {
            alert("Could not access camera: " + err.message);
        }
    });

    function captureWebcamFrameBase64() {
        if (!webcamStream) return null;
        captureCanvas.width = webcamVideo.videoWidth || 640;
        captureCanvas.height = webcamVideo.videoHeight || 480;
        const ctx = captureCanvas.getContext("2d");
        ctx.drawImage(webcamVideo, 0, 0, captureCanvas.width, captureCanvas.height);
        return captureCanvas.toDataURL("image/jpeg", 0.85);
    }

    async function sendFrameForLivePrediction(b64) {
        const formData = new FormData();
        formData.append("image_base64", b64);
        formData.append("model_id", currentModel);
        formData.append("generate_heatmap", "false");

        try {
            const resp = await fetch("/api/classify", {
                method: "POST",
                body: formData
            });
            if (resp.ok) {
                const res = await resp.json();
                const top = res.predictions[0];
                webcamLiveLabel.innerText = top.label;
                webcamLiveConf.innerText = `${top.confidence}% (${res.latency_ms}ms)`;
                return res;
            }
        } catch (e) {
            console.error("Live inference error:", e);
        }
        return null;
    }

    btnCapturePhoto.addEventListener("click", async () => {
        const b64 = captureWebcamFrameBase64();
        if (!b64) return;
        webcamLiveLabel.innerText = "Analyzing Snapshot...";
        
        const res = await sendFrameForLivePrediction(b64);
        if (res) {
            // Add snapshot to sidebar history
            const snap = document.createElement("div");
            snap.className = "snap-item";
            snap.innerHTML = `
                <img src="${b64}" alt="Snap">
                <div class="snap-info">
                    <h6>${res.predictions[0].label}</h6>
                    <span>${res.predictions[0].confidence}% &bull; ${res.latency_ms}ms</span>
                </div>
            `;
            if (snapshotsStrip.querySelector(".empty-hint")) {
                snapshotsStrip.innerHTML = "";
            }
            snapshotsStrip.prepend(snap);
        }
    });

    btnToggleLiveInference.addEventListener("click", () => {
        if (liveInferenceInterval) {
            clearInterval(liveInferenceInterval);
            liveInferenceInterval = null;
            btnToggleLiveInference.innerHTML = `<i class="fa-solid fa-bolt"></i> Continuous Mode`;
            btnToggleLiveInference.classList.remove("btn-success");
            btnToggleLiveInference.classList.add("btn-secondary");
        } else {
            btnToggleLiveInference.innerHTML = `<i class="fa-solid fa-pause"></i> Pause Continuous`;
            btnToggleLiveInference.classList.remove("btn-secondary");
            btnToggleLiveInference.classList.add("btn-success");
            
            // Run inference every 600ms
            liveInferenceInterval = setInterval(() => {
                const b64 = captureWebcamFrameBase64();
                if (b64) sendFrameForLivePrediction(b64);
            }, 600);
        }
    });

    btnStopCamera.addEventListener("click", () => {
        if (liveInferenceInterval) {
            clearInterval(liveInferenceInterval);
            liveInferenceInterval = null;
        }
        if (webcamStream) {
            webcamStream.getTracks().forEach(t => t.stop());
            webcamStream = null;
        }
        webcamVideo.srcObject = null;
        btnStartCamera.disabled = false;
        btnCapturePhoto.disabled = true;
        btnToggleLiveInference.disabled = true;
        btnStopCamera.disabled = true;
        webcamLiveLabel.innerText = "Camera Idle";
        webcamLiveConf.innerText = "-- %";
    });

    // ── 8. Batch Inference Studio ───────────────────────────────────────────
    btnSelectBatch.addEventListener("click", () => batchFileInput.click());

    batchFileInput.addEventListener("change", async (e) => {
        const files = Array.from(e.target.files);
        if (files.length === 0) return;

        batchEmpty.classList.add("hidden");
        batchTable.classList.add("hidden");
        batchLoading.classList.remove("hidden");

        const formData = new FormData();
        files.forEach(f => formData.append("files", f));
        formData.append("model_id", currentModel);

        try {
            const resp = await fetch("/api/batch-classify", {
                method: "POST",
                body: formData
            });
            if (!resp.ok) throw new Error("Batch processing failed");
            const data = await resp.json();
            batchResultsData = data.results;

            batchLoading.classList.add("hidden");
            batchTable.classList.remove("hidden");
            btnExportBatchCsv.disabled = false;

            batchTableBody.innerHTML = "";
            data.results.forEach((item, i) => {
                const row = document.createElement("tr");
                if (item.success) {
                    row.innerHTML = `
                        <td><strong>${i + 1}</strong></td>
                        <td>${item.filename}</td>
                        <td><strong>${item.top_prediction.label}</strong></td>
                        <td><span style="color:#10b981; font-weight:700;">${item.top_prediction.confidence}%</span></td>
                        <td>${item.latency_ms} ms</td>
                        <td><span class="status-badge status-top">Success</span></td>
                    `;
                } else {
                    row.innerHTML = `
                        <td><strong>${i + 1}</strong></td>
                        <td>${item.filename}</td>
                        <td colspan="3" style="color:#f43f5e;">${item.error}</td>
                        <td><span class="status-badge status-alt">Failed</span></td>
                    `;
                }
                batchTableBody.appendChild(row);
            });
        } catch (err) {
            alert(`Batch error: ${err.message}`);
            batchLoading.classList.add("hidden");
            batchEmpty.classList.remove("hidden");
        }
    });

    btnExportBatchCsv.addEventListener("click", () => {
        if (!batchResultsData.length) return;
        let csv = "Index,Filename,TopPrediction,ConfidencePercent,LatencyMs,Status\n";
        batchResultsData.forEach((item, i) => {
            if (item.success) {
                csv += `${i + 1},"${item.filename}","${item.top_prediction.label}",${item.top_prediction.confidence},${item.latency_ms},Success\n`;
            } else {
                csv += `${i + 1},"${item.filename}","N/A",0,0,Failed\n`;
            }
        });

        const blob = new Blob([csv], { type: "text/csv" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `VisionLens_Batch_Results_${Date.now()}.csv`;
        a.click();
    });

    // Start
    initApp();
});
