const mediaInput = document.getElementById("mediaInput");
const chooseMediaBtn = document.getElementById("chooseMediaBtn");
const selectedFileName = document.getElementById("selectedFileName");
const sampleFpsInput = document.getElementById("sampleFps");
const processBtn = document.getElementById("processBtn");
const statusBadge = document.getElementById("statusBadge");
const frameCounter = document.getElementById("frameCounter");
const responseBox = document.getElementById("responseBox");
const previewCanvas = document.getElementById("previewCanvas");
const previewCtx = previewCanvas.getContext("2d");

let isProcessing = false;

processBtn.disabled = true;

chooseMediaBtn.addEventListener("click", () => {
    mediaInput.click();
});

mediaInput.addEventListener("change", () => {
    const mediaFile = mediaInput.files?.[0];
    if (!mediaFile) {
        selectedFileName.textContent = "Файл не выбран";
        processBtn.disabled = true;
        setStatus("Ожидание файла", "idle");
        return;
    }

    selectedFileName.textContent = `Выбран файл: ${mediaFile.name}`;
    processBtn.disabled = false;
    setStatus("Файл выбран, нажмите «Запустить обработку»", "idle");
});

processBtn.addEventListener("click", () => {
    void processMedia();
});

async function processMedia() {
    const mediaFile = mediaInput.files?.[0];
    if (!mediaFile) {
        setStatus("Сначала выберите файл", "error");
        return;
    }

    if (isProcessing) {
        return;
    }

    isProcessing = true;
    processBtn.disabled = true;

    try {
        const mediaKind = detectMediaKind(mediaFile);
        if (mediaKind === "image") {
            await processImage(mediaFile);
        } else if (mediaKind === "video") {
            await processVideo(mediaFile);
        } else {
            throw new Error("Поддерживаются только image/* и video/* файлы");
        }
    } catch (error) {
        const message = error instanceof Error ? error.message : "Неизвестная ошибка";
        setStatus(`Ошибка: ${message}`, "error");
    } finally {
        isProcessing = false;
        processBtn.disabled = !mediaInput.files?.[0];
    }
}

async function processImage(imageFile) {
    setStatus("Отправка изображения на backend", "busy");
    const result = await sendFrame(imageFile, 0);
    await drawFrameAndMasks(result);
    frameCounter.textContent = "Кадр: 0";
    responseBox.textContent = formatResponse(result);
    const doneStatus = result.violation_detected
        ? "Изображение обработано, нарушение найдено"
        : "Изображение обработано";
    setStatus(doneStatus, "done");
}

async function processVideo(videoFile) {
    const sampleFps = normalizeFps(Number(sampleFpsInput.value));
    const timeStep = 1 / sampleFps;

    const video = document.createElement("video");
    video.src = URL.createObjectURL(videoFile);
    video.muted = true;
    video.playsInline = true;

    await once(video, "loadedmetadata");

    if (!Number.isFinite(video.duration) || video.duration <= 0) {
        URL.revokeObjectURL(video.src);
        throw new Error("Видео не удалось прочитать");
    }

    let frameIndex = 0;
    const totalFrames = Math.max(1, Math.ceil(video.duration * sampleFps));

    for (let t = 0; t < video.duration; t += timeStep) {
        setStatus(`Обработка видео: кадр ${frameIndex + 1}/${totalFrames}`, "busy");

        await seekVideo(video, Math.min(t, video.duration));
        const frameBlob = await captureVideoFrame(video);

        const result = await sendFrame(frameBlob, frameIndex);
        await drawFrameAndMasks(result);
        responseBox.textContent = formatResponse(result);
        frameCounter.textContent = `Кадр: ${frameIndex}`;
        frameIndex += 1;
    }

    URL.revokeObjectURL(video.src);
    setStatus(`Видео обработано, отправлено кадров: ${frameIndex}`, "done");
}

async function sendFrame(frameBlob, frameIndex) {
    const formData = new FormData();
    formData.append("frame", frameBlob, `frame_${frameIndex}.jpg`);
    formData.append("frame_index", String(frameIndex));

    const response = await fetch("/api/infer/violation", {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        const message = await response.text();
        throw new Error(`backend ${response.status}: ${message}`);
    }

    return response.json();
}

async function drawFrameAndMasks(result) {
    const image = await loadImage(result.frame_data_url);

    configurePreviewCanvas(result.frame_width, result.frame_height);

    previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
    previewCtx.drawImage(image, 0, 0, previewCanvas.width, previewCanvas.height);

    for (const mask of result.masks || []) {
        drawMaskPolygon(mask, previewCanvas.width, previewCanvas.height);
    }

    if (result.violation_score > 0.7){
        console.log(result.violation_score)
        for (const region of result.violation_regions || []) {
            drawViolationRegion(region, previewCanvas.width, previewCanvas.height);
        }
    }
}

function configurePreviewCanvas(width, height) {
    previewCanvas.width = width;
    previewCanvas.height = height;
    previewCanvas.style.aspectRatio = `${width} / ${height}`;
    previewCanvas.setAttribute("aria-label", `Кадр ${width} на ${height}`);
}

function drawMaskPolygon(mask, width, height) {
    const color = mask.class_id === 1 ? "#07a78f" : "#f1613f";
    const points = mask.points || [];
    if (points.length < 3) {
        return;
    }

    previewCtx.save();
    previewCtx.lineWidth = 3;
    previewCtx.strokeStyle = color;
    previewCtx.fillStyle = `${color}33`;

    previewCtx.beginPath();
    previewCtx.moveTo(points[0][0] * width, points[0][1] * height);
    for (let i = 1; i < points.length; i += 1) {
        previewCtx.lineTo(points[i][0] * width, points[i][1] * height);
    }
    previewCtx.closePath();
    previewCtx.fill();
    previewCtx.stroke();

    const labelX = points[0][0] * width + 6;
    const labelY = points[0][1] * height - 8;
    previewCtx.font = "600 14px 'IBM Plex Sans'";
    previewCtx.fillStyle = color;
    previewCtx.fillText(`${mask.model_name}: ${mask.class_name}`, labelX, labelY);
    previewCtx.restore();
}

function drawViolationRegion(region, width, height) {
    const points = region.points || [];
    if (points.length < 3) {
        return;
    }

    previewCtx.save();
    previewCtx.lineWidth = Math.max(3, Math.round(Math.min(width, height) * 0.004));
    previewCtx.strokeStyle = "#e31937";
    previewCtx.fillStyle = "rgba(227, 25, 55, 0.32)";

    previewCtx.beginPath();
    previewCtx.moveTo(points[0][0] * width, points[0][1] * height);
    for (let i = 1; i < points.length; i += 1) {
        previewCtx.lineTo(points[i][0] * width, points[i][1] * height);
    }
    previewCtx.closePath();
    previewCtx.fill();
    previewCtx.stroke();

    const labelX = points[0][0] * width + previewCtx.lineWidth * 2;
    const labelY = Math.max(
        points[0][1] * height + previewCtx.lineWidth * 8,
        previewCtx.lineWidth * 8,
    );
    const confidence = Number.isFinite(region.confidence)
        ? ` ${(region.confidence * 100).toFixed(0)}%`
        : "";
    // const fontSize = Math.max(14, Math.round(Math.min(width, height) * 0.018));
    previewCtx.font = `700 14px 'IBM Plex Sans'`;
    previewCtx.fillStyle = "#ffffff";
    previewCtx.fillText(`Нарушение${confidence}`, labelX, labelY);
    previewCtx.restore();
}

function formatResponse(result) {
    const lines = [
        `frame_index: ${result.frame_index}`,
        `resolution: ${result.frame_width}x${result.frame_height}`,
        "",
    ];

    if (typeof result.violation_detected === "boolean") {
        lines.push(`violation_detected: ${result.violation_detected}`);
        lines.push(`violation_score: ${Number(result.violation_score || 0).toFixed(4)}`);
        lines.push(`reason: ${result.reason || "-"}`);
        lines.push("");
    }

    for (const region of result.violation_regions || []) {
        lines.push(`violation_model: ${region.model_name}`);
        lines.push(`class: ${region.class_id} (${region.class_name})`);
        lines.push(`confidence: ${Number(region.confidence || 0).toFixed(4)}`);
        lines.push(`bbox_xyxy: ${region.bbox_xyxy}`);
        lines.push(`yolo: ${region.yolo_segmentation}`);
        lines.push("");
    }

    for (const mask of result.masks || []) {
        lines.push(`model: ${mask.model_name}`);
        lines.push(`class: ${mask.class_id} (${mask.class_name})`);
        lines.push(`yolo: ${mask.yolo_segmentation}`);
        lines.push("");
    }

    return lines.join("\n");
}

function normalizeFps(value) {
    if (!Number.isFinite(value)) {
        return 2;
    }
    return Math.min(15, Math.max(0.5, value));
}

function detectMediaKind(file) {
    const mimeType = (file.type || "").toLowerCase();
    if (mimeType.startsWith("image/")) {
        return "image";
    }

    if (mimeType.startsWith("video/")) {
        return "video";
    }

    const fileName = (file.name || "").toLowerCase();
    if (/\.(jpg|jpeg|png|bmp|webp|gif)$/.test(fileName)) {
        return "image";
    }

    if (/\.(mp4|mov|avi|mkv|webm|m4v)$/.test(fileName)) {
        return "video";
    }

    return "unknown";
}

async function captureVideoFrame(videoElement) {
    const frameCanvas = document.createElement("canvas");
    frameCanvas.width = videoElement.videoWidth;
    frameCanvas.height = videoElement.videoHeight;
    const ctx = frameCanvas.getContext("2d");
    ctx.drawImage(videoElement, 0, 0, frameCanvas.width, frameCanvas.height);

    return new Promise((resolve, reject) => {
        frameCanvas.toBlob(
            (blob) => {
                if (blob) {
                    resolve(blob);
                    return;
                }
                reject(new Error("Не удалось получить кадр из видео"));
            },
            "image/jpeg",
            0.92,
        );
    });
}

async function seekVideo(videoElement, timeSeconds) {
    return new Promise((resolve, reject) => {
        if (
            Math.abs(videoElement.currentTime - timeSeconds) < 0.001 &&
            videoElement.readyState >= 2
        ) {
            resolve();
            return;
        }

        const onSeeked = () => {
            videoElement.removeEventListener("seeked", onSeeked);
            videoElement.removeEventListener("error", onError);
            resolve();
        };

        const onError = () => {
            videoElement.removeEventListener("seeked", onSeeked);
            videoElement.removeEventListener("error", onError);
            reject(new Error("Не удалось перемотать видео"));
        };

        videoElement.addEventListener("seeked", onSeeked, { once: true });
        videoElement.addEventListener("error", onError, { once: true });
        videoElement.currentTime = timeSeconds;
    });
}

function loadImage(src) {
    return new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = () => reject(new Error("Не удалось декодировать кадр backend"));
        image.src = src;
    });
}

function once(target, eventName) {
    return new Promise((resolve, reject) => {
        const onEvent = () => {
            target.removeEventListener(eventName, onEvent);
            target.removeEventListener("error", onError);
            resolve();
        };

        const onError = () => {
            target.removeEventListener(eventName, onEvent);
            target.removeEventListener("error", onError);
            reject(new Error(`Событие ${eventName} завершилось ошибкой`));
        };

        target.addEventListener(eventName, onEvent, { once: true });
        target.addEventListener("error", onError, { once: true });
    });
}

function setStatus(text, state) {
    statusBadge.textContent = text;
    statusBadge.className = `status status-${state}`;
}
