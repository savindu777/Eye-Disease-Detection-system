"use strict";

(() => {
  const MAX_BYTES = 8 * 1024 * 1024;
  const acceptedTypes = new Set(["image/jpeg", "image/png"]);
  const allowedExtensions = /\.(jpe?g|png)$/i;

  const input = document.getElementById("imageInput");
  const dropzone = document.getElementById("dropzone");
  const dropzoneEmpty = document.getElementById("dropzoneEmpty");
  const dropzonePreview = document.getElementById("dropzonePreview");
  const previewImage = document.getElementById("imagePreview");
  const fileDetails = document.getElementById("selectedFile");
  const fileName = document.getElementById("fileName");
  const fileSize = document.getElementById("fileSize");
  const removeButton = document.getElementById("removeImage");
  const submitButton = document.getElementById("analyzeButton");
  const submitText = submitButton.querySelector(".analyze-default");
  const submitLoading = submitButton.querySelector(".analyze-loading");
  const errorAlert = document.getElementById("errorAlert");
  const errorText = document.getElementById("errorText");
  const emptyResult = document.getElementById("emptyResult");
  const loadingResult = document.getElementById("loadingResult");
  const completeResult = document.getElementById("completedResult");
  const resultBottom = document.getElementById("resultBottom");
  const analyzeAnother = document.getElementById("analyzeAnother");
  const probabilityList = document.getElementById("probabilityList");
  const resultClass = document.getElementById("predictedClass");
  const resultDescription = document.getElementById("resultDescription");
  const confidenceRing = document.getElementById("confidenceRing");
  const confidenceValue = document.getElementById("confidenceValue");

  let selectedFile = null;
  let previewUrl = null;
  let analyzing = false;

  function showError(message) {
    errorText.textContent = message;
    errorAlert.classList.remove("d-none");
  }
  function clearError() { errorAlert.classList.add("d-none"); }
  function readableSize(bytes) {
    return bytes >= 1024 * 1024
      ? `${(bytes / (1024 * 1024)).toFixed(2)} MB`
      : `${Math.max(1, Math.ceil(bytes / 1024))} KB`;
  }
  function resetResult() {
    completeResult.classList.add("d-none");
    loadingResult.classList.add("d-none");
    emptyResult.classList.remove("d-none");
    resultBottom.classList.remove("d-none");
    probabilityList.replaceChildren();
  }
  function setBusy(isBusy) {
    analyzing = isBusy;
    submitButton.disabled = isBusy || !selectedFile;
    submitText.classList.toggle("d-none", isBusy);
    submitLoading.classList.toggle("d-none", !isBusy);
    input.disabled = isBusy;
    dropzone.setAttribute("aria-disabled", String(isBusy));
    removeButton.disabled = isBusy;
  }
  function clearPreview() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    previewImage.removeAttribute("src");
  }
  function resetAll() {
    if (analyzing) return;
    selectedFile = null;
    input.value = "";
    clearPreview();
    clearError();
    dropzoneEmpty.classList.remove("d-none");
    dropzonePreview.classList.add("d-none");
    fileDetails.classList.add("d-none");
    submitButton.disabled = true;
    resetResult();
  }
  function selectFile(file) {
    if (analyzing || !file) return;
    clearError();
    if (!allowedExtensions.test(file.name) || !acceptedTypes.has(file.type)) {
      showError("Please choose a JPG, JPEG, or PNG image.");
      return;
    }
    if (file.size === 0 || file.size > MAX_BYTES) {
      showError("Choose an image larger than 0 bytes and smaller than 8 MB.");
      return;
    }

    const freshUrl = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      if (analyzing) { URL.revokeObjectURL(freshUrl); return; }
      clearPreview();
      previewUrl = freshUrl;
      selectedFile = file;
      previewImage.src = freshUrl;
      dropzoneEmpty.classList.add("d-none");
      dropzonePreview.classList.remove("d-none");
      fileName.textContent = file.name;
      fileSize.textContent = `${readableSize(file.size)} · ${image.naturalWidth} × ${image.naturalHeight} px`;
      fileDetails.classList.remove("d-none");
      submitButton.disabled = false;
      resetResult();
    };
    image.onerror = () => {
      URL.revokeObjectURL(freshUrl);
      showError("Could not preview this image. Please choose another JPG or PNG file.");
    };
    image.src = freshUrl;
  }
  function openPicker() { if (!analyzing) input.click(); }

  dropzone.addEventListener("click", openPicker);
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openPicker();
    }
  });
  input.addEventListener("change", () => {
    if (input.files && input.files.length) selectFile(input.files[0]);
  });
  ["dragenter", "dragover"].forEach((name) => {
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      if (analyzing) return;
      dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "dragend"].forEach((name) => {
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
    if (!analyzing && event.dataTransfer.files.length) selectFile(event.dataTransfer.files[0]);
  });
  removeButton.addEventListener("click", resetAll);
  analyzeAnother.addEventListener("click", () => {
    resetAll();
    document.getElementById("analyze").scrollIntoView({ behavior: "smooth", block: "start" });
  });

  function renderPrediction(result) {
    emptyResult.classList.add("d-none");
    loadingResult.classList.add("d-none");
    resultBottom.classList.add("d-none");
    completeResult.classList.remove("d-none");

    const confidence = Math.max(0, Math.min(100, Number(result.confidence) || 0));
    resultClass.textContent = result.display_name;
    resultDescription.textContent = result.summary;
    confidenceValue.textContent = `${confidence.toFixed(1)}%`;
    confidenceRing.style.setProperty("--score", `${confidence}%`);
    probabilityList.replaceChildren();
    const probabilities = [...result.probabilities].sort((a, b) => b.percentage - a.percentage);
    for (const item of probabilities) {
      const percent = Math.max(0, Math.min(100, Number(item.percentage) || 0));
      const isTop = item.label === result.label;
      const row = document.createElement("div");
      row.className = "probability-item";
      const heading = document.createElement("div");
      heading.className = "probability-line";
      const label = document.createElement("span");
      label.textContent = item.display_name;
      if (isTop) label.classList.add("is-winner");
      const value = document.createElement("strong");
      value.textContent = `${percent.toFixed(2)}%`;
      heading.append(label, value);
      const track = document.createElement("div");
      track.className = "probability-track";
      track.setAttribute("role", "progressbar");
      track.setAttribute("aria-label", `${item.display_name} model score`);
      track.setAttribute("aria-valuemin", "0");
      track.setAttribute("aria-valuemax", "100");
      track.setAttribute("aria-valuenow", percent.toString());
      const fill = document.createElement("span");
      fill.className = `probability-fill${isTop ? " top-score" : ""}`;
      track.appendChild(fill);
      row.append(heading, track);
      probabilityList.appendChild(row);
      requestAnimationFrame(() => { fill.style.width = `${percent}%`; });
    }
  }

  submitButton.addEventListener("click", async () => {
    if (!selectedFile || analyzing) return;
    clearError();
    setBusy(true);
    emptyResult.classList.add("d-none");
    completeResult.classList.add("d-none");
    loadingResult.classList.remove("d-none");
    resultBottom.classList.add("d-none");

    const data = new FormData();
    data.append("image", selectedFile, selectedFile.name);
    try {
      const response = await fetch("/predict", { method: "POST", body: data });
      let payload;
      try { payload = await response.json(); }
      catch { throw new Error("The server returned an unreadable response."); }
      if (!response.ok || !payload.success || !payload.prediction) {
        throw new Error(payload.error || "Image analysis failed. Please try again.");
      }
      renderPrediction(payload.prediction);
    } catch (error) {
      resetResult();
      showError(error instanceof Error ? error.message : "An unexpected error occurred.");
    } finally {
      setBusy(false);
    }
  });
})();
