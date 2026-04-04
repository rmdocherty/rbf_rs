import './style.css'
import './sw.js'
import init, { bilateral_filter, initThreadPool } from "./pkg/rbf_rs.js";


if ("serviceWorker" in navigator) {
  // Register service worker
  navigator.serviceWorker.register(new URL("./sw.js", import.meta.url)).then(
    function (registration) {
      console.log("COOP/COEP Service Worker registered", registration.scope);
      // If the registration is active, but it's not controlling the page
      if (registration.active && !navigator.serviceWorker.controller) {
          window.location.reload();
      }
    },
    function (err) {
      console.log("COOP/COEP Service Worker failed to register", err);
    }
  );
} else {
  console.warn("Cannot register a service worker");
}


let inp_img_elem = document.getElementById("input");
let outp_canvas_elem = document.getElementById("output");
let outp_ctx = outp_canvas_elem.getContext("2d");

const sigmaSpatialSlider = document.getElementById("sigma-spatial");
const sigmaRangeSlider = document.getElementById("sigma-range");
const sigmaSpatialValue = document.getElementById("sigma-spatial-value");
const sigmaRangeValue = document.getElementById("sigma-range-value");
const imageUploadInput = document.getElementById("image-upload");
const downloadImageButton = document.getElementById("download-image");

let sigma_spatial = parseFloat(sigmaSpatialSlider.value);
let sigma_range = parseFloat(sigmaRangeSlider.value);
let wasmModule = null;
let img_buf_rgb = null;
let img_width = null;
let img_height = null;
let result_buf = null;

function getImageData(img_elem) {
  const canvas = document.createElement("canvas");
  canvas.width = img_elem.naturalWidth || img_elem.width;
  canvas.height = img_elem.naturalHeight || img_elem.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img_elem, 0, 0, canvas.width, canvas.height);
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

function drawToOutput(u8buf, width, height) {
  outp_canvas_elem.width = width;
  outp_canvas_elem.height = height;
  const clamped = new Uint8ClampedArray(u8buf);
  const imgData = new ImageData(clamped, width, height);
  outp_ctx.putImageData(imgData, 0, 0);
}

function downloadImage() {
  const link = document.createElement("a");
  link.download = "filtered_image.png";
  link.href = outp_canvas_elem.toDataURL();
  link.click();
}

async function runFilterAndUpdate() {
  if (!img_buf_rgb || !img_width || !img_height || !wasmModule) return;
  result_buf = bilateral_filter(
    img_buf_rgb,
    img_width,
    img_height,
    sigma_spatial,
    sigma_range
  );
  drawToOutput(result_buf, img_width, img_height);
}

async function main() {
  if (!wasmModule) {
    wasmModule = await init();
    await initThreadPool(navigator.hardwareConcurrency);
  }

  const imgData = getImageData(inp_img_elem);
  img_width = imgData.width;
  img_height = imgData.height;
  const data = imgData.data;
  img_buf_rgb = data.filter((_, i) => (i + 1) % 4 !== 0);

  await runFilterAndUpdate();

  sigmaSpatialValue.textContent = sigma_spatial.toFixed(3);
  sigmaRangeValue.textContent = sigma_range.toFixed(3);
}

// Update values & labels live, but run filter only when user releases slider.
sigmaSpatialSlider.addEventListener("input", (e) => {
  sigma_spatial = parseFloat(e.target.value);
  sigmaSpatialValue.textContent = sigma_spatial.toFixed(3);
});

sigmaSpatialSlider.addEventListener("change", async () => {
  await runFilterAndUpdate();
});

sigmaRangeSlider.addEventListener("input", (e) => {
  sigma_range = parseFloat(e.target.value);
  sigmaRangeValue.textContent = sigma_range.toFixed(3);
});

sigmaRangeSlider.addEventListener("change", async () => {
  await runFilterAndUpdate();
});

// Allow user to choose a custom image.
imageUploadInput.addEventListener("change", (e) => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (event) => {
    inp_img_elem.src = event.target.result;
    // 'load' listener on inp_img_elem will call main() and re-run filtering
  };
  reader.readAsDataURL(file);
});

downloadImageButton.addEventListener("click", () => {
  downloadImage();
});

inp_img_elem.addEventListener("load", async () => {
  await main();
});
if (inp_img_elem.complete) {
  main();
}