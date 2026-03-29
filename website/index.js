import init, { bilateral_filter, initThreadPool } from "./pkg/rbf_rs.js";

// import summary from "./public/summary.png";

// const inp_img_elem = document.createElement("img");
// inp_img_elem.src = summary;

let inp_img_elem = document.getElementById("input");
let outp_img_elem = document.getElementById("output");

const sigmaSpatialSlider = document.getElementById("sigma-spatial");
const sigmaRangeSlider = document.getElementById("sigma-range");
const sigmaSpatialValue = document.getElementById("sigma-spatial-value");
const sigmaRangeValue = document.getElementById("sigma-range-value");
const maskSlider = document.getElementById("mask-slider");
const maskSliderValue = document.getElementById("mask-slider-value");

let sigma_spatial = parseFloat(sigmaSpatialSlider.value);
let sigma_range = parseFloat(sigmaRangeSlider.value);
let wasmModule = null;
let img_buf_rgb = null;
let img_width = null;
let img_height = null;
let result_buf = null;

function getImageData(img_elem) {
  const canvas = document.createElement("canvas");
  canvas.width = img_elem.width;
  canvas.height = img_elem.height;
  console.log(`canvas size: ${canvas.width}x${canvas.height}`);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img_elem, 0, 0);
  return ctx.getImageData(0, 0, canvas.width, canvas.height).data;
}

function setImageData(img_elem, u8buf, width, height) {
  img_elem.width = width;
  img_elem.height = height;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  const clamped = new Uint8ClampedArray(u8buf);
  const imgData = new ImageData(clamped, width, height);
  ctx.putImageData(imgData, 0, 0);
  canvas.toBlob((blob) => {
    img_elem.src = URL.createObjectURL(blob);
  }, "image/png");
}

function updateMask(percent) {
  // percent: 0-100
  outp_img_elem.style.setProperty("--mask-position", `${percent}%`);
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
  setImageData(outp_img_elem, result_buf, img_width, img_height);
}

async function main() {
  wasmModule = await init();
  await initThreadPool(navigator.hardwareConcurrency);

  // Wait for input image to be loaded
  img_width = inp_img_elem.width;
  img_height = inp_img_elem.height;
  let img_buf_rgba = getImageData(inp_img_elem);
  img_buf_rgb = img_buf_rgba.filter((_, i) => (i + 1) % 4 !== 0);

  await runFilterAndUpdate();

  // Set initial mask
  updateMask(parseInt(maskSlider.value, 10));

  // Set initial slider values
  sigmaSpatialValue.textContent = sigma_spatial.toFixed(3);
  sigmaRangeValue.textContent = sigma_range.toFixed(3);
  maskSliderValue.textContent = `${maskSlider.value}%`;
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

// Keep live transparency mask updates while sliding.
maskSlider.addEventListener("input", (e) => {
  let percent = parseInt(e.target.value, 10);
  maskSliderValue.textContent = `${percent}%`;
  updateMask(percent);
});

inp_img_elem.addEventListener("load", async () => {
  await main();
});
if (inp_img_elem.complete) {
  main();
}