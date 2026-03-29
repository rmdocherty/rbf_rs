import init, { bilateral_filter, initThreadPool } from "./pkg/rbf_rs.js";

// import summary from "./public/summary.png";

// const inp_img_elem = document.createElement("img");
// inp_img_elem.src = summary;

let inp_img_elem = document.getElementById("input");
// document.body.appendChild(inp_img_elem);


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

async function main() {
  let wasmModule = await init();
  console.log(wasmModule.memory.buffer instanceof SharedArrayBuffer); // must be true
  await initThreadPool(navigator.hardwareConcurrency);

  // let inp_img_elem = document.getElementById("input");

  

  let outp_img_elem = document.getElementById("output");

  let img_buf_rgba = getImageData(inp_img_elem);
  let img_buf_rgb = img_buf_rgba.filter((_, i) => (i + 1) % 4 !== 0);
  console.log(`input image has ${img_buf_rgb.length / 3} pixels`);
  let result_buf = bilateral_filter(img_buf_rgb, inp_img_elem.width, inp_img_elem.height, 0.01, 0.05);

  const startTime = performance.now();
  setImageData(outp_img_elem, result_buf, inp_img_elem.width, inp_img_elem.height);
  const endTime = performance.now();
  console.log(`filtered in ${endTime - startTime} milliseconds`);
}

inp_img_elem.addEventListener("load", async () => { 
  await main();
} );
if (inp_img_elem.complete) {
    main();
}

// window.addEventListener("DOMContentLoaded", async () => {
//   main();
// });

// window.onload = async () => {
//   await main();
// }