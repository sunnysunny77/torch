let drawing = false;

const CANVAS_WIDTH = 280;
const CANVAS_HEIGHT = 280;
const INVERT = false;

const host = "http://127.0.0.1:8000";

const canvas = document.querySelector(".quad");
const resetBtn = document.querySelector("#resetBtn");
const predictBtn = document.querySelector("#predictBtn");
const clearBtn = document.querySelector("#clearBtn");
const message = document.querySelector("#message");
const output = document.querySelector("#output");

canvas.width = CANVAS_WIDTH;
canvas.height = CANVAS_HEIGHT;
const ctx = canvas.getContext("2d");

let token = "";

const setRandomLabels = async () => {
  try {
    const res = await fetch(`${host}/random_label_image`);
    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();
    token = data.token;
    output.innerHTML = `<img src='data:image/png;base64,${data.image}' alt='label' />`;
  } catch (err) {
    console.error(err);
    message.innerText = "Error";
  }
};

const clear = () => {
  if (INVERT) {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
  } else {
    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
  }
};

resetBtn.addEventListener("click",  async () => {
  clear();
  await setRandomLabels();
  message.innerText = "Draw the word";
});

clearBtn.addEventListener("click", () => {
  clear();
  message.innerText = "Draw the word";
});

predictBtn.addEventListener("click", async () => {
  try {
    predictBtn.disabled = true;
    message.innerHTML = "<img class='spinner' src='./images/spinner.gif' width='30' height='30' alt='spinner' />";

    const fullCanvas = document.createElement("canvas");
    fullCanvas.width = canvas.width;
    fullCanvas.height = canvas.height;
    const fullCtx = fullCanvas.getContext("2d");

    const image = ctx.getImageData(0, 0, canvas.width, canvas.height);;

    if (INVERT) {
      const invertedData = ctx.createImageData(image.width, image.height);
      for (let i = 0; i < image.data.length; i += 4) {
        invertedData.data[i]     = 255 - image.data[i];
        invertedData.data[i + 1] = 255 - image.data[i + 1];
        invertedData.data[i + 2] = 255 - image.data[i + 2];
        invertedData.data[i + 3] = image.data[i + 3];
      };
      fullCtx.putImageData(invertedData, 0, 0);
    } else {

      fullCtx.putImageData(image, 0, 0);
    }

    const imageData = fullCtx.getImageData(0, 0, canvas.width, canvas.height).data;

    let minX = fullCanvas.width, minY = fullCanvas.height;
    let maxX = 0, maxY = 0;

    for (let y = 0; y < fullCanvas.height; y++) {
      for (let x = 0; x < fullCanvas.width; x++) {
        const idx = (y * fullCanvas.width + x) * 4;
        if (imageData[idx] > 0) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }

    const boxWidth = maxX - minX + 1;
    const boxHeight = maxY - minY + 1;
    const scale = 20 / Math.max(boxWidth, boxHeight);
    const dx = (28 - boxWidth * scale) / 2;
    const dy = (28 - boxHeight * scale) / 2;

    const tmpCanvas = document.createElement("canvas");
    tmpCanvas.width = 28;
    tmpCanvas.height = 28;
    const tmpCtx = tmpCanvas.getContext("2d");

    tmpCtx.drawImage(
      fullCanvas,
      minX, minY, boxWidth, boxHeight,
      dx, dy, boxWidth * scale, boxHeight * scale
    );

    const blob = await new Promise(resolve => tmpCanvas.toBlob(resolve, "image/png"));

    const formData = new FormData();
    formData.append("file", blob, "canvas.png");
    formData.append("token", token); 

    const res = await fetch(`${host}/predict`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) throw new Error(res.statusText);

    const data = await res.json();
    message.innerText = data.match ? "Correct": "Wrong";

  } catch (err) {
    console.error(err);
    message.innerText = "Error";
  } finally {
    predictBtn.disabled = false;
  }
});

const getCanvasCoords = (event, canvas) => {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
};

canvas.addEventListener("pointerdown", event => {
  if (["mouse", "pen", "touch"].includes(event.pointerType)) {
    drawing = true;
    const { x, y } = getCanvasCoords(event, canvas);
    ctx.strokeStyle = INVERT ? "black" : "white";
    const minDim = Math.min(canvas.width, canvas.height);
    ctx.lineWidth = Math.max(1, Math.round(minDim / 18));
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(x, y);
    event.preventDefault();
  }
});

canvas.addEventListener("pointermove", event => {
  if (drawing) {
    const { x, y } = getCanvasCoords(event, canvas);
    ctx.lineTo(x, y);
    ctx.stroke();
    event.preventDefault();
  }
});

["pointerup", "pointercancel", "pointerleave"].forEach(event =>
  canvas.addEventListener(event, () => (drawing = false))
);

export const tfjs = async () => {
  if (INVERT) {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
  } else {
    ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
  }
  await setRandomLabels();
  message.innerText = "Draw the word";
};
