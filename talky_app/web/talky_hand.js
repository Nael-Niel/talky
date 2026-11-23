import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

const video = document.createElement("video");
video.autoplay = true;
video.muted = true;
video.playsInline = true;
video.style.width = "100%";
video.style.height = "100%";
video.style.objectFit = "cover";

let landmarker = null;
let stream = null;
let lastTime = -1;
let lastJson = "";

window.talkyHand = {
  video,

  async init(modelUrl) {
    if (landmarker) return true;
    const vision = await FilesetResolver.forVisionTasks(WASM);
    const buat = (delegate) =>
      HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: modelUrl, delegate },
        runningMode: "VIDEO",
        numHands: 1,
      });
    try {
      landmarker = await buat("GPU");
    } catch (e) {
      landmarker = await buat("CPU");
    }
    return true;
  },

  async start(facing) {
    this.stop();
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: facing,
        width: { ideal: 640, max: 640 },
        height: { ideal: 480, max: 480 },
        frameRate: { ideal: 15, max: 20 },
      },
      audio: false,
    });
    video.srcObject = stream;
    video.style.transform = facing === "user" ? "scaleX(-1)" : "none";
    await video.play();
    lastTime = -1;
    lastJson = "";
    return true;
  },

  stop() {
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
    video.srcObject = null;
    lastJson = "";
  },

  detect() {
    if (!landmarker || !stream || video.readyState < 2) return "";
    if (video.currentTime === lastTime) return lastJson;
    lastTime = video.currentTime;
    const r = landmarker.detectForVideo(video, performance.now());
    if (r.landmarks && r.landmarks.length > 0) {
      const a = [];
      for (const l of r.landmarks[0]) a.push(l.x, l.y, l.z);
      lastJson = JSON.stringify(a);
    } else {
      lastJson = "";
    }
    return lastJson;
  },

  size() {
    return video.videoWidth + "x" + video.videoHeight;
  },
};