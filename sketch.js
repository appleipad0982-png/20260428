// ============================================
// ML5.js 0.6.1  PoseNet + Handpose
// 全螢幕畫布 + 攝影機置中 + 手指連線 + 指尖水泡
// ============================================

let video;

// PoseNet
let poseNet;
let pose;
let skeleton;
let poseNetReady = false;

// Handpose
let handpose;
let predictions = [];
let handposeReady = false;

// 攝影機顯示區域（畫布中央 50%）
let videoX, videoY, videoW, videoH;

// 水泡陣列
let bubbles = [];

// 指尖 keypoint 編號（拇指、食、中、無名、小指尖）
const FINGER_TIPS = [4, 8, 12, 16, 20];

// 5 段手指連線
const FINGER_SEGMENTS = [
  [0, 1, 2, 3, 4],     // 拇指
  [5, 6, 7, 8],        // 食指（從 5 起算掌骨關節 → 指尖）
  [9, 10, 11, 12],     // 中指
  [13, 14, 15, 16],    // 無名指
  [17, 18, 19, 20]     // 小指
];

// 系統訊息
let statusMessage = '';
let webglSupported = true;

function setup() {
  createCanvas(windowWidth, windowHeight);

  // 1) 先檢查 WebGL 是否支援
  webglSupported = checkWebGL();
  if (!webglSupported) {
    statusMessage = '⚠ 您的裝置不支援 WebGL，無法執行影像辨識';
    return;  // 直接停在這，不載模型
  }

  // 2) 計算攝影機顯示位置（畫布中央，寬高 = 畫布的 50%）
  calcVideoLayout();

  // 3) 開啟攝影機
  video = createCapture(VIDEO, () => {
    statusMessage = '攝影機已啟用,正在載入 AI 模型...';
  });
  video.size(640, 480);
  video.hide();

  // 4) 載入 PoseNet
  poseNet = ml5.poseNet(video, () => {
    poseNetReady = true;
    updateLoadingMessage();
  });
  poseNet.on('pose', poses => {
    if (poses.length > 0) {
      pose     = poses[0].pose;
      skeleton = poses[0].skeleton;
    }
  });

  // 5) 載入 Handpose
  handpose = ml5.handpose(video, () => {
    handposeReady = true;
    updateLoadingMessage();
  });
  handpose.on('predict', results => {
    predictions = results;
  });
}

// 視窗縮放時自動重算
function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  calcVideoLayout();
}

function calcVideoLayout() {
  videoW = width  * 0.5;
  videoH = height * 0.5;
  videoX = (width  - videoW) / 2;
  videoY = (height - videoH) / 2;
}

// 檢查 WebGL（ml5 / TensorFlow.js 需要）
function checkWebGL() {
  try {
    const c = document.createElement('canvas');
    const gl = c.getContext('webgl') || c.getContext('experimental-webgl');
    return !!gl;
  } catch (e) {
    return false;
  }
}

function updateLoadingMessage() {
  if (poseNetReady && handposeReady) {
    statusMessage = '✔ 模型載入完成';
    // 3 秒後清空訊息
    setTimeout(() => { statusMessage = ''; }, 3000);
  } else if (poseNetReady) {
    statusMessage = '✔ PoseNet 已載入,等待 Handpose...';
  } else if (handposeReady) {
    statusMessage = '✔ Handpose 已載入,等待 PoseNet...';
  }
}

function draw() {
  background('#e7c6ff');

  // 畫布上方標題（背景在 canvas 之外，所以這裡也畫一次以確保位置正確）
  drawTitle();

  // WebGL 失敗 → 只顯示錯誤訊息
  if (!webglSupported) {
    drawStatus();
    return;
  }

  // 模型還沒好 → 顯示載入訊息
  if (!poseNetReady || !handposeReady) {
    drawStatus();
    // 即使模型還沒好，攝影機可以先顯示
    if (video) {
      drawMirroredVideo();
    }
    return;
  }

  // 模型都就緒，正常顯示
  drawMirroredVideo();
  drawDetections();
  updateAndDrawBubbles();
  drawStatus();
}

// 畫攝影機（鏡像，置中，50% 大小）
function drawMirroredVideo() {
  push();
    // 把座標移到 video 區域的右上角，再水平翻轉
    translate(videoX + videoW, videoY);
    scale(-1, 1);
    image(video, 0, 0, videoW, videoH);
  pop();
}

// 把偵測到的座標(原始 640x480)轉成畫布上的座標(加上鏡像)
// 原始 x → 畫布 x：videoX + videoW - (rawX / videoSrcW) * videoW
function mapX(rawX) {
  return videoX + videoW - (rawX / 640) * videoW;
}
function mapY(rawY) {
  return videoY + (rawY / 480) * videoH;
}

// 繪製所有偵測結果
function drawDetections() {
  // ---- PoseNet ----
  if (pose) {
    let eyeR = pose.rightEye;
    let eyeL = pose.leftEye;
    let d = dist(eyeR.x, eyeR.y, eyeL.x, eyeL.y);
    // d 也要按比例縮小
    let scaleFactor = videoW / 640;

    noStroke();
    fill(255, 0, 0);
    ellipse(mapX(pose.nose.x), mapY(pose.nose.y), d * scaleFactor);

    fill(0, 0, 255);
    ellipse(mapX(pose.rightWrist.x), mapY(pose.rightWrist.y), 30);
    ellipse(mapX(pose.leftWrist.x),  mapY(pose.leftWrist.y),  30);

    if (pose.rightEar.confidence > 0.5) {
      fill(255, 220, 0);
      ellipse(mapX(pose.rightEar.x), mapY(pose.rightEar.y), 15, 25);
    }
    if (pose.leftEar.confidence > 0.5) {
      fill(255, 220, 0);
      ellipse(mapX(pose.leftEar.x), mapY(pose.leftEar.y), 15, 25);
    }

    drawKeypoints();
    drawSkeleton();
  }

  // ---- Handpose ----
  for (let i = 0; i < predictions.length; i++) {
    const prediction = predictions[i];

    // 左右手分色：以攝影機中線判斷
    const wristRawX = prediction.landmarks[0][0];
    const isLeftSide = wristRawX < 320;
    const handColor = isLeftSide
      ? color(255, 0, 200)   // 粉紅
      : color(255, 230, 0);  // 黃

    drawHand(prediction, handColor);

    // 五個指尖各噴水泡
    for (const tipIndex of FINGER_TIPS) {
      const [x, y] = prediction.landmarks[tipIndex];
      spawnBubble(mapX(x), mapY(y));
    }
  }
}

// 畫單一隻手：5 條手指線 + 21 個關鍵點
function drawHand(prediction, c) {
  const lm = prediction.landmarks;

  // 手指連線（用 keypoint 編號 0-4, 5-8, 9-12, 13-16, 17-20）
  stroke(255);
  strokeWeight(2);
  noFill();
  for (const segment of FINGER_SEGMENTS) {
    for (let i = 0; i < segment.length - 1; i++) {
      const a = lm[segment[i]];
      const b = lm[segment[i + 1]];
      line(mapX(a[0]), mapY(a[1]), mapX(b[0]), mapY(b[1]));
    }
  }

  // 21 個關鍵點
  noStroke();
  fill(c);
  for (let j = 0; j < lm.length; j++) {
    const [x, y] = lm[j];
    ellipse(mapX(x), mapY(y), 10, 10);
  }
}

// PoseNet 全身綠色關鍵點
function drawKeypoints() {
  noStroke();
  fill(0, 255, 0);
  for (let i = 0; i < pose.keypoints.length; i++) {
    let x = pose.keypoints[i].position.x;
    let y = pose.keypoints[i].position.y;
    ellipse(mapX(x), mapY(y), 8, 8);
  }
}

// PoseNet 紅色骨架
function drawSkeleton() {
  stroke(255, 0, 0);
  strokeWeight(2);
  for (let i = 0; i < skeleton.length; i++) {
    let a = skeleton[i][0];
    let b = skeleton[i][1];
    line(mapX(a.position.x), mapY(a.position.y),
         mapX(b.position.x), mapY(b.position.y));
  }
}

// ---------- 水泡系統 ----------
function spawnBubble(x, y) {
  bubbles.push({
    x: x + random(-3, 3),
    y: y,
    r: random(4, 10),
    vx: random(-0.5, 0.5),
    vy: random(-2.5, -1),     // 持續上升
    grow: random(0.05, 0.15), // 慢慢變大
    maxR: random(20, 35),     // 達到此大小就破掉
    life: 255
  });
}

function updateAndDrawBubbles() {
  noFill();
  strokeWeight(1.5);

  for (let i = bubbles.length - 1; i >= 0; i--) {
    const b = bubbles[i];
    b.x += b.vx;
    b.y += b.vy;
    b.r += b.grow;

    // 破掉條件:超過最大半徑、飄出畫面、或生命結束
    if (b.r >= b.maxR || b.y < 0 || b.life <= 0) {
      bubbles.splice(i, 1);
      continue;
    }

    // 接近上限時開始淡出（破掉前的視覺效果）
    if (b.r > b.maxR * 0.8) {
      b.life -= 8;
    }

    stroke(255, b.life);
    ellipse(b.x, b.y, b.r * 2);
  }
}

// ---------- 標題 ----------
function drawTitle() {
  noStroke();
  fill(0);
  textAlign(CENTER, TOP);
  textSize(32);
  textStyle(BOLD);
  text('414737055 林宇翔', width / 2, 20);
}

// ---------- 狀態訊息 ----------
function drawStatus() {
  if (!statusMessage) return;
  noStroke();
  fill(50);
  textAlign(CENTER, TOP);
  textSize(18);
  textStyle(NORMAL);
  text(statusMessage, width / 2, 70);
}