const canvas = document.getElementById('sceneCanvas');
const ctx = canvas.getContext('2d');

const MOVEMENT_KEYS = new Set(['w', 'a', 's', 'd']);

const CHEF_DIMENSIONS = {
  halfWidth: 240,
  topOffset: 176,
  bottomOffset: 212,
};

const sceneBounds = {
  minX: CHEF_DIMENSIONS.halfWidth,
  maxX: canvas.width - CHEF_DIMENSIONS.halfWidth,
  minY: CHEF_DIMENSIONS.topOffset,
  maxY: canvas.height - CHEF_DIMENSIONS.bottomOffset,
};

const chefState = {
  position: {
    x: (sceneBounds.minX + sceneBounds.maxX) / 2,
    y: sceneBounds.maxY - 10,
  },
  speed: 2.8,
  keys: {
    w: false,
    a: false,
    s: false,
    d: false,
  },
};

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function handleKeyDown(event) {
  const key = event.key.toLowerCase();
  if (!MOVEMENT_KEYS.has(key)) {
    return;
  }

  chefState.keys[key] = true;
  event.preventDefault();
}

function handleKeyUp(event) {
  const key = event.key.toLowerCase();
  if (!MOVEMENT_KEYS.has(key)) {
    return;
  }

  chefState.keys[key] = false;
  event.preventDefault();
}

function updateChefPosition() {
  const movement = { x: 0, y: 0 };

  if (chefState.keys.a) {
    movement.x -= 1;
  }
  if (chefState.keys.d) {
    movement.x += 1;
  }
  if (chefState.keys.w) {
    movement.y -= 1;
  }
  if (chefState.keys.s) {
    movement.y += 1;
  }

  if (movement.x === 0 && movement.y === 0) {
    return;
  }

  if (movement.x !== 0 && movement.y !== 0) {
    const scale = 1 / Math.sqrt(2);
    movement.x *= scale;
    movement.y *= scale;
  }

  chefState.position.x += movement.x * chefState.speed;
  chefState.position.y += movement.y * chefState.speed;

  chefState.position.x = clamp(
    chefState.position.x,
    sceneBounds.minX,
    sceneBounds.maxX,
  );
  chefState.position.y = clamp(
    chefState.position.y,
    sceneBounds.minY,
    sceneBounds.maxY,
  );
}

function drawFrame() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawBackground();
  drawChef(chefState.position);
}

function loop() {
  updateChefPosition();
  drawFrame();
  requestAnimationFrame(loop);
}

window.addEventListener('keydown', handleKeyDown);
window.addEventListener('keyup', handleKeyUp);
window.addEventListener('blur', () => {
  Object.keys(chefState.keys).forEach((key) => {
    chefState.keys[key] = false;
  });
});

function drawBackground() {
  const skyGradient = ctx.createLinearGradient(0, 0, 0, canvas.height * 0.55);
  skyGradient.addColorStop(0, '#9fd7ff');
  skyGradient.addColorStop(1, '#c8ecff');
  ctx.fillStyle = skyGradient;
  ctx.fillRect(0, 0, canvas.width, canvas.height * 0.55);

  const fieldGradient = ctx.createLinearGradient(0, canvas.height * 0.4, 0, canvas.height);
  fieldGradient.addColorStop(0, '#7bcf68');
  fieldGradient.addColorStop(0.6, '#5fae45');
  fieldGradient.addColorStop(1, '#4f903a');
  ctx.fillStyle = fieldGradient;
  ctx.fillRect(0, canvas.height * 0.45, canvas.width, canvas.height * 0.55);

  ctx.fillStyle = 'rgba(255, 255, 255, 0.25)';
  for (let i = 0; i < 4; i += 1) {
    const startX = 80 + i * 140;
    const startY = 80 + Math.sin(i) * 12;
    ctx.beginPath();
    ctx.moveTo(startX, startY);
    ctx.bezierCurveTo(startX + 40, startY - 30, startX + 120, startY - 20, startX + 160, startY);
    ctx.bezierCurveTo(startX + 120, startY + 20, startX + 40, startY + 10, startX, startY);
    ctx.fill();
  }

  ctx.lineWidth = 1.5;
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.28)';
  for (let i = 0; i < 9; i += 1) {
    const y = canvas.height * 0.55 + i * 22;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.bezierCurveTo(canvas.width * 0.35, y - 12, canvas.width * 0.65, y + 12, canvas.width, y - 6);
    ctx.stroke();
  }
}

function drawChef(position) {
  ctx.save();
  ctx.translate(position.x, position.y);

  ctx.fillStyle = 'rgba(0, 0, 0, 0.25)';
  ctx.beginPath();
  ctx.ellipse(0, 140, 120, 28, 0, 0, Math.PI * 2);
  ctx.fill();

  ctx.translate(0, -60);

  ctx.fillStyle = '#111';
  ctx.beginPath();
  ctx.arc(-90, -80, 36, 0, Math.PI * 2);
  ctx.arc(90, -80, 36, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = '#2b2b2b';
  ctx.beginPath();
  ctx.ellipse(0, -72, 180, 120, Math.PI / 2.5, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = '#f8d0b5';
  ctx.beginPath();
  ctx.arc(0, -40, 76, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = '#f1b8a1';
  ctx.beginPath();
  ctx.arc(-32, -26, 10, 0, Math.PI * 2);
  ctx.arc(32, -26, 10, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = '#231f20';
  ctx.beginPath();
  ctx.arc(-24, -44, 8, 0, Math.PI * 2);
  ctx.arc(24, -44, 8, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = '#231f20';
  ctx.lineWidth = 4;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(-18, -18);
  ctx.quadraticCurveTo(0, 0, 18, -18);
  ctx.stroke();

  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.ellipse(0, -144, 170, 60, 0, 0, Math.PI, true);
  ctx.fill();

  ctx.fillStyle = '#ffffff';
  ctx.beginPath();
  ctx.roundRect(-80, -150, 160, 36, 18);
  ctx.fill();

  ctx.fillStyle = '#e6e6e6';
  ctx.beginPath();
  ctx.roundRect(-72, -128, 144, 18, 12);
  ctx.fill();

  const bodyGradient = ctx.createLinearGradient(0, 20, 0, 220);
  bodyGradient.addColorStop(0, '#ff7fb0');
  bodyGradient.addColorStop(1, '#f15692');
  ctx.fillStyle = bodyGradient;
  ctx.beginPath();
  ctx.roundRect(-110, -10, 220, 210, 90);
  ctx.fill();

  ctx.fillStyle = '#ffd4e6';
  ctx.beginPath();
  ctx.roundRect(-110, 30, 220, 120, 80);
  ctx.fill();

  ctx.strokeStyle = '#fef6fb';
  ctx.lineWidth = 6;
  ctx.beginPath();
  ctx.moveTo(0, 40);
  ctx.lineTo(0, 190);
  ctx.stroke();

  ctx.fillStyle = '#ffd4e6';
  ctx.beginPath();
  ctx.arc(-110, 64, 42, Math.PI / 2, (Math.PI * 3) / 2, true);
  ctx.arc(110, 64, 42, -Math.PI / 2, Math.PI / 2, true);
  ctx.fill();

  ctx.save();
  ctx.translate(-140, 40);
  drawArm('left');
  ctx.restore();

  ctx.save();
  ctx.translate(140, 40);
  drawArm('right');
  ctx.restore();

  ctx.fillStyle = '#ff7fb0';
  ctx.beginPath();
  ctx.roundRect(-66, 150, 60, 110, 30);
  ctx.roundRect(6, 150, 60, 110, 30);
  ctx.fill();

  ctx.fillStyle = '#343434';
  ctx.beginPath();
  ctx.roundRect(-76, 240, 80, 32, 14);
  ctx.roundRect(-4, 240, 80, 32, 14);
  ctx.fill();

  ctx.restore();
}

function drawArm(side) {
  const direction = side === 'left' ? -1 : 1;
  const armColor = ctx.createLinearGradient(0, 0, 0, 100);
  armColor.addColorStop(0, '#ffc0da');
  armColor.addColorStop(1, '#ff90bd');
  ctx.fillStyle = armColor;

  ctx.save();
  ctx.scale(direction, 1);

  ctx.beginPath();
  ctx.roundRect(-40, -10, 120, 70, 28);
  ctx.fill();

  ctx.fillStyle = '#f8d0b5';
  ctx.beginPath();
  ctx.roundRect(56, 18, 44, 44, 18);
  ctx.fill();

  if (side === 'right') {
    drawKnife();
  }

  ctx.restore();
}

function drawKnife() {
  ctx.save();
  ctx.translate(116, 14);
  ctx.rotate(Math.PI / 12);

  ctx.fillStyle = '#5b3710';
  ctx.beginPath();
  ctx.roundRect(-6, -6, 48, 18, 8);
  ctx.fill();

  ctx.fillStyle = '#cfd4dd';
  ctx.beginPath();
  ctx.moveTo(40, -10);
  ctx.lineTo(160, -26);
  ctx.lineTo(160, 26);
  ctx.lineTo(40, 10);
  ctx.closePath();
  ctx.fill();

  ctx.strokeStyle = '#f7f7f7';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(46, -4);
  ctx.lineTo(146, -16);
  ctx.stroke();

  ctx.restore();
}

loop();
