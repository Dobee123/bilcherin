const game = document.getElementById('game');
const player = document.getElementById('player');

const speed = 10;
const state = {
  x: 180,
  y: 180,
};

const bounds = {
  width: game.clientWidth,
  height: game.clientHeight,
  player: player.clientWidth,
};

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function updatePlayerPosition() {
  const maxX = bounds.width - bounds.player;
  const maxY = bounds.height - bounds.player;

  state.x = clamp(state.x, 0, maxX);
  state.y = clamp(state.y, 0, maxY);

  player.style.transform = `translate(${state.x}px, ${state.y}px)`;
}

function handleKeydown(event) {
  const key = event.key.toLowerCase();

  switch (key) {
    case 'a':
      state.x -= speed;
      break;
    case 'd':
      state.x += speed;
      break;
    case 'w':
      state.y -= speed;
      break;
    case 's':
      state.y += speed;
      break;
    default:
      return;
  }

  event.preventDefault();
  updatePlayerPosition();
}

document.addEventListener('keydown', handleKeydown);

window.addEventListener('resize', () => {
  bounds.width = game.clientWidth;
  bounds.height = game.clientHeight;
  bounds.player = player.clientWidth;
  updatePlayerPosition();
});

updatePlayerPosition();
