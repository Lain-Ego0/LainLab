import "./style.css";
import loadMujoco from "@mujoco/mujoco";
import mujocoWasmUrl from "@mujoco/mujoco/mujoco.wasm?url";
import * as ort from "onnxruntime-web";
import ortWasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";
import ortMjsUrl from "onnxruntime-web/ort-wasm-simd-threaded.mjs?url";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { g1JointNames, g1Policies, createG1State, g1Observation, g1Targets } from "./g1.js";

// Keep ONNX Runtime's sidecar binary out of Vite's dependency-cache URL too.
// The demo is intentionally single-threaded so it remains usable without
// cross-origin-isolation response headers.
ort.env.wasm.wasmPaths = {
  mjs: ortMjsUrl,
  wasm: ortWasmUrl,
};
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;

const policies = {
  ...g1Policies,
  handstand: {
    name: "前倒立",
    file: "/policies/go2-handstand.onnx",
    inputSize: 48,
    mode: "stand",
    defaultPose: [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5],
    commandRanges: { vx: [-0.4, 0.4], vy: [0, 0], yaw: [-0.4, 0.4] },
    note: "静止起步；可用前进和转向控制调整指令。",
  },
  rearStand: {
    name: "后立",
    file: "/policies/go2-rear-stand.onnx",
    inputSize: 45,
    mode: "stand",
    defaultPose: [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5],
    commandRanges: { vx: [-0.2, 0.6], vy: [0, 0], yaw: [-0.4, 0.4] },
    note: "前脚离地、后脚支撑的策略展示。",
  },
  trot: {
    name: "小跑",
    file: "/policies/go2-trot.onnx",
    inputSize: 470,
    mode: "gait",
    cycleTime: 0.5,
    defaultPose: [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5],
    commandRanges: { vx: [-1, 1], vy: [-1, 1], yaw: [-1, 1] },
    note: "已通过迁移回放验证的小跑策略；从低速指令开始体验。",
  },
  jump: {
    name: "原地跳跃",
    file: "/policies/go2-jump.onnx",
    inputSize: 470,
    mode: "gait",
    cycleTime: 1.5,
    defaultPose: [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5],
    commandRanges: { vx: [-1, 1], vy: [-1, 1], yaw: [-1, 1] },
    note: "已通过接触修正验证的跳跃策略；启动后会按自身节律执行。",
  },
  springJump: {
    name: "弹簧跳跃",
    file: "/policies/go2-spring-jump.onnx",
    inputSize: 470,
    mode: "spring",
    defaultPose: [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5],
    commandRanges: { vx: [0, 1], vy: [0, 0], yaw: [0, 0] },
    note: "来自 Gym 工程的弹簧跳跃策略；vx 前进指令会作为起跳输入。",
  },
  arenaWalk: {
    name: "Arena 平地行走",
    file: "/policies/go2-arena-flat.onnx",
    inputSize: 270,
    mode: "arenaHistory",
    defaultPose: [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5],
    commandRanges: { vx: [-1, 1], vy: [-1, 1], yaw: [-1, 1] },
    note: "ArenaX 参考工程的 6 帧历史平地行走策略。",
  },
  dreamwaq: {
    name: "DreamWaQ 越野步态",
    file: "/policies/go2-dreamwaq.onnx",
    inputSize: 270,
    mode: "arenaHistory",
    defaultPose: [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5],
    commandRanges: { vx: [-1, 1], vy: [-1, 1], yaw: [-1, 1] },
    note: "DreamWaQ 发布导出的 6 帧历史越野步态策略。",
  },
  ampCts: {
    name: "AMP-CTS 步态",
    file: "/policies/go2-amp-cts.onnx",
    inputSize: 270,
    mode: "arenaHistory",
    defaultPose: [0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5],
    commandRanges: { vx: [-1, 1], vy: [-1, 1], yaw: [-1, 1] },
    note: "AMP-CTS 发布导出的 6 帧历史行走策略。",
  },
};

const go2JointNames = [
  "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
  "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
  "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
  "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
];
const go2ActuatorNames = [
  "FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf",
  "RL_hip", "RL_thigh", "RL_calf", "RR_hip", "RR_thigh", "RR_calf",
];
const robots = {
  go2: { name: "Unitree Go2", scene: "robot/scene_go2.xml", jointNames: go2JointNames, actuatorNames: go2ActuatorNames,
    height: .42, camera: [1.6, -2.7, 1.25], target: .38, defaultPolicy: "handstand" },
  g1: { name: "Unitree G1 · 29 DOF", scene: "robot/g1/scene.xml", jointNames: g1JointNames,
    actuatorNames: g1JointNames.map((name) => name.replace(/_joint$/, "")),
    height: .793, camera: [2.4, -3.6, 1.8], target: .8, defaultPolicy: "g1Amp" },
};
let activeRobotKey = "go2", activePolicyKey = "handstand", selectionBusy = false;
let g1State = createG1State();
const robotCache = new Map();

const $ = (id) => document.getElementById(id);
const publicAsset = (path) => `${import.meta.env.BASE_URL}${path.replace(/^\/+/, "")}`;
const engineState = $("engineState");
const simulation = {
  running: false,
  elapsed: 0,
  realTimeAccumulator: 0,
  action: new Float32Array(12),
  actionHistory: [],
  command: [0, 0, 0],
  history: [],
};
const MUJOCO_TIMESTEP = 0.002;
const POLICY_PHYSICS_STEPS = 10;
const POLICY_DT = MUJOCO_TIMESTEP * POLICY_PHYSICS_STEPS;
const terrainState = { kind: "flat", seed: 7, height: 0.28, tool: "platform", elements: [], selected: null };
const TERRAIN_AREA = { x: 10, y: 8 };
const TERRAIN_SLOT_COUNT = 96;
let mujoco, model, data, policySession, activePolicy, jointAddresses, actuatorAddresses, baseSceneXml, robotVfs, terrainSlotIds, terrainSlotSet;
let physicsStepsPerPolicy = POLICY_PHYSICS_STEPS;
let actualPolicyDt = POLICY_DT;
const controllerState = {
  commandSpeed: [1, 1, 1],
  keyboardKeys: new Set(),
  joystick: { left: { x: 0, y: 0 }, right: { x: 0, y: 0 } },
};

function setStatus(text, kind = "") {
  engineState.textContent = text;
  engineState.className = `pill ${kind}`;
}

function setNotice(text, error = false) {
  if (error) console.warn(text);
}

function setBootStage(text, progress) {
  $("bootStage").textContent = text;
  $("bootProgress").style.width = `${progress}%`;
  $("bootPercent").textContent = `${progress}%`;
}

function revealScene() {
  $("bootScreen").classList.add("reveal");
}

function createScene() {
  const mount = $("scene");
  const scene = new THREE.Scene();
  scene.background = new THREE.Color("#edf0f3");
  const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 100);
  camera.up.set(0, 0, 1);
  camera.position.set(1.6, -2.7, 1.25);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const canvasMount = document.createElement("div");
  canvasMount.className = "scene-canvas";
  canvasMount.append(renderer.domElement);
  mount.prepend(canvasMount);
  const orbit = new OrbitControls(camera, renderer.domElement);
  orbit.target.set(0, 0, 0.38);
  orbit.enableDamping = true;
  orbit.dampingFactor = 0.18;
  orbit.rotateSpeed = 0.48;
  orbit.panSpeed = 0.55;
  orbit.zoomSpeed = 0.72;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x98a5b1, 2.2));
  const key = new THREE.DirectionalLight(0xffffff, 2.4);
  key.position.set(2, -3, 5);
  key.castShadow = true;
  scene.add(key);
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(TERRAIN_AREA.x * 2.4, TERRAIN_AREA.y * 2.4),
    new THREE.MeshStandardMaterial({ color: 0xdfe4e8, roughness: 0.92 }),
  );
  floor.receiveShadow = true;
  floor.position.z = -0.002;
  scene.add(floor);
  const grid = new THREE.GridHelper(TERRAIN_AREA.x * 2, 40, 0xb7c1ca, 0xd7dde2);
  // GridHelper is horizontal in Three.js' Y-up world. This scene and MuJoCo
  // model use Z-up, so rotate it onto the physical floor plane.
  grid.rotation.x = Math.PI / 2;
  grid.position.z = 0.001;
  scene.add(grid);
  const resize = () => {
    const width = mount.clientWidth;
    const height = mount.clientHeight;
    if (!width || !height) return;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  };
  new ResizeObserver(resize).observe(mount);
  resize();
  return { scene, camera, renderer, orbit, floor, grid };
}

const view = createScene();
const terrainVisual = new THREE.Group();
terrainVisual.name = "playground-terrain";
view.scene.add(terrainVisual);
const robot = new THREE.Group();
view.scene.add(robot);
const bodyNodes = new Map();
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const dragPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
const dragArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 0, 0x1e75ae, .13, .07);
dragArrow.visible = false;
view.scene.add(dragArrow);
let robotDrag = null;

function attribute(element, name, fallback = "") {
  return element.getAttribute(name) ?? fallback;
}

function vector(text, size, fallback) {
  const values = text.trim().split(/\s+/).map(Number);
  return values.length === size && values.every(Number.isFinite) ? values : fallback;
}

function localTransform(node, element) {
  const pos = vector(attribute(element, "pos"), 3, [0, 0, 0]);
  const quat = vector(attribute(element, "quat"), 4, [1, 0, 0, 0]);
  node.position.fromArray(pos);
  node.quaternion.set(quat[1], quat[2], quat[3], quat[0]);
}

async function fetchAsset(path, format = "text") {
  const response = await fetch(publicAsset(path));
  if (!response.ok) throw new Error(`资源加载失败 (${response.status}): ${path}`);
  if (response.headers.get("content-type")?.includes("text/html")) {
    throw new Error(`资源 ${path} 返回了网页而非文件，请刷新页面或重启试玩服务`);
  }
  return response[format]();
}

async function robotPackage(key) {
  if (!robotCache.has(key)) {
    const task = (async () => {
      const scenePath = robots[key].scene;
      const xml = await fetchAsset(scenePath);
      const document = new DOMParser().parseFromString(xml, "text/xml");
      if (document.querySelector("parsererror") || document.documentElement.tagName !== "mujoco") {
        throw new Error(`机器人场景 XML 无效：${scenePath}`);
      }
      const directory = scenePath.slice(0, scenePath.lastIndexOf("/") + 1);
      const meshDir = document.querySelector("compiler")?.getAttribute("meshdir") || "";
      const meshes = await Promise.all([...document.querySelectorAll("asset > mesh")].map(async (element) => {
        const file = [meshDir === "." ? "" : meshDir, attribute(element, "file")].filter(Boolean).join("/");
        return { name: attribute(element, "name", file.split("/").at(-1).replace(/\.[^.]+$/, "")), file, scale: vector(attribute(element, "scale"), 3, [1, 1, 1]),
          bytes: await fetchAsset(directory + file, "arrayBuffer") };
      }));
      return { xml, meshes };
    })();
    robotCache.set(key, task);
    task.catch(() => robotCache.delete(key));
  }
  return robotCache.get(key);
}

function disposeRobot(group) {
  const geometries = new Set(), materials = new Set();
  group.traverse((node) => { if (node.isMesh) { geometries.add(node.geometry); materials.add(node.material); } });
  geometries.forEach((geometry) => geometry.dispose());
  materials.forEach((material) => material.dispose());
  group.clear();
}

function buildRobot(asset) {
  const meshes = new Map(asset.meshes.map(({ name, file, bytes, scale }) => {
    const object = /\.stl$/i.test(file) ? new THREE.Mesh(new STLLoader().parse(bytes))
      : new OBJLoader().parse(new TextDecoder().decode(bytes));
    object.scale.fromArray(scale);
    return [name, object];
  }));
  const document = new DOMParser().parseFromString(asset.xml, "text/xml");
  const materialColors = { metal: 0xe0e8e8, black: 0x1c242c, white: 0xf2f4f5, gray: 0xabb1c5 };
  const group = new THREE.Group(), nodes = new Map();
  function visit(body) {
    const name = attribute(body, "name");
    if (!name) return;
    const node = new THREE.Group();
    nodes.set(name, node);
    group.add(node);
    for (const geom of body.children) {
      if (geom.tagName !== "geom" || !geom.hasAttribute("mesh")) continue;
      // G1 carries separate visual and collision meshes at identical transforms.
      if (geom.getAttribute("contype") !== "0" && [...body.children].some((other) =>
        other !== geom && other.tagName === "geom" && other.getAttribute("mesh") === geom.getAttribute("mesh")
        && other.getAttribute("contype") === "0")) continue;
      const source = meshes.get(attribute(geom, "mesh"));
      if (!source) throw new Error(`缺少机器人网格：${attribute(geom, "mesh")}`);
      const mesh = source.clone(true);
      const rgba = vector(attribute(geom, "rgba"), 4, null);
      mesh.traverse((child) => {
        if (!child.isMesh) return;
        child.material = new THREE.MeshStandardMaterial({
          color: rgba ? new THREE.Color(rgba[0], rgba[1], rgba[2]) : materialColors[attribute(geom, "material")] ?? 0xb8c0c8,
          roughness: .62, metalness: .12,
        });
        child.castShadow = true;
      });
      localTransform(mesh, geom);
      node.add(mesh);
    }
    for (const child of body.children) if (child.tagName === "body") visit(child);
  }
  for (const body of document.querySelectorAll("worldbody > body")) visit(body);
  // The clones own the shared geometries; loader-created materials are unused.
  meshes.forEach((object) => object.traverse((child) => {
    if (child.isMesh) for (const material of [child.material].flat()) material.dispose();
  }));
  return { group, nodes };
}

async function initializeMujoco() {
  mujoco = await loadMujoco({ locateFile: (file) => (file === "mujoco.wasm" ? mujocoWasmUrl : file) });
}

function createMujocoModel(xml, terrainCount, vfs = robotVfs, config = robots[activeRobotKey]) {
  const nextModel = mujoco.MjModel.from_xml_string(xml, vfs);
  let nextData;
  try {
    nextData = new mujoco.MjData(nextModel);
    const ratio = POLICY_DT / nextModel.opt.timestep;
    if (Math.abs(nextModel.opt.timestep - MUJOCO_TIMESTEP) > 1e-9 || !Number.isInteger(ratio) || ratio !== POLICY_PHYSICS_STEPS) {
      throw new Error(`MuJoCo timestep 必须为 ${MUJOCO_TIMESTEP}s，策略周期必须由 ${POLICY_PHYSICS_STEPS} 个物理步组成。`);
    }
    const joints = config.jointNames.map((name) => ({ qpos: nextModel.jnt(name).qposadr, dof: nextModel.jnt(name).dofadr }));
    const actuators = config.actuatorNames.map((name) => nextModel.actuator(name).id);
    const slots = Array.from({ length: terrainCount }, (_, i) => nextModel.geom(`playground_terrain_${i}`).id);
    data?.delete?.();
    model?.delete?.();
    model = nextModel; data = nextData;
    jointAddresses = joints; actuatorAddresses = actuators;
    terrainSlotIds = slots; terrainSlotSet = new Set(slots);
    physicsStepsPerPolicy = ratio;
    actualPolicyDt = ratio * model.opt.timestep;
  } catch (error) {
    nextData?.delete?.(); nextModel.delete(); throw error;
  }
}

function seededRandom(seed) {
  let value = (Number(seed) || 1) >>> 0;
  return () => {
    value += 0x6d2b79f5;
    let result = value;
    result = Math.imul(result ^ (result >>> 15), result | 1);
    result ^= result + Math.imul(result ^ (result >>> 7), result | 61);
    return ((result ^ (result >>> 14)) >>> 0) / 4294967296;
  };
}

function box(x, y, z, sx, sy, sz, yaw = 0, label = "障碍") {
  return { x, y, z, sx, sy, sz, yaw, label };
}

// A closed wedge: the low edge rests on z=0 and the high edge reaches height.
const rampFaces = [0, 1, 2, 3, 5, 4, 0, 3, 4, 0, 4, 1, 1, 4, 5, 1, 5, 2, 0, 2, 5, 0, 5, 3];

function ramp(x, y, height, sx, sy, yaw = 0, label = "斜坡") {
  return { type: "ramp", x, y, z: height / 2, sx, sy, sz: height / 2, yaw, label };
}

function rampVertices({ sx, sy, sz }) {
  return [
    -sx, -sy, -sz, sx, -sy, -sz, sx, -sy, sz,
    -sx, sy, -sz, sx, sy, -sz, sx, sy, sz,
  ];
}

function profileBoxes() {
  const height = terrainState.height;
  if (terrainState.kind === "slope") return [ramp(3.8, 0, height, 3.2, 3.2, 0, "坡道")];
  if (terrainState.kind === "stairs") return Array.from({ length: 7 }, (_, index) => {
    const step = index + 1;
    return box(2 + index * 0.62, 0, height * step / 14, 0.31, 1.8, height * step / 14, 0, "阶梯");
  });
  if (terrainState.kind === "obstacle_mix") {
    const random = seededRandom(terrainState.seed);
    const boxes = [];
    while (boxes.length < 10) {
      const x = -TERRAIN_AREA.x + .8 + random() * (TERRAIN_AREA.x * 2 - 1.6);
      const y = -TERRAIN_AREA.y + .8 + random() * (TERRAIN_AREA.y * 2 - 1.6);
      if (Math.hypot(x, y) < 1.05) continue;
      boxes.push(box(x, y, height * (0.35 + random() * 0.45) / 2, 0.18 + random() * 0.24, 0.18 + random() * 0.24, height * (0.35 + random() * 0.45) / 2, 0, "随机障碍"));
    }
    return boxes;
  }
  return [];
}

function elementBoxes(element) {
  const { x, y, kind } = element;
  const height = Number.isFinite(element.height) ? element.height : terrainState.height;
  const length = element.length ?? 1.8;
  const width = element.width ?? 1.8;
  const yaw = element.yaw || 0;
  const placedBox = (localX, localY, z, sx, sy, sz, label) => {
    const rotatedX = localX * Math.cos(yaw) - localY * Math.sin(yaw);
    const rotatedY = localX * Math.sin(yaw) + localY * Math.cos(yaw);
    return box(x + rotatedX, y + rotatedY, z, sx, sy, sz, yaw, label);
  };
  if (kind === "stairs") {
    const count = element.stepCount ?? 5, depth = element.stepDepth ?? .28;
    const stepHeight = element.stepHeight ?? .12, stepWidth = element.stepWidth ?? 1.6;
    return Array.from({ length: count }, (_, index) => placedBox(
      (index - (count - 1) / 2) * depth, 0, stepHeight * (index + 1) / 2,
      depth / 2, stepWidth / 2, stepHeight * (index + 1) / 2, "台阶"));
  }
  if (kind === "ramp") return [ramp(x, y, height, length / 2, width / 2, yaw)];
  if (kind === "stones") {
    const columns = element.columns ?? 3, rows = element.rows ?? 2;
    const gapX = element.gapX ?? .14, gapY = element.gapY ?? .32;
    const pitchX = length + gapX, pitchY = width + gapY;
    return Array.from({ length: columns * rows }, (_, index) => {
      const col = index % columns, row = Math.floor(index / columns);
      const offset = element.layout === "staggered" && row % 2 ? pitchX / 2
        : element.layout === "diagonal" ? (row - (rows - 1) / 2) * pitchX / 2 : 0;
      return placedBox((col - (columns - 1) / 2) * pitchX + offset, (row - (rows - 1) / 2) * pitchY,
        height / 2, length / 2, width / 2, height / 2, "梅花桩");
    });
  }
  return [placedBox(0, 0, height / 2, length / 2, width / 2, height / 2, kind === "wall" ? "矮墙" : "高台")];
}

function terrainBoxes() {
  // The Three.js floor is visual only. This explicit MuJoCo box prevents the
  // robot from falling through a flat area or gaps between placed obstacles.
  return [box(0, 0, -0.08, TERRAIN_AREA.x, TERRAIN_AREA.y, .08, 0, "场地地面"), ...profileBoxes(), ...terrainState.elements.flatMap(elementBoxes)];
}

function clearTerrainVisual() {
  terrainVisual.traverse((node) => {
    if (node.isMesh) {
      node.geometry.dispose();
      node.material.dispose();
    }
  });
  terrainVisual.clear();
}

function addVisualBox(definition, accent = 0x7f9bb0) {
  let geometry;
  if (definition.type === "ramp") {
    geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.Float32BufferAttribute(rampVertices(definition), 3));
    geometry.setIndex(rampFaces);
    geometry = geometry.toNonIndexed();
    geometry.computeVertexNormals();
  } else geometry = new THREE.BoxGeometry(definition.sx * 2, definition.sy * 2, definition.sz * 2);
  const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color: accent, roughness: .76, metalness: .03 }));
  mesh.position.set(definition.x, definition.y, definition.z);
  mesh.rotation.z = definition.yaw || 0;
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  terrainVisual.add(mesh);
}

function renderTerrain() {
  clearTerrainVisual();
  view.floor.material.color.set(terrainState.kind === "flat" ? 0xdfe4e8 : 0xd7e1e6);
  for (const definition of profileBoxes()) addVisualBox(definition, 0x9aabba);
  for (const element of terrainState.elements) for (const definition of elementBoxes(element)) addVisualBox(definition, 0x6f94ae);
}

function terrainXml(xml = baseSceneXml) {
  const document = new DOMParser().parseFromString(xml, "text/xml");
  const worldbodies = document.querySelectorAll("worldbody");
  const worldbody = worldbodies[worldbodies.length - 1];
  // scene_go2.xml already contains an infinite plane. Keep one explicit box
  // floor for the editor's bounded playground, but remove the original plane
  // so coincident contacts do not make the foot solver fight two surfaces.
  worldbody.querySelector('geom[name="floor"]')?.remove();
  // The visual foot mesh extends about 26 mm below its attachment point,
  // while the source collision sphere is only 22 mm and uses a soft contact
  // response. Slightly enlarge and stiffen the four foot geoms so the visible
  // sole is covered by collision and does not repeatedly pass below z=0.
  ["FL", "FR", "RL", "RR"].forEach((name) => {
    const foot = document.querySelector(`geom[name="${name}"]`);
    foot?.setAttribute("size", "0.026");
    foot?.setAttribute("solref", "0.005 1");
    foot?.setAttribute("solimp", "0.95 0.99 0.001");
  });
  const definitions = terrainBoxes();
  if (definitions.length > TERRAIN_SLOT_COUNT) throw new Error(`地形最多支持 ${TERRAIN_SLOT_COUNT} 个碰撞组件。`);
  definitions.forEach((definition, index) => {
    const isRamp = definition.type === "ramp";
    if (isRamp) {
      const asset = document.querySelector("asset") || document.documentElement.insertBefore(document.createElement("asset"), worldbody);
      const mesh = document.createElement("mesh");
      mesh.setAttribute("name", `playground_ramp_${index}`);
      mesh.setAttribute("vertex", rampVertices(definition).join(" "));
      mesh.setAttribute("face", rampFaces.join(" "));
      asset.append(mesh);
    }
    const geom = document.createElement("geom");
    geom.setAttribute("name", `playground_terrain_${index}`);
    geom.setAttribute("type", isRamp ? "mesh" : "box");
    geom.setAttribute("pos", `${definition.x} ${definition.y} ${definition.z}`);
    if (isRamp) {
      geom.setAttribute("mesh", `playground_ramp_${index}`);
      geom.setAttribute("euler", `0 0 ${definition.yaw}`);
    } else {
      geom.setAttribute("size", `${definition.sx} ${definition.sy} ${definition.sz}`);
      geom.setAttribute("euler", `0 0 ${definition.yaw || 0}`);
    }
    geom.setAttribute("rgba", "0 0 0 0");
    geom.setAttribute("friction", "0.9 0.1 0.1");
    // Match MuJoCo's stable default contact response instead of the very
    // lightly damped direct-format values that previously allowed foot sink.
    geom.setAttribute("solref", "0.02 1");
    geom.setAttribute("solimp", "0.9 0.95 0.001");
    geom.setAttribute("contype", index < definitions.length ? "1" : "0");
    geom.setAttribute("conaffinity", index < definitions.length ? "1" : "0");
    worldbody.append(geom);
  });
  return new XMLSerializer().serializeToString(document);
}

async function applyTerrain() {
  if (!mujoco || !baseSceneXml || selectionBusy) return;
  setSelectionBusy(true);
  await stopSimulation();
  simulation.running = false;
  $("start").textContent = "开始";
  const definitions = terrainBoxes();
  $("terrainApply").disabled = true;
  setStatus("正在同步地形碰撞…");
  setNotice("正在同步地形碰撞…");
  try {
    if (definitions.length > TERRAIN_SLOT_COUNT) throw new Error(`地形最多支持 ${TERRAIN_SLOT_COUNT} 个碰撞组件。`);
    // Geometry topology is part of MuJoCo's collision broad-phase. Compile
    // only active geoms, while keeping robot mesh assets cached in memory.
    await new Promise((resolve) => requestAnimationFrame(resolve));
    createMujocoModel(terrainXml(), definitions.length);
    renderTerrain();
    reset();
    setStatus("浏览器物理引擎已就绪", "ready");
    setNotice(`已应用${terrainState.kind === "flat" ? "平地" : "自定义"}地形；物理碰撞已同步。`);
  } finally {
    setSelectionBusy(false);
  }
}

function setSelectionBusy(busy) {
  selectionBusy = busy;
  for (const id of ["robotModel", "policy", "start", "reset", "terrainApply"]) $(id).disabled = busy;
}

async function stopSimulation() {
  simulation.running = false;
  $("start").textContent = "开始";
  // Inference yields to the UI. Finish that step before resetting/deleting data.
  while (pendingStep) await new Promise((resolve) => requestAnimationFrame(resolve));
}

function populatePolicies(robotKey, selected) {
  $("policy").replaceChildren(...Object.entries(policies)
    .filter(([, policy]) => (policy.robot || "go2") === robotKey)
    .map(([key, policy]) => new Option(policy.name, key, false, key === selected)));
}

async function loadSelection(robotKey, policyKey = robots[robotKey].defaultPolicy) {
  if (selectionBusy) return;
  setSelectionBusy(true);
  await stopSimulation();
  const nextPolicy = policies[policyKey];
  let session, visual, vfs;
  try {
    if (!nextPolicy || (nextPolicy.robot || "go2") !== robotKey) throw new Error("机器人与策略不匹配");
    setStatus(`正在加载 ${robots[robotKey].name}…`);
    const changeRobot = !model || robotKey !== activeRobotKey;
    session = await ort.InferenceSession.create(publicAsset(nextPolicy.file), { executionProviders: ["wasm"] });
    const metadata = session.inputMetadata[0];
    if (metadata.shape.at(-1) !== nextPolicy.inputSize || session.outputMetadata[0].shape.at(-1) !== robots[robotKey].jointNames.length) {
      throw new Error("策略输入输出维度与机器人配置不匹配");
    }
    if (changeRobot) {
      const asset = await robotPackage(robotKey);
      visual = buildRobot(asset);
      vfs = new mujoco.MjVFS();
      for (const mesh of asset.meshes) vfs.addBuffer(mesh.file, new Uint8Array(mesh.bytes));
      createMujocoModel(terrainXml(asset.xml), terrainBoxes().length, vfs, robots[robotKey]);
      robotVfs?.delete?.(); robotVfs = vfs; vfs = null;
      baseSceneXml = asset.xml;
      disposeRobot(robot);
      robot.add(visual.group);
      bodyNodes.clear();
      visual.nodes.forEach((node, name) => bodyNodes.set(name, node));
      visual = null;
      view.camera.position.fromArray(robots[robotKey].camera);
      view.orbit.target.set(0, 0, robots[robotKey].target);
    }
    const previous = policySession;
    policySession = session; session = null;
    activeRobotKey = robotKey; activePolicyKey = policyKey;
    activePolicy = nextPolicy;
    simulation.action = new Float32Array(robots[robotKey].jointNames.length);
    controllerState.keyboardKeys.clear(); resetJoysticks(); simulation.command.fill(0);
    configureCommandControls(); recomputeCommand();
    $("policyName").textContent = activePolicy.name;
    $("policyMeta").textContent = `${activePolicy.inputSize} 维策略输入 · ${simulation.action.length} 个关节动作`;
    $("policyMeta").title = activePolicy.note;
    $("scene").setAttribute("aria-label", `${robots[robotKey].name} MuJoCo 策略试玩画面`);
    $("actionValues").setAttribute("aria-label", `当前 ${simulation.action.length} 维动作输出`);
    reset(); renderTerrain(); drawActionChart();
    setStatus("浏览器物理引擎已就绪", "ready");
    if (previous) await previous.release();
  } catch (error) {
    if (session) await session.release();
    if (visual) disposeRobot(visual.group);
    vfs?.delete?.();
    setStatus(`加载失败：${error.message || error}`, "error");
    throw error;
  } finally {
    $("robotModel").value = activeRobotKey;
    populatePolicies(activeRobotKey, activePolicyKey);
    setSelectionBusy(false);
    $("start").disabled = $("reset").disabled = !policySession;
  }
}

function reset() {
  if (!model || !data || !activePolicy) return;
  mujoco.mj_resetData(model, data);
  data.qpos[0] = 0;
  data.qpos[1] = 0;
  data.qpos[2] = robots[activeRobotKey].height;
  data.qpos[3] = 1;
  data.qpos[4] = data.qpos[5] = data.qpos[6] = 0;
  activePolicy.defaultPose.forEach((value, index) => { data.qpos[jointAddresses[index].qpos] = value; });
  data.qvel.fill(0);
  data.ctrl.fill(0);
  simulation.action.fill(0);
  simulation.actionHistory = [];
  simulation.history = [];
  g1State = createG1State();
  robotDrag = null;
  dragArrow.visible = false;
  view.orbit.enabled = true;
  simulation.elapsed = 0;
  simulation.realTimeAccumulator = 0;
  mujoco.mj_forward(model, data);
  updateRobot();
}

function bodyQuaternion() {
  return new THREE.Quaternion(data.qpos[4], data.qpos[5], data.qpos[6], data.qpos[3]);
}

function observation() {
  if (activePolicy.robot === "g1") {
    const gravity = new THREE.Vector3(0, 0, -1).applyQuaternion(bodyQuaternion().invert());
    return g1Observation(activePolicy, g1State, {
      q: jointAddresses.map((a) => data.qpos[a.qpos]), dq: jointAddresses.map((a) => data.qvel[a.dof]),
      gyro: [data.qvel[3], data.qvel[4], data.qvel[5]], gravity: gravity.toArray(),
      command: simulation.command, action: simulation.action,
    });
  }
  if (activePolicy.mode === "gait" || activePolicy.mode === "spring") return gaitObservation();
  if (activePolicy.mode === "arenaHistory") return arenaHistoryObservation();
  const quaternion = bodyQuaternion();
  const inverse = quaternion.clone().invert();
  const angularVelocity = new THREE.Vector3(data.qvel[3], data.qvel[4], data.qvel[5]);
  const gravity = new THREE.Vector3(0, 0, -1).applyQuaternion(inverse);
  const values = activePolicy.inputSize === 48 ? [0, 0, 0] : [];
  values.push(angularVelocity.x * 0.25, angularVelocity.y * 0.25, angularVelocity.z * 0.25);
  values.push(gravity.x, gravity.y, gravity.z);
  values.push(simulation.command[0] * 2, simulation.command[1] * 2, simulation.command[2] * 0.25);
  jointAddresses.forEach((address, index) => values.push(data.qpos[address.qpos] - activePolicy.defaultPose[index]));
  jointAddresses.forEach((address) => values.push(data.qvel[address.dof] * 0.05));
  values.push(...simulation.action);
  return new Float32Array(values);
}

function arenaHistoryObservation() {
  const quaternion = bodyQuaternion();
  const inverse = quaternion.clone().invert();
  const angularVelocity = new THREE.Vector3(data.qvel[3], data.qvel[4], data.qvel[5]);
  const gravity = new THREE.Vector3(0, 0, -1).applyQuaternion(inverse);
  const frame = [
    simulation.command[0] * 2, simulation.command[1] * 2, simulation.command[2] * .25,
    angularVelocity.x * .25, angularVelocity.y * .25, angularVelocity.z * .25,
    gravity.x, gravity.y, gravity.z,
  ];
  jointAddresses.forEach((address, index) => frame.push(data.qpos[address.qpos] - activePolicy.defaultPose[index]));
  jointAddresses.forEach((address) => frame.push(data.qvel[address.dof] * .05));
  frame.push(...simulation.action);
  simulation.history.push(frame);
  if (simulation.history.length > 6) simulation.history.shift();
  const values = Array.from({ length: 6 - simulation.history.length }, () => new Array(45).fill(0));
  values.push(...simulation.history);
  return new Float32Array(values.flat());
}

function gaitObservation() {
  const quaternion = bodyQuaternion();
  const angularVelocity = new THREE.Vector3(data.qvel[3], data.qvel[4], data.qvel[5]);
  const euler = new THREE.Euler().setFromQuaternion(quaternion, "XYZ");
  const frame = activePolicy.mode === "spring"
    // The exported 470-D checkpoint predates the 45-D mjlab migration:
    // legacy frame = zeros(2) + [0.7, 0, vx] + ang_vel + euler + q/dq/action.
    ? [0, 0, 0.7, 0, simulation.command[0], angularVelocity.x * 0.25, angularVelocity.y * 0.25, angularVelocity.z * 0.25, euler.x, euler.y, euler.z]
    : (() => {
      const phase = simulation.elapsed / activePolicy.cycleTime;
      return [
        Math.sin(2 * Math.PI * phase), Math.cos(2 * Math.PI * phase),
        simulation.command[0] * 2, simulation.command[1] * 2, simulation.command[2] * 0.25,
        angularVelocity.x * 0.25, angularVelocity.y * 0.25, angularVelocity.z * 0.25,
        euler.x, euler.y, euler.z,
      ];
    })();
  jointAddresses.forEach((address, index) => frame.push(data.qpos[address.qpos] - activePolicy.defaultPose[index]));
  jointAddresses.forEach((address) => frame.push(data.qvel[address.dof] * 0.05));
  frame.push(...simulation.action);
  simulation.history.push(frame);
  if (simulation.history.length > 10) simulation.history.shift();
  const values = Array.from({ length: 10 - simulation.history.length }, () => new Array(47).fill(0));
  values.push(...simulation.history);
  return new Float32Array(values.flat());
}

async function policyStep() {
  const obs = observation();
  if (obs.length !== activePolicy.inputSize) throw new Error(`策略观测维度不匹配：${obs.length} / ${activePolicy.inputSize}`);
  const inputName = policySession.inputNames[0];
  const outputName = policySession.outputNames[0];
  const feeds = { [inputName]: new ort.Tensor("float32", obs, [1, obs.length]) };
  let output;
  try {
    output = await policySession.run(feeds);
    const actions = output[outputName].data;
    if (actions.length !== simulation.action.length || !actions.every(Number.isFinite)) throw new Error("策略输出无效");
    simulation.action.set(activePolicy.robot === "g1" ? Float32Array.from(actions, (v) => THREE.MathUtils.clamp(v, -100, 100)) : actions);
  } finally {
    Object.values(feeds).forEach((tensor) => tensor.dispose());
    Object.values(output || {}).forEach((tensor) => tensor.dispose());
  }
  simulation.actionHistory.push(Array.from(simulation.action));
  if (simulation.actionHistory.length > 90) simulation.actionHistory.shift();
}

function applyControl() {
  if (activePolicy.robot === "g1") {
    const targets = g1Targets(activePolicy, simulation.action);
    jointAddresses.forEach((address, index) => {
      const actuator = actuatorAddresses[index];
      const torque = activePolicy.kp[index] * (targets[index] - data.qpos[address.qpos]) - activePolicy.kd[index] * data.qvel[address.dof];
      data.ctrl[actuator] = THREE.MathUtils.clamp(torque, model.actuator_ctrlrange[actuator * 2], model.actuator_ctrlrange[actuator * 2 + 1]);
    });
    return;
  }
  jointAddresses.forEach((address, index) => {
    const target = activePolicy.defaultPose[index] + 0.25 * simulation.action[index];
    const limit = index % 3 === 2 ? 31.995 : 21.33;
    const torque = THREE.MathUtils.clamp(40 * (target - data.qpos[address.qpos]) - data.qvel[address.dof], -limit, limit);
    data.ctrl[actuatorAddresses[index]] = torque;
  });
}

function updateRobot() {
  if (!model || !data) return;
  for (const [name, node] of bodyNodes) {
    const id = model.body(name).id;
    node.position.fromArray(data.xpos, id * 3);
    const matrix = data.xmat;
    const offset = id * 9;
    const rotation = new THREE.Matrix4().set(
      matrix[offset], matrix[offset + 1], matrix[offset + 2], 0,
      matrix[offset + 3], matrix[offset + 4], matrix[offset + 5], 0,
      matrix[offset + 6], matrix[offset + 7], matrix[offset + 8], 0,
      0, 0, 0, 1,
    );
    node.quaternion.setFromRotationMatrix(rotation);
  }
}

function updateControls() {
  const [vx, vy, yaw] = simulation.command;
  $("command").textContent = `vx ${vx.toFixed(2)} · vy ${vy.toFixed(2)} · ω ${yaw.toFixed(2)}`;
  $("commandVxValue").textContent = `${vx.toFixed(2)} m/s`;
  $("commandVyValue").textContent = `${vy.toFixed(2)} m/s`;
  $("commandYawValue").textContent = `${yaw.toFixed(2)} rad/s`;
  ["commandSpeedVxValue", "commandSpeedVyValue", "commandSpeedYawValue"].forEach((id, index) => {
    const unit = index === 2 ? "rad/s" : "m/s";
    $(id).textContent = `${controllerState.commandSpeed[index].toFixed(2)} ${unit}`;
  });
  const keyboardActive = controllerState.keyboardKeys.size > 0;
  const joystickActive = Math.abs(controllerState.joystick.left.x) > .001 || Math.abs(controllerState.joystick.left.y) > .001 || Math.abs(controllerState.joystick.right.x) > .001;
  $("controlSource").textContent = keyboardActive ? "键盘 · 满幅" : joystickActive ? "摇杆 · 线性" : "待命";
  syncJoystickVisuals();
  $("leftJoystickValue").textContent = `vx ${vx.toFixed(2)} · vy ${vy.toFixed(2)}`;
  $("rightJoystickValue").textContent = `ω ${yaw.toFixed(2)}`;
}

function configureCommandControls() {
  if (!activePolicy) return;
  const controls = [
    ["commandSpeedVx", activePolicy.commandRanges.vx],
    ["commandSpeedVy", activePolicy.commandRanges.vy],
    ["commandSpeedYaw", activePolicy.commandRanges.yaw],
  ];
  controls.forEach(([id, [min, max]], index) => {
    const input = $(id);
    const maxSpeed = Math.max(Math.abs(min), Math.abs(max));
    input.min = "0";
    input.max = String(maxSpeed || 1);
    input.value = String(maxSpeed);
    input.disabled = maxSpeed === 0;
    input.setAttribute("aria-valuetext", `0 到 ${maxSpeed}`);
    controllerState.commandSpeed[index] = maxSpeed;
  });
}

function keyboardAxis(index) {
  const keys = controllerState.keyboardKeys;
  if (index === 0) {
    if (keys.has("w")) return 1;
    if (keys.has("s")) return -1;
  } else if (index === 1) {
    if (keys.has("a")) return 1;
    if (keys.has("d")) return -1;
  } else {
    if (keys.has("q")) return 1;
    if (keys.has("e")) return -1;
  }
  return null;
}

function inputAxis(index) {
  const keyboard = keyboardAxis(index);
  if (keyboard !== null) return keyboard;
  if (index === 0) return -controllerState.joystick.left.y;
  if (index === 1) return -controllerState.joystick.left.x;
  return -controllerState.joystick.right.x;
}

function recomputeCommand() {
  if (!activePolicy) return;
  ["vx", "vy", "yaw"].forEach((key, index) => {
    const [min, max] = activePolicy.commandRanges[key];
    const value = inputAxis(index) * controllerState.commandSpeed[index];
    simulation.command[index] = THREE.MathUtils.clamp(value, min, max);
  });
  updateControls();
}

function keyboardJoystickPosition(stickKey) {
  const state = controllerState.joystick[stickKey];
  if (stickKey === "left") {
    const vx = keyboardAxis(0);
    const vy = keyboardAxis(1);
    return {
      x: vy === null ? state.x : -vy,
      y: vx === null ? state.y : -vx,
    };
  }
  const yaw = keyboardAxis(2);
  return { x: yaw === null ? state.x : -yaw, y: state.y };
}

function renderJoystickPosition(stickKey) {
  const mount = $(stickKey === "left" ? "leftJoystick" : "rightJoystick");
  const base = mount.querySelector(".joystick-base");
  const knob = mount.querySelector(".joystick-knob");
  const { x, y } = keyboardJoystickPosition(stickKey);
  const radius = Math.max(1, base.getBoundingClientRect().width / 2 - knob.offsetWidth / 2 - 3);
  knob.style.left = `calc(50% + ${x * radius}px)`;
  knob.style.top = `calc(50% + ${y * radius}px)`;
}

function syncJoystickVisuals() {
  renderJoystickPosition("left");
  renderJoystickPosition("right");
}

function drawActionChart() {
  const canvas = $("actionChart");
  const context = canvas.getContext("2d");
  const { width, height } = canvas;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#edf2f5";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "#cbd8e0";
  context.lineWidth = 1;
  for (let row = 1; row < 4; row += 1) { const y = height * row / 4; context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke(); }
  context.strokeStyle = "#9bb0be";
  context.beginPath(); context.moveTo(0, height / 2); context.lineTo(width, height / 2); context.stroke();
  const samples = simulation.actionHistory;
  const palette = ["#185a91", "#4d7b4a", "#ae6c32", "#825eaa", "#b14955", "#347d8b"];
  if (samples.length > 1) {
    for (let joint = 0; joint < simulation.action.length; joint += 1) {
      context.beginPath();
      context.strokeStyle = palette[joint % palette.length];
      context.globalAlpha = joint < 6 ? .78 : .43;
      samples.forEach((sample, index) => {
        const x = index * width / (samples.length - 1);
        const y = height / 2 - THREE.MathUtils.clamp(sample[joint], -1.2, 1.2) * height / 3;
        if (index) context.lineTo(x, y); else context.moveTo(x, y);
      });
      context.stroke();
    }
  }
  context.globalAlpha = 1;
  $("actionValues").replaceChildren(...Array.from(simulation.action, (value, index) => {
    const item = document.createElement("span");
    item.innerHTML = `a${index + 1} <b>${value.toFixed(2)}</b>`;
    return item;
  }));
}

let lastTelemetryDraw = 0;
function updateTelemetry() {
  if (!data) return;
  $("bodyPosition").textContent = `x ${data.qpos[0].toFixed(2)} · y ${data.qpos[1].toFixed(2)} · z ${data.qpos[2].toFixed(2)}`;
  $("contactCount").textContent = String(data.ncon ?? 0);
  let terrainContacts = 0;
  for (let index = 0; index < data.ncon; index += 1) {
    const contact = data.contact.get(index);
    if (contact && (terrainSlotSet.has(contact.geom1) || terrainSlotSet.has(contact.geom2))) terrainContacts += 1;
  }
  $("terrainContactCount").textContent = String(terrainContacts);
  if (performance.now() - lastTelemetryDraw > 100) {
    drawActionChart();
    lastTelemetryDraw = performance.now();
  }
}

function updatePointer(event) {
  const bounds = view.renderer.domElement.getBoundingClientRect();
  pointer.set((event.clientX - bounds.left) / bounds.width * 2 - 1, -((event.clientY - bounds.top) / bounds.height) * 2 + 1);
  raycaster.setFromCamera(pointer, view.camera);
}

function robotHit(event) {
  updatePointer(event);
  return raycaster.intersectObject(robot, true)[0];
}

function installRobotDrag() {
  const canvas = view.renderer.domElement;
  const pointOnDragPlane = (target) => raycaster.ray.intersectPlane(dragPlane, target);
  canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || !data) return;
    const hit = robotHit(event);
    if (!hit) return;
    dragPlane.setFromNormalAndCoplanarPoint(view.camera.getWorldDirection(new THREE.Vector3()), hit.point);
    const start = pointOnDragPlane(new THREE.Vector3());
    if (!start) return;
    robotDrag = { start, vector: new THREE.Vector3() };
    view.orbit.enabled = false;
    canvas.setPointerCapture(event.pointerId);
    canvas.style.cursor = "grabbing";
    event.preventDefault();
    event.stopImmediatePropagation();
  }, true);
  canvas.addEventListener("pointermove", (event) => {
    if (!robotDrag) {
      canvas.style.cursor = robotHit(event) ? "pointer" : "";
      return;
    }
    updatePointer(event);
    const end = pointOnDragPlane(new THREE.Vector3());
    if (!end) return;
    robotDrag.vector.copy(end).sub(robotDrag.start).clampLength(0, 1.15);
    const length = robotDrag.vector.length();
    dragArrow.position.copy(robotDrag.start);
    dragArrow.setDirection(length > .001 ? robotDrag.vector.clone().normalize() : new THREE.Vector3(1, 0, 0));
    dragArrow.setLength(length, .13, .07);
    dragArrow.visible = true;
    event.preventDefault();
    event.stopImmediatePropagation();
  }, true);
  const release = (event) => {
    if (!robotDrag || !data) return;
    const impulse = robotDrag.vector.clone().multiplyScalar(4.8);
    data.qvel[0] += impulse.x;
    data.qvel[1] += impulse.y;
    data.qvel[2] += impulse.z;
    dragArrow.visible = false;
    robotDrag = null;
    view.orbit.enabled = true;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    canvas.style.cursor = "";
    if (impulse.lengthSq() > .0025) setNotice("已向机器人施加拖拽向量。", false);
    event.preventDefault();
    event.stopImmediatePropagation();
  };
  canvas.addEventListener("pointerup", release, true);
  canvas.addEventListener("pointercancel", release, true);
}

let pendingStep = false;
let lastFrameTime = null;
async function tick(timestamp) {
  const now = Number.isFinite(timestamp) ? timestamp : performance.now();
  if (lastFrameTime === null) lastFrameTime = now;
  const frameDelta = Math.max(0, (now - lastFrameTime) / 1000);
  lastFrameTime = now;

  if (simulation.running) simulation.realTimeAccumulator += frameDelta;
  else simulation.realTimeAccumulator = 0;

  if (simulation.running && !pendingStep) {
    pendingStep = true;
    try {
      while (simulation.running && simulation.realTimeAccumulator >= actualPolicyDt) {
        await policyStep();
        if (!simulation.running) break;
        for (let index = 0; index < physicsStepsPerPolicy; index += 1) {
          applyControl();
          mujoco.mj_step(model, data);
        }
        simulation.elapsed = data.time;
        simulation.realTimeAccumulator -= actualPolicyDt;

        if (data.qpos[2] < -0.2) {
          reset();
          setNotice("机器人离开场地，已自动重置。");
          break;
        }
      }

    } catch (error) {
      simulation.running = false;
      $("start").textContent = "开始";
      setStatus(`运行失败：${error.message}`, "error");
      setNotice(error.message, true);
    } finally { pendingStep = false; }
  }
  updateRobot();
  $("clock").textContent = `${simulation.elapsed.toFixed(2)} s`;
  updateTelemetry();
  view.orbit.update();
  view.renderer.render(view.scene, view.camera);
  requestAnimationFrame(tick);
}

const terrainLabels = { platform: "高台", stairs: "台阶", ramp: "斜坡", stones: "梅花桩", wall: "矮墙" };
const terrainDefaults = {
  platform: { length: 1.8, width: 1.8, height: .28 },
  wall: { length: 2.5, width: .24, height: .28 },
  stairs: { stepDepth: .28, stepWidth: 1.6, stepHeight: .12, stepCount: 5 },
  ramp: { length: 2.4, width: 1.8, height: .36, slope: 15 },
  stones: { length: .28, width: .28, height: .14, columns: 3, rows: 2, gapX: .14, gapY: .32, layout: "aligned" },
};

function newTerrainElement(kind, x, y) {
  return { kind, x, y, yaw: 0, ...terrainDefaults[kind] };
}

function terrainCanvasPosition(event) {
  const canvas = $("terrainMap");
  const bounds = canvas.getBoundingClientRect();
  return {
    x: THREE.MathUtils.clamp(((event.clientX - bounds.left) / bounds.width - .5) * TERRAIN_AREA.x * 2, -TERRAIN_AREA.x, TERRAIN_AREA.x),
    y: THREE.MathUtils.clamp((.5 - (event.clientY - bounds.top) / bounds.height) * TERRAIN_AREA.y * 2, -TERRAIN_AREA.y, TERRAIN_AREA.y),
  };
}

function repaintTerrainMap() {
  const canvas = $("terrainMap");
  const context = canvas.getContext("2d");
  const { width, height } = canvas;
  context.clearRect(0, 0, width, height);
  context.fillStyle = terrainState.kind === "flat" ? "#e7eee8" : "#dde8eb";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "#c1d0d5";
  context.lineWidth = 1;
  for (let x = 0; x <= width; x += width / 10) { context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke(); }
  for (let y = 0; y <= height; y += height / 8) { context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke(); }
  if (terrainState.kind === "slope") { context.fillStyle = "#9db6c5"; context.fillRect(width * .56, height * .16, width * .29, height * .68); }
  if (terrainState.kind === "stairs") {
    context.fillStyle = "#9db6c5";
    for (let index = 0; index < 7; index += 1) context.fillRect(width * (.54 + index * .038), height * .18, width * .034, height * .64);
  }
  terrainState.elements.forEach((element, index) => {
    const x = (element.x / (TERRAIN_AREA.x * 2) + .5) * width;
    const y = (.5 - element.y / (TERRAIN_AREA.y * 2)) * height;
    context.fillStyle = index === terrainState.selected ? "#7ca8c8" : "#a2b8c9";
    context.strokeStyle = index === terrainState.selected ? "#185a91" : "#6486a0";
    for (const definition of elementBoxes(element)) {
      const angle = definition.yaw || 0;
      const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
      context.beginPath();
      corners.forEach(([sideX, sideY], corner) => {
        const dx = sideX * definition.sx, dy = sideY * definition.sy;
        const px = ((definition.x + dx * Math.cos(angle) - dy * Math.sin(angle)) / (TERRAIN_AREA.x * 2) + .5) * width;
        const py = (.5 - (definition.y + dx * Math.sin(angle) + dy * Math.cos(angle)) / (TERRAIN_AREA.y * 2)) * height;
        if (corner === 0) context.moveTo(px, py); else context.lineTo(px, py);
      });
      context.closePath(); context.fill(); context.stroke();
    }
    context.beginPath();
    context.arc(x, y, 9, 0, Math.PI * 2);
    context.fillStyle = index === terrainState.selected ? "#185a91" : "#6f94ae";
    context.fill();
    context.fillStyle = "#fff";
    context.font = "10px system-ui";
    context.textAlign = "center";
    context.fillText(String(index + 1), x, y + 3);
  });
}

function renderTerrainElements() {
  const mount = $("terrainElements");
  mount.replaceChildren(...terrainState.elements.map((element, index) => {
    const row = document.createElement("div");
    row.className = `terrain-element${index === terrainState.selected ? " selected" : ""}`;
    const detail = element.kind === "stairs" ? `${element.stepCount} 阶`
      : element.kind === "stones" ? `${element.columns} × ${element.rows} 株`
        : `${(element.height ?? terrainState.height).toFixed(2)} m`;
    row.innerHTML = `<span>${index + 1}. ${terrainLabels[element.kind] ?? "导入障碍"} · ${detail}</span>`;
    row.onclick = () => { terrainState.selected = index; repaintTerrainEditor(); };
    const remove = document.createElement("button");
    remove.textContent = "删除";
    remove.onclick = (event) => {
      event.stopPropagation();
      terrainState.elements.splice(index, 1);
      terrainState.selected = null;
      repaintTerrainEditor();
    };
    row.append(remove);
    return row;
  }));
}

function repaintTerrainEditor() {
  repaintTerrainMap();
  renderTerrainElements();
  updateTerrainElementEditor();
  renderTerrain();
}

function updateTerrainElementEditor() {
  const element = terrainState.elements[terrainState.selected];
  $("terrainElementEditor").hidden = !element;
  if (!element) return;
  $("elementEditorTitle").textContent = `${terrainLabels[element.kind]}参数`;
  document.querySelectorAll("[data-element-for]").forEach((label) => {
    label.hidden = !label.dataset.elementFor.split(" ").includes(element.kind);
  });
  $("elementX").value = element.x.toFixed(2);
  $("elementY").value = element.y.toFixed(2);
  $("elementYaw").value = THREE.MathUtils.radToDeg(element.yaw || 0).toFixed(0);
  const fields = {
    elementLength: element.length, elementWidth: element.width, elementHeight: element.height,
    stepDepth: element.stepDepth, stepWidth: element.stepWidth, stepHeight: element.stepHeight,
    stepCount: element.stepCount, elementSlope: element.slope,
    stoneColumns: element.columns, stoneRows: element.rows,
    stoneGapX: element.gapX, stoneGapY: element.gapY, stoneLayout: element.layout,
  };
  for (const [id, value] of Object.entries(fields)) {
    if (value !== undefined) $(id).value = typeof value === "number" ? Number(value.toFixed(2)) : value;
  }
}

function terrainElementAt(position) {
  return terrainState.elements.findLastIndex((element) => elementBoxes(element).some((definition) => {
    const dx = position.x - definition.x, dy = position.y - definition.y, yaw = definition.yaw || 0;
    const localX = dx * Math.cos(yaw) + dy * Math.sin(yaw);
    const localY = -dx * Math.sin(yaw) + dy * Math.cos(yaw);
    return Math.abs(localX) <= definition.sx && Math.abs(localY) <= definition.sy;
  }) || Math.hypot(element.x - position.x, element.y - position.y) < .25);
}

function moveTerrainElement(index, x, y) {
  const element = terrainState.elements[index];
  if (!element) return;
  element.x = THREE.MathUtils.clamp(x, -TERRAIN_AREA.x, TERRAIN_AREA.x);
  element.y = THREE.MathUtils.clamp(y, -TERRAIN_AREA.y, TERRAIN_AREA.y);
  terrainState.selected = index;
  repaintTerrainEditor();
  $("terrainHint").textContent = "障碍位置已更新预览；点击“应用到场景”同步物理碰撞。";
}

function addTerrainElement(position) {
  const nearby = terrainElementAt(position);
  if (nearby >= 0) terrainState.selected = nearby;
  else {
    terrainState.elements.push(newTerrainElement(terrainState.tool, position.x, position.y));
    terrainState.selected = terrainState.elements.length - 1;
  }
  repaintTerrainEditor();
}

function installTerrainMapDrag() {
  const canvas = $("terrainMap");
  let drag = null;
  let suppressClick = false;
  canvas.addEventListener("contextmenu", (event) => event.preventDefault());
  canvas.addEventListener("pointerdown", (event) => {
    suppressClick = false;
    if (selectionBusy || (event.button !== 0 && !(event.button === 2 && event.shiftKey))) return;
    const point = terrainCanvasPosition(event);
    const index = terrainElementAt(point);
    if (index < 0) return;
    const element = terrainState.elements[index];
    drag = { pointerId: event.pointerId, index, startX: event.clientX, startY: event.clientY,
      moved: false, offsetX: element.x - point.x, offsetY: element.y - point.y };
    terrainState.selected = index;
    repaintTerrainMap();
    renderTerrainElements();
    updateTerrainElementEditor();
    canvas.setPointerCapture(event.pointerId);
    canvas.style.cursor = "grabbing";
    event.preventDefault();
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!drag) {
      canvas.style.cursor = terrainElementAt(terrainCanvasPosition(event)) >= 0 ? "grab" : "";
      return;
    }
    if (drag.pointerId !== event.pointerId) return;
    if (Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) < 3) return;
    drag.moved = true;
    const point = terrainCanvasPosition(event);
    moveTerrainElement(drag.index, point.x + drag.offsetX, point.y + drag.offsetY);
    canvas.style.cursor = "grabbing";
    event.preventDefault();
  });
  const release = (event) => {
    if (drag?.pointerId !== event.pointerId) return;
    suppressClick = event.type === "pointerup" && drag.moved;
    drag = null;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    canvas.style.cursor = "";
    event.preventDefault();
  };
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);
  canvas.addEventListener("click", (event) => {
    if (suppressClick) { suppressClick = false; return; }
    if (event.button === 0 && !selectionBusy) addTerrainElement(terrainCanvasPosition(event));
  });
}

function normalizeTerrainElement(source) {
  const kind = ({ stepping_stones: "stones", high_wall: "wall" })[source.kind] ?? source.kind;
  const safeKind = terrainDefaults[kind] ? kind : "platform";
  const element = newTerrainElement(safeKind,
    THREE.MathUtils.clamp(Number(source.x) || 0, -TERRAIN_AREA.x, TERRAIN_AREA.x),
    THREE.MathUtils.clamp(Number(source.y) || 0, -TERRAIN_AREA.y, TERRAIN_AREA.y));
  element.yaw = Number.isFinite(Number(source.yaw)) ? Number(source.yaw) : 0;
  const legacyX = Math.max(.05, Number(source.scaleX ?? source.scale_x ?? 1) || 1);
  const legacyY = Math.max(.05, Number(source.scaleY ?? source.scale_y ?? 1) || 1);
  if (safeKind === "stairs") {
    element.stepDepth *= legacyX;
    element.stepWidth *= legacyY;
    if (Number(source.height) > 0) element.stepHeight = Number(source.height) / 5;
  } else {
    element.length *= legacyX;
    element.width *= legacyY;
    if (Number(source.height) > 0) element.height = Number(source.height);
    if (safeKind === "stones" && (source.scaleX !== undefined || source.scaleY !== undefined)) {
      element.gapX *= legacyX;
      element.gapY *= legacyY;
      if (Number(source.height) > 0) element.height = Number(source.height) / 2;
    }
  }
  for (const key of ["length", "width", "height", "stepDepth", "stepWidth", "stepHeight", "stepCount", "slope", "columns", "rows", "gapX", "gapY"]) {
    if (source[key] !== undefined && Number.isFinite(Number(source[key]))) element[key] = Number(source[key]);
  }
  for (const key of ["length", "width", "height", "stepDepth", "stepWidth", "stepHeight"]) {
    if (element[key] !== undefined) element[key] = THREE.MathUtils.clamp(element[key], .02, 10);
  }
  for (const [key, maximum] of [["stepCount", 20], ["columns", 8], ["rows", 8]]) {
    if (element[key] !== undefined) element[key] = THREE.MathUtils.clamp(Math.round(element[key]), 1, maximum);
  }
  for (const key of ["gapX", "gapY"]) if (element[key] !== undefined) element[key] = THREE.MathUtils.clamp(element[key], 0, 5);
  if (["aligned", "staggered", "diagonal"].includes(source.layout)) element.layout = source.layout;
  if (safeKind === "ramp") element.slope = element.height / element.length * 100;
  return element;
}

function importTerrainScene(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const content = String(reader.result);
      if (file.name.toLowerCase().endsWith(".xml")) {
        const document = new DOMParser().parseFromString(content, "text/xml");
        const imported = [...document.querySelectorAll('geom[type="box"]')].map((geom) => {
          const position = vector(geom.getAttribute("pos") ?? "", 3, [0, 0, 0]);
          const size = vector(geom.getAttribute("size") ?? "", 3, [.8, .8, .2]);
          const euler = vector(geom.getAttribute("euler") ?? "", 3, [0, 0, 0]);
          return normalizeTerrainElement({ kind: size[0] > 1.1 && size[1] < .3 ? "wall" : "platform",
            x: position[0], y: position[1], yaw: euler[2] || 0,
            length: size[0] * 2, width: size[1] * 2, height: size[2] * 2 });
        });
        if (!imported.length) throw new Error("未找到可导入的 box 地形；高度场和外部 mesh 需先导出为 ArenaX JSON。");
        terrainState.elements = imported;
      } else {
        const scene = JSON.parse(content);
        const terrain = scene.terrain ?? scene;
        if (terrain.kind && ["flat", "slope", "stairs", "obstacle_mix"].includes(terrain.kind)) terrainState.kind = terrain.kind;
        terrainState.seed = Number(terrain.seed ?? terrainState.seed);
        terrainState.height = Number(terrain.height ?? terrain.obstacle_height ?? terrainState.height);
        terrainState.elements = (scene.elements ?? []).map((element) => normalizeTerrainElement({
          ...element, height: element.height ?? element.size?.[2] ?? terrainState.height,
        }));
      }
      terrainState.selected = null;
      $("terrainKind").value = terrainState.kind;
      $("terrainSeed").value = terrainState.seed;
      repaintTerrainEditor();
      setNotice("地形已导入预览；点击“应用到场景”后写入浏览器内 MuJoCo 物理模型。");
    } catch (error) { setNotice(error.message || String(error), true); }
  };
  reader.readAsText(file);
}

function exportTerrainScene() {
  const payload = { version: 2, name: "src-playground-terrain", terrain: { kind: terrainState.kind, seed: terrainState.seed, height: terrainState.height }, elements: terrainState.elements };
  const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "src-playground-terrain.json";
  anchor.click();
  URL.revokeObjectURL(url);
}

function installJoystick(id, stickKey) {
  const mount = $(id);
  const base = mount.querySelector(".joystick-base");
  const knob = mount.querySelector(".joystick-knob");
  let pointerId = null;

  const setPosition = (event) => {
    const bounds = base.getBoundingClientRect();
    const radius = Math.max(1, bounds.width / 2 - knob.offsetWidth / 2 - 3);
    let x = event.clientX - (bounds.left + bounds.width / 2);
    let y = event.clientY - (bounds.top + bounds.height / 2);
    const length = Math.hypot(x, y);
    if (length > radius) {
      x = x / length * radius;
      y = y / length * radius;
    }
    const state = controllerState.joystick[stickKey];
    state.x = x / radius;
    state.y = y / radius;
    renderJoystickPosition(stickKey);
    recomputeCommand();
  };

  const release = (event) => {
    if (pointerId !== event.pointerId) return;
    pointerId = null;
    controllerState.joystick[stickKey].x = 0;
    controllerState.joystick[stickKey].y = 0;
    renderJoystickPosition(stickKey);
    if (mount.hasPointerCapture(event.pointerId)) mount.releasePointerCapture(event.pointerId);
    recomputeCommand();
  };

  mount.addEventListener("pointerdown", (event) => {
    if (pointerId !== null) return;
    pointerId = event.pointerId;
    mount.setPointerCapture(pointerId);
    setPosition(event);
    event.preventDefault();
  });
  mount.addEventListener("pointermove", (event) => {
    if (pointerId !== event.pointerId) return;
    setPosition(event);
    event.preventDefault();
  });
  mount.addEventListener("pointerup", release);
  mount.addEventListener("pointercancel", release);
}

function resetJoysticks() {
  for (const [key, stick] of Object.entries(controllerState.joystick)) {
    stick.x = 0;
    stick.y = 0;
  }
  syncJoystickVisuals();
}

function installKeyboardControl() {
  const supportedKeys = new Set(["q", "w", "e", "a", "s", "d"]);
  const isTyping = (target) => ["INPUT", "SELECT", "TEXTAREA"].includes(target?.tagName) || target?.isContentEditable;
  window.addEventListener("keydown", (event) => {
    const key = event.key.toLowerCase();
    if (!supportedKeys.has(key) || isTyping(event.target)) return;
    controllerState.keyboardKeys.add(key);
    event.preventDefault();
    recomputeCommand();
  });
  window.addEventListener("keyup", (event) => {
    const key = event.key.toLowerCase();
    if (!supportedKeys.has(key)) return;
    controllerState.keyboardKeys.delete(key);
    event.preventDefault();
    recomputeCommand();
  });
  window.addEventListener("blur", () => {
    if (!controllerState.keyboardKeys.size) return;
    controllerState.keyboardKeys.clear();
    recomputeCommand();
  });
}

let panelLayer = 3;
function installDraggablePanel(id) {
  const panel = $(id), handle = panel.querySelector(".panel-heading");
  let drag = null;
  panel.addEventListener("pointerdown", () => { panel.style.zIndex = String(++panelLayer); });
  handle.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || event.target.closest("button")) return;
    const bounds = panel.getBoundingClientRect();
    drag = { pointerId: event.pointerId, offsetX: event.clientX - bounds.left, offsetY: event.clientY - bounds.top };
    handle.setPointerCapture(event.pointerId);
    event.preventDefault();
  });
  handle.addEventListener("pointermove", (event) => {
    if (drag?.pointerId !== event.pointerId) return;
    panel.style.left = `${THREE.MathUtils.clamp(event.clientX - drag.offsetX, 0, Math.max(0, window.innerWidth - panel.offsetWidth))}px`;
    panel.style.top = `${THREE.MathUtils.clamp(event.clientY - drag.offsetY, 0, Math.max(0, window.innerHeight - panel.offsetHeight))}px`;
    panel.style.right = "auto";
    event.preventDefault();
  });
  const release = (event) => {
    if (drag?.pointerId !== event.pointerId) return;
    drag = null;
    if (handle.hasPointerCapture(event.pointerId)) handle.releasePointerCapture(event.pointerId);
  };
  handle.addEventListener("pointerup", release);
  handle.addEventListener("pointercancel", release);
  window.addEventListener("resize", () => {
    if (!panel.style.left || panel.hidden) return;
    panel.style.left = `${THREE.MathUtils.clamp(parseFloat(panel.style.left), 0, Math.max(0, window.innerWidth - panel.offsetWidth))}px`;
    panel.style.top = `${THREE.MathUtils.clamp(parseFloat(panel.style.top), 0, Math.max(0, window.innerHeight - panel.offsetHeight))}px`;
  });
}

for (const [key, config] of Object.entries(robots)) $("robotModel").add(new Option(config.name, key));
populatePolicies(activeRobotKey, activePolicyKey);
$("robotModel").onchange = () => loadSelection($("robotModel").value).catch((error) => setNotice(error.message, true));
$("policy").onchange = () => loadSelection(activeRobotKey, $("policy").value).catch((error) => setNotice(error.message, true));
$("start").onclick = () => {
  if (selectionBusy || !policySession) return;
  simulation.running = !simulation.running;
  $("start").textContent = simulation.running ? "暂停" : "继续";
};
$("reset").onclick = async () => {
  if (selectionBusy) return;
  setSelectionBusy(true);
  const resume = simulation.running;
  await stopSimulation();
  reset();
  simulation.running = resume;
  $("start").textContent = resume ? "暂停" : "开始";
  setSelectionBusy(false);
};
$("cmdToggle").onclick = () => {
  const panel = $("cmdPanel");
  panel.hidden = !panel.hidden;
};
$("cmdClose").onclick = () => { $("cmdPanel").hidden = true; };
$("monitorToggle").onclick = () => {
  const panel = $("monitorPanel");
  panel.hidden = !panel.hidden;
  if (!panel.hidden) drawActionChart();
};
$("monitorClose").onclick = () => { $("monitorPanel").hidden = true; };
$("terrainToggle").onclick = () => {
  const panel = $("terrainPanel");
  panel.hidden = !panel.hidden;
  if (!panel.hidden) repaintTerrainEditor();
};
$("terrainClose").onclick = () => { $("terrainPanel").hidden = true; };
$("terrainKind").onchange = (event) => { terrainState.kind = event.target.value; repaintTerrainEditor(); };
$("terrainSeed").oninput = (event) => { terrainState.seed = Number(event.target.value) || 0; repaintTerrainEditor(); };
function editSelectedTerrain(key, value, input) {
  const element = terrainState.elements[terrainState.selected];
  if (!element || input.value.trim() === "" || !Number.isFinite(value) || !input.checkValidity()) {
    updateTerrainElementEditor();
    return;
  }
  if (key === "x") value = THREE.MathUtils.clamp(value, -TERRAIN_AREA.x, TERRAIN_AREA.x);
  if (key === "y") value = THREE.MathUtils.clamp(value, -TERRAIN_AREA.y, TERRAIN_AREA.y);
  element[key] = value;
  if (element.kind === "ramp") {
    if (key === "slope" || key === "length") element.height = element.length * element.slope / 100;
    else if (key === "height") element.slope = element.height / element.length * 100;
  }
  repaintTerrainEditor();
}
const elementFields = {
  elementX: "x", elementY: "y", elementLength: "length", elementWidth: "width", elementHeight: "height",
  stepDepth: "stepDepth", stepWidth: "stepWidth", stepHeight: "stepHeight", stepCount: "stepCount",
  elementSlope: "slope", stoneColumns: "columns", stoneRows: "rows", stoneGapX: "gapX", stoneGapY: "gapY",
};
for (const [id, key] of Object.entries(elementFields)) {
  $(id).onchange = (event) => editSelectedTerrain(key, Number(event.target.value), event.target);
}
$("elementYaw").onchange = (event) => editSelectedTerrain("yaw", THREE.MathUtils.degToRad(Number(event.target.value)), event.target);
$("stoneLayout").onchange = (event) => {
  const element = terrainState.elements[terrainState.selected];
  if (!element || element.kind !== "stones") return;
  element.layout = event.target.value;
  repaintTerrainEditor();
};
document.querySelectorAll("[data-terrain-tool]").forEach((button) => {
  button.onclick = () => {
    terrainState.tool = button.dataset.terrainTool;
    document.querySelectorAll("[data-terrain-tool]").forEach((item) => item.classList.toggle("active", item === button));
    $("terrainHint").textContent = `当前工具：${terrainLabels[terrainState.tool]}。点击空白处放置，直接拖动已有标记调整位置。`;
  };
});
installTerrainMapDrag();
$("terrainClear").onclick = () => { terrainState.elements = []; terrainState.selected = null; repaintTerrainEditor(); };
$("terrainApply").onclick = () => {
  applyTerrain().catch((error) => {
    setStatus("地形同步失败", "error");
    $("terrainHint").textContent = error.message || String(error);
    setNotice(error.message || String(error), true);
  });
};
$("terrainExport").onclick = exportTerrainScene;
$("terrainImport").onchange = (event) => {
  const [file] = event.target.files;
  if (file) importTerrainScene(file);
  event.target.value = "";
};
[["commandSpeedVx", 0], ["commandSpeedVy", 1], ["commandSpeedYaw", 2]].forEach(([id, index]) => {
  $(id).oninput = (event) => {
    controllerState.commandSpeed[index] = Number(event.target.value);
    recomputeCommand();
  };
});
$("commandStop").onclick = () => {
  controllerState.keyboardKeys.clear();
  resetJoysticks();
  recomputeCommand();
};
installJoystick("leftJoystick", "left");
installJoystick("rightJoystick", "right");
installKeyboardControl();
for (const id of ["cmdPanel", "monitorPanel", "terrainPanel"]) installDraggablePanel(id);

setBootStage("加载 MuJoCo 物理引擎…", 8);
$("robotModel").disabled = $("policy").disabled = true;
initializeMujoco()
  .then(() => {
    setBootStage("加载机器人与首个策略…", 68);
    return loadSelection(activeRobotKey, activePolicyKey);
  })
  .then(() => {
    $("cmdToggle").disabled = false;
    $("monitorToggle").disabled = false;
    $("terrainToggle").disabled = false;
    installRobotDrag();
    setBootStage("场景已就绪", 100);
    tick();
    requestAnimationFrame(revealScene);
  })
  .catch((error) => {
    console.error(error);
    setStatus("试玩区初始化失败", "error");
    setNotice(error.message || String(error), true);
    setBootStage(`初始化失败：${error.message || error}`, 100);
  });
