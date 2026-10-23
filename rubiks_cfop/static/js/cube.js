// 3D Rubik's Cube with Three.js
let scene, camera, renderer, controls;
let cubies = [];
const cubeGroup = new THREE.Group();

// Colors: 0: White, 1: Orange, 2: Yellow, 3: Green, 4: Blue, 5: Red
const COLOR_MAP = {
    0: 0xffffff, // White (Upper)
    1: 0xffa500, // Orange (Front)
    2: 0xffff00, // Yellow (Lower)
    3: 0x00ff00, // Green (Right)
    4: 0x0000ff, // Blue (Left)
    5: 0xff0000  // Red (Back)
};

// Logical cube representation (6 faces 3x3 array)
// FACE_ORDER = ["upper", "front", "lower", "right", "left", "back"]
let internalCube = {
    upper: Array(3).fill(0).map(() => Array(3).fill(0)),
    front: Array(3).fill(0).map(() => Array(3).fill(1)),
    lower: Array(3).fill(0).map(() => Array(3).fill(2)),
    right: Array(3).fill(0).map(() => Array(3).fill(3)),
    left:  Array(3).fill(0).map(() => Array(3).fill(4)),
    back:  Array(3).fill(0).map(() => Array(3).fill(5))
};

let isAnimating = false;
const ANIMATION_SPEED = 180; // ms per turn - smooth, fast, clearly visible

function init() {
    const container = document.getElementById("canvas-container");
    scene = new THREE.Scene();

    camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.set(5, 5, 7);

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.5);
    dirLight.position.set(10, 20, 10);
    scene.add(dirLight);

    build3DCube();
    scene.add(cubeGroup);

    window.addEventListener("resize", onWindowResize);
    setupEventListeners();
    animate();

    // Initial matrix upload
    syncMatrixToFile();
}

function build3DCube() {
    // Clear previous
    while(cubeGroup.children.length > 0){ 
        cubeGroup.remove(cubeGroup.children[0]); 
    }
    cubies = [];

    const gap = 0.02;
    const size = 0.98;
    const geometry = new THREE.BoxGeometry(size, size, size);

    for (let x = -1; x <= 1; x++) {
        for (let y = -1; y <= 1; y++) {
            for (let z = -1; z <= 1; z++) {
                const materials = createCubieMaterials(x, y, z);
                const cubie = new THREE.Mesh(geometry, materials);
                cubie.position.set(x * (size + gap), y * (size + gap), z * (size + gap));
                cubie.userData = { customX: x, customY: y, customZ: z };
                cubeGroup.add(cubie);
                cubies.push(cubie);
            }
        }
    }
}

function createCubieMaterials(x, y, z) {
    // Right (X+), Left (X-), Upper (Y+), Lower (Y-), Front (Z+), Back (Z-)
    const materials = [];
    const blackMat = new THREE.MeshLambertMaterial({ color: 0x181818 });

    // Right: green (3)
    materials.push(x === 1 ? new THREE.MeshLambertMaterial({ color: COLOR_MAP[3] }) : blackMat);
    // Left: blue (4)
    materials.push(x === -1 ? new THREE.MeshLambertMaterial({ color: COLOR_MAP[4] }) : blackMat);
    // Upper: white (0)
    materials.push(y === 1 ? new THREE.MeshLambertMaterial({ color: COLOR_MAP[0] }) : blackMat);
    // Lower: yellow (2)
    materials.push(y === -1 ? new THREE.MeshLambertMaterial({ color: COLOR_MAP[2] }) : blackMat);
    // Front: orange (1)
    materials.push(z === 1 ? new THREE.MeshLambertMaterial({ color: COLOR_MAP[1] }) : blackMat);
    // Back: red (5)
    materials.push(z === -1 ? new THREE.MeshLambertMaterial({ color: COLOR_MAP[5] }) : blackMat);

    return materials;
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function onWindowResize() {
    const container = document.getElementById("canvas-container");
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

// -------------------------------------------------------------
// LOGICAL MATRIX ROTATIONS & FILE SYNC
// -------------------------------------------------------------
function rot90CW(mat) {
    return mat[0].map((val, index) => mat.map(row => row[index]).reverse());
}
function rot90CCW(mat) {
    return mat[0].map((val, index) => mat.map(row => row[index])).reverse();
}

function applyLogicalMove(move) {
    const c = internalCube;
    let temp;

    switch(move) {
        case "R":
            temp = [c.upper[0][2], c.upper[1][2], c.upper[2][2]];
            c.upper[0][2] = c.front[0][2]; c.upper[1][2] = c.front[1][2]; c.upper[2][2] = c.front[2][2];
            c.front[0][2] = c.lower[0][2]; c.front[1][2] = c.lower[1][2]; c.front[2][2] = c.lower[2][2];
            c.lower[0][2] = c.back[2][0];  c.lower[1][2] = c.back[1][0];  c.lower[2][2] = c.back[0][0];
            c.back[2][0] = temp[0];       c.back[1][0] = temp[1];       c.back[0][0] = temp[2];
            c.right = rot90CW(c.right);
            break;
        case "R'": for(let i=0; i<3; i++) applyLogicalMove("R"); break;

        case "L":
            temp = [c.upper[0][0], c.upper[1][0], c.upper[2][0]];
            c.upper[0][0] = c.back[2][2];  c.upper[1][0] = c.back[1][2];  c.upper[2][0] = c.back[0][2];
            c.back[2][2] = c.lower[0][0];  c.back[1][2] = c.lower[1][0];  c.back[0][2] = c.lower[2][0];
            c.lower[0][0] = c.front[0][0]; c.lower[1][0] = c.front[1][0]; c.lower[2][0] = c.front[2][0];
            c.front[0][0] = temp[0];       c.front[1][0] = temp[1];       c.front[2][0] = temp[2];
            c.left = rot90CW(c.left);
            break;
        case "L'": for(let i=0; i<3; i++) applyLogicalMove("L"); break;

        case "U":
            temp = [...c.front[0]];
            c.front[0] = [...c.right[0]];
            c.right[0] = [...c.back[0]];
            c.back[0]  = [...c.left[0]];
            c.left[0]  = temp;
            c.upper = rot90CW(c.upper);
            break;
        case "U'": for(let i=0; i<3; i++) applyLogicalMove("U"); break;

        case "D":
            temp = [...c.front[2]];
            c.front[2] = [...c.left[2]];
            c.left[2]  = [...c.back[2]];
            c.back[2]  = [...c.right[2]];
            c.right[2] = temp;
            c.lower = rot90CW(c.lower);
            break;
        case "D'": for(let i=0; i<3; i++) applyLogicalMove("D"); break;

        case "F":
            temp = [...c.upper[2]];
            c.upper[2] = [c.left[2][2], c.left[1][2], c.left[0][2]];
            c.left[0][2] = c.lower[0][0]; c.left[1][2] = c.lower[0][1]; c.left[2][2] = c.lower[0][2];
            c.lower[0][0] = c.right[2][0]; c.lower[0][1] = c.right[1][0]; c.lower[0][2] = c.right[0][0];
            c.right[0][0] = temp[0]; c.right[1][0] = temp[1]; c.right[2][0] = temp[2];
            c.front = rot90CW(c.front);
            break;
        case "F'": for(let i=0; i<3; i++) applyLogicalMove("F"); break;

        case "B":
            temp = [...c.upper[0]];
            c.upper[0] = [c.right[0][2], c.right[1][2], c.right[2][2]];
            c.right[0][2] = c.lower[2][2]; c.right[1][2] = c.lower[2][1]; c.right[2][2] = c.lower[2][0];
            c.lower[2][0] = c.left[0][0]; c.lower[2][1] = c.left[1][0]; c.lower[2][2] = c.left[2][0];
            c.left[0][0] = temp[2]; c.left[1][0] = temp[1]; c.left[2][0] = temp[0];
            c.back = rot90CW(c.back);
            break;
        case "B'": for(let i=0; i<3; i++) applyLogicalMove("B"); break;

        case "X":
        case "x":
            temp = c.upper; c.upper = c.front; c.front = c.lower;
            c.lower = c.back.map(row => [...row].reverse()).reverse();
            c.back = temp.map(row => [...row].reverse()).reverse();
            c.right = rot90CW(c.right); c.left = rot90CCW(c.left);
            break;
        case "X'": case "x'": for(let i=0; i<3; i++) applyLogicalMove("X"); break;

        case "Y": case "y":
            temp = c.front; c.front = c.right; c.right = c.back; c.back = c.left; c.left = temp;
            c.upper = rot90CW(c.upper); c.lower = rot90CCW(c.lower);
            break;
        case "Y'": case "y'": for(let i=0; i<3; i++) applyLogicalMove("Y"); break;

        case "Z": case "z":
            applyLogicalMove("X"); applyLogicalMove("Y"); applyLogicalMove("X'");
            break;
        case "Z'": case "z'": for(let i=0; i<3; i++) applyLogicalMove("Z"); break;
    }

    // Handle double turns (e.g. U2, X2)
    if (move.endsWith("2")) {
        const base = move.slice(0, -1);
        applyLogicalMove(base);
        applyLogicalMove(base);
    }
}

function syncMatrixToFile() {
    fetch("/api/update_matrix", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ matrices: internalCube })
    });
}

// -------------------------------------------------------------
// ANIMATING THE 3D CUBE VISUALLY
// -------------------------------------------------------------
function animate3DMove(move) {
    return new Promise((resolve) => {
        let axis = new THREE.Vector3(0, 1, 0);
        let angle = -Math.PI / 2;
        let selector = (c) => true;

        const isPrime = move.includes("'");
        const isDouble = move.includes("2");

        const base = move.replace("'", "").replace("2", "").toUpperCase();

        switch (base) {
            case "U": axis.set(0, 1, 0); angle = -Math.PI / 2; selector = c => c.position.y > 0.5; break;
            case "D": axis.set(0, 1, 0); angle = Math.PI / 2; selector = c => c.position.y < -0.5; break;
            case "R": axis.set(1, 0, 0); angle = -Math.PI / 2; selector = c => c.position.x > 0.5; break;
            case "L": axis.set(1, 0, 0); angle = Math.PI / 2; selector = c => c.position.x < -0.5; break;
            case "F": axis.set(0, 0, 1); angle = -Math.PI / 2; selector = c => c.position.z > 0.5; break;
            case "B": axis.set(0, 0, 1); angle = Math.PI / 2; selector = c => c.position.z < -0.5; break;
            case "X": axis.set(1, 0, 0); angle = -Math.PI / 2; selector = c => true; break;
            case "Y": axis.set(0, 1, 0); angle = -Math.PI / 2; selector = c => true; break;
            case "Z": axis.set(0, 0, 1); angle = -Math.PI / 2; selector = c => true; break;
        }

        if (isPrime) angle = -angle;
        if (isDouble) angle *= 2;

        const movingCubies = cubies.filter(selector);
        const pivot = new THREE.Group();
        scene.add(pivot);

        movingCubies.forEach(c => pivot.add(c));

        const startTime = performance.now();
        const duration = ANIMATION_SPEED;

        function step(now) {
            const elapsed = now - startTime;
            const progress = Math.min(elapsed / duration, 1.0);
            const currentAngle = angle * progress;

            pivot.rotation.set(0, 0, 0);
            pivot.rotateOnAxis(axis, currentAngle);

            if (progress < 1.0) {
                requestAnimationFrame(step);
            } else {
                // Finalize positions
                pivot.updateMatrixWorld();
                movingCubies.forEach(c => {
                    c.applyMatrix4(pivot.matrixWorld);
                    scene.add(c);
                    // Snap coordinates to integers
                    c.position.x = Math.round(c.position.x * 10) / 10;
                    c.position.y = Math.round(c.position.y * 10) / 10;
                    c.position.z = Math.round(c.position.z * 10) / 10;
                });
                scene.remove(pivot);

                // Update logical state & sync file
                applyLogicalMove(move);
                syncMatrixToFile();
                resolve();
            }
        }

        requestAnimationFrame(step);
    });
}
// Solve Button
// -------------------------------------------------------------
// EVENT LISTENERS & SOLVER INTEGRATION
// -------------------------------------------------------------
function setupEventListeners() {
    // Manual Move Buttons
    document.querySelectorAll(".move-btn").forEach(btn => {
        btn.addEventListener("click", async () => {
            if (isAnimating) return;
            isAnimating = true;
            const move = btn.getAttribute("data-move");
            document.getElementById("animation-status").innerText = "Executing " + move;
            await animate3DMove(move);
            document.getElementById("animation-status").innerText = "Ready";
            isAnimating = false;
        });
    });

    // Reset Button
    document.getElementById("reset-btn").addEventListener("click", () => {
        if (isAnimating) return;
        internalCube = {
            upper: Array(3).fill(0).map(() => Array(3).fill(0)),
            front: Array(3).fill(0).map(() => Array(3).fill(1)),
            lower: Array(3).fill(0).map(() => Array(3).fill(2)),
            right: Array(3).fill(0).map(() => Array(3).fill(3)),
            left:  Array(3).fill(0).map(() => Array(3).fill(4)),
            back:  Array(3).fill(0).map(() => Array(3).fill(5))
        };
        build3DCube();
        syncMatrixToFile();
        clearUI();
    });

    // Scramble Button
    document.getElementById("scramble-btn").addEventListener("click", async () => {
        if (isAnimating) return;
        isAnimating = true;
        const moves = ["R", "L", "U", "D", "F", "B", "R'", "U'", "F'", "D'"];
        document.getElementById("animation-status").innerText = "Scrambling...";
        for (let i = 0; i < 10; i++) {
            const randMove = moves[Math.floor(Math.random() * moves.length)];
            await animate3DMove(randMove);
        }
        document.getElementById("animation-status").innerText = "Ready";
        isAnimating = false;
    });

    // Solve Button
    document.getElementById("solve-btn").addEventListener("click", async () => {
        if (isAnimating) return;
        isAnimating = true;
        document.getElementById("animation-status").innerText = "Solving CFOP...";
        // await syncMatrixToFile();
        try {
            const res = await fetch("/api/solve", { method: "POST" });
            const data = await res.json();

            if (!data.success) {
                alert("Solver Error: " + data.error);
                document.getElementById("animation-status").innerText = "Error";
                isAnimating = false;
                return;
            }

            const parsed = data.parsed;
            updateSolverUI(parsed);

            // Execute solution moves sequentially
            const allMoves = parsed.total_moves;
            document.getElementById("animation-status").innerText = "Animating Solution...";

            for (let i = 0; i < allMoves.length; i++) {
                const move = allMoves[i];
                highlightMoveToken(i);
                await animate3DMove(move);
            }

            document.getElementById("animation-status").innerText = "Cube Solved!";
        } catch (err) {
            alert("Failed to solve: " + err);
            document.getElementById("animation-status").innerText = "Error";
        } finally {
            isAnimating = false;
        }
    });
}

function updateSolverUI(parsed) {
    document.getElementById("total-time").innerText = parsed.total_time;
    document.getElementById("total-moves-count").innerText = parsed.total_moves.length;

    document.getElementById("cross-time").innerText = parsed.cross_time;
    document.getElementById("f2l-time").innerText = parsed.f2l_time;
    document.getElementById("oll-time").innerText = parsed.oll_time;
    document.getElementById("pll-time").innerText = parsed.pll_time;

    const movesContainer = document.getElementById("stage-moves");
    movesContainer.innerHTML = "";

    const stages = [
        { name: "1. Cross", moves: parsed.cross_moves },
        { name: "2. F2L", moves: parsed.f2l_moves },
        { name: "3. OLL", moves: parsed.oll_moves },
        { name: "4. PLL", moves: parsed.pll_moves }
    ];

    let globalIdx = 0;
    stages.forEach(st => {
        const block = document.createElement("div");
        block.className = "stage-block";

        const title = document.createElement("div");
        title.className = "stage-title";
        title.innerText = st.name;
        block.appendChild(title);

        st.moves.forEach(m => {
            const span = document.createElement("span");
            span.className = "move-token";
            span.id = "move-token-" + globalIdx;
            span.innerText = m;
            block.appendChild(span);
            globalIdx++;
        });

        movesContainer.appendChild(block);
    });
}

function highlightMoveToken(index) {
    document.querySelectorAll(".move-token").forEach(el => el.classList.remove("active"));
    const activeEl = document.getElementById("move-token-" + index);
    if (activeEl) {
        activeEl.classList.add("active");
        activeEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
}

function clearUI() {
    document.getElementById("total-time").innerText = "--";
    document.getElementById("total-moves-count").innerText = "--";
    document.getElementById("cross-time").innerText = "--";
    document.getElementById("f2l-time").innerText = "--";
    document.getElementById("oll-time").innerText = "--";
    document.getElementById("pll-time").innerText = "--";
    document.getElementById("stage-moves").innerHTML = '<p class="placeholder">Click "Solve Cube" to run CFOP solver...</p>';
}

window.onload = init;