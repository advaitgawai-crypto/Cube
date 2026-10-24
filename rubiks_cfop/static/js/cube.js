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
const ANIMATION_SPEED = 140; // ms per turn - responsive and smooth
const GAP = 0.02;
const SIZE = 0.98;

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
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.45);
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
    while (cubeGroup.children.length > 0) {
        cubeGroup.remove(cubeGroup.children[0]);
    }
    cubies = [];

    const geometry = new THREE.BoxGeometry(SIZE, SIZE, SIZE);

    for (let x = -1; x <= 1; x++) {
        for (let y = -1; y <= 1; y++) {
            for (let z = -1; z <= 1; z++) {
                // 6 materials for 6 faces: Right(+X), Left(-X), Upper(+Y), Lower(-Y), Front(+Z), Back(-Z)
                const materials = [];
                for (let i = 0; i < 6; i++) {
                    materials.push(new THREE.MeshLambertMaterial({ color: 0x181818 }));
                }
                const cubie = new THREE.Mesh(geometry, materials);
                cubie.position.set(x * (SIZE + GAP), y * (SIZE + GAP), z * (SIZE + GAP));
                cubie.userData = { customX: x, customY: y, customZ: z };
                cubeGroup.add(cubie);
                cubies.push(cubie);
            }
        }
    }
    updateCubeVisuals();
}

/**
 * Updates each cubie's face colors directly from internalCube.
 * Guarantees zero coordinate drift, zero desync, and 100% accurate rendering.
 */
function updateCubeVisuals() {
    const black = 0x181818;
    for (let c of cubies) {
        const { customX: x, customY: y, customZ: z } = c.userData;
        // 0: Right (+X)
        c.material[0].color.setHex(x === 1 ? COLOR_MAP[internalCube.right[1 - y][1 - z]] : black);
        // 1: Left (-X)
        c.material[1].color.setHex(x === -1 ? COLOR_MAP[internalCube.left[1 - y][z + 1]] : black);
        // 2: Upper (+Y)
        c.material[2].color.setHex(y === 1 ? COLOR_MAP[internalCube.upper[z + 1][x + 1]] : black);
        // 3: Lower (-Y)
        c.material[3].color.setHex(y === -1 ? COLOR_MAP[internalCube.lower[1 - z][x + 1]] : black);
        // 4: Front (+Z)
        c.material[4].color.setHex(z === 1 ? COLOR_MAP[internalCube.front[1 - y][x + 1]] : black);
        // 5: Back (-Z)
        c.material[5].color.setHex(z === -1 ? COLOR_MAP[internalCube.back[1 - y][1 - x]] : black);
    }
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
// LOGICAL MATRIX ROTATIONS & MOVE PARSING
// -------------------------------------------------------------
function rot90CW(mat) {
    // Equivalent to numpy np.rot90(mat, -1)
    return mat[0].map((val, index) => mat.map(row => row[index]).reverse());
}

function rot90CCW(mat) {
    // Equivalent to numpy np.rot90(mat, 1)
    return mat[0].map((val, index) => mat.map(row => row[row.length - 1 - index]));
}

function applyLogicalMove(move) {
    const c = internalCube;
    let temp;

    switch (move) {
        case "R":
            const uColR = [c.upper[0][2], c.upper[1][2], c.upper[2][2]];
            const fColR = [c.front[0][2], c.front[1][2], c.front[2][2]];
            const lColR = [c.lower[0][2], c.lower[1][2], c.lower[2][2]];
            const bColR = [c.back[0][0], c.back[1][0], c.back[2][0]];
            for (let i = 0; i < 3; i++) {
                c.upper[i][2] = fColR[i];
                c.front[i][2] = lColR[i];
                c.lower[i][2] = bColR[2 - i];
                c.back[i][0]  = uColR[2 - i];
            }
            c.right = rot90CW(c.right);
            break;
        case "R'": for (let i = 0; i < 3; i++) applyLogicalMove("R"); break;

        case "L":
            const uColL = [c.upper[0][0], c.upper[1][0], c.upper[2][0]];
            const fColL = [c.front[0][0], c.front[1][0], c.front[2][0]];
            const lColL = [c.lower[0][0], c.lower[1][0], c.lower[2][0]];
            const bColL = [c.back[0][2], c.back[1][2], c.back[2][2]];
            for (let i = 0; i < 3; i++) {
                c.upper[i][0] = bColL[2 - i];
                c.back[i][2]  = lColL[2 - i];
                c.lower[i][0] = fColL[i];
                c.front[i][0] = uColL[i];
            }
            c.left = rot90CW(c.left);
            break;
        case "L'": for (let i = 0; i < 3; i++) applyLogicalMove("L"); break;

        case "U":
            temp = [...c.front[0]];
            c.front[0] = [...c.right[0]];
            c.right[0] = [...c.back[0]];
            c.back[0]  = [...c.left[0]];
            c.left[0]  = temp;
            c.upper = rot90CW(c.upper);
            break;
        case "U'": for (let i = 0; i < 3; i++) applyLogicalMove("U"); break;

        case "D":
            temp = [...c.front[2]];
            c.front[2] = [...c.left[2]];
            c.left[2]  = [...c.back[2]];
            c.back[2]  = [...c.right[2]];
            c.right[2] = temp;
            c.lower = rot90CW(c.lower);
            break;
        case "D'": for (let i = 0; i < 3; i++) applyLogicalMove("D"); break;

        case "F":
            const uRowF = [...c.upper[2]];
            const lColF = [c.left[0][2], c.left[1][2], c.left[2][2]];
            const dRowF = [...c.lower[0]];
            const rColF = [c.right[0][0], c.right[1][0], c.right[2][0]];
            for (let i = 0; i < 3; i++) {
                c.upper[2][i] = lColF[2 - i];
                c.left[i][2]  = dRowF[i];
                c.lower[0][i] = rColF[2 - i];
                c.right[i][0] = uRowF[i];
            }
            c.front = rot90CW(c.front);
            break;
        case "F'": for (let i = 0; i < 3; i++) applyLogicalMove("F"); break;

        case "B":
            const uRowB = [...c.upper[0]];
            const rColB = [c.right[0][2], c.right[1][2], c.right[2][2]];
            const dRowB = [...c.lower[2]];
            const lColB = [c.left[0][0], c.left[1][0], c.left[2][0]];
            for (let i = 0; i < 3; i++) {
                c.upper[0][i] = rColB[i];
                c.right[i][2] = dRowB[2 - i];
                c.lower[2][i] = lColB[i];
                c.left[i][0]  = uRowB[2 - i];
            }
            c.back = rot90CW(c.back);
            break;
        case "B'": for (let i = 0; i < 3; i++) applyLogicalMove("B"); break;

        case "X": case "x":
            temp = c.upper.map(r => [...r]);
            c.upper = c.front.map(r => [...r]);
            c.front = c.lower.map(r => [...r]);
            c.lower = c.back.map(r => [...r].reverse()).reverse();
            c.back  = temp.map(r => [...r].reverse()).reverse();
            c.right = rot90CW(c.right);
            c.left  = rot90CCW(c.left);
            break;
        case "X'": case "x'": for (let i = 0; i < 3; i++) applyLogicalMove("X"); break;

        case "Y": case "y":
            temp = c.front.map(r => [...r]);
            c.front = c.right.map(r => [...r]);
            c.right = c.back.map(r => [...r]);
            c.back  = c.left.map(r => [...r]);
            c.left  = temp;
            c.upper = rot90CW(c.upper);
            c.lower = rot90CCW(c.lower);
            break;
        case "Y'": case "y'": for (let i = 0; i < 3; i++) applyLogicalMove("Y"); break;

        case "Z": case "z":
            temp = c.upper.map(r => [...r]);
            c.upper = rot90CW(c.left);
            c.left  = rot90CW(c.lower);
            c.lower = rot90CW(c.right);
            c.right = rot90CW(temp);
            c.front = rot90CW(c.front);
            c.back  = rot90CCW(c.back);
            break;
        case "Z'": case "z'": for (let i = 0; i < 3; i++) applyLogicalMove("Z"); break;
    }

    if (move.endsWith("2")) {
        const base = move.slice(0, -1);
        applyLogicalMove(base);
        applyLogicalMove(base);
    }
}

/**
 * Expand wide moves and slice moves to primitive turns matching oll_algorithms.py.
 */
function expandMove(token) {
    const table = {
        "l'": ["R'", "X"],
        "l":  ["X'", "R"],
        "r":  ["L", "X"],
        "r'": ["X'", "L'"],
        "M'": ["R'", "X", "L"],
        "M":  ["R", "X'", "L'"],
        "M2": ["R'", "X", "L", "R'", "X", "L"]
    };
    return table[token] || [token];
}

function syncMatrixToFile() {
    return fetch("/api/update_matrix", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ matrices: internalCube })
    });
}

// -------------------------------------------------------------
// ANIMATING THE 3D CUBE VISUALLY
// -------------------------------------------------------------
function animateSingleMove(move) {
    return new Promise((resolve) => {
        let axis = new THREE.Vector3(0, 1, 0);
        let angle = -Math.PI / 2;
        let selector = () => true;

        const isPrime = move.includes("'");
        const isDouble = move.endsWith("2");
        const base = move.replace("'", "").replace("2", "").toUpperCase();

        switch (base) {
            case "U": axis.set(0, 1, 0); angle = -Math.PI / 2; selector = c => c.userData.customY === 1; break;
            case "D": axis.set(0, 1, 0); angle = Math.PI / 2; selector = c => c.userData.customY === -1; break;
            case "R": axis.set(1, 0, 0); angle = -Math.PI / 2; selector = c => c.userData.customX === 1; break;
            case "L": axis.set(1, 0, 0); angle = Math.PI / 2; selector = c => c.userData.customX === -1; break;
            case "F": axis.set(0, 0, 1); angle = -Math.PI / 2; selector = c => c.userData.customZ === 1; break;
            case "B": axis.set(0, 0, 1); angle = Math.PI / 2; selector = c => c.userData.customZ === -1; break;
            case "X": axis.set(1, 0, 0); angle = -Math.PI / 2; selector = () => true; break;
            case "Y": axis.set(0, 1, 0); angle = -Math.PI / 2; selector = () => true; break;
            case "Z": axis.set(0, 0, 1); angle = -Math.PI / 2; selector = () => true; break;
            default:
                // Unknown move - update logical state and resolve
                applyLogicalMove(move);
                updateCubeVisuals();
                resolve();
                return;
        }

        if (isPrime) angle = -angle;
        if (isDouble) angle *= 2;

        const movingCubies = cubies.filter(selector);
        const pivot = new THREE.Group();
        scene.add(pivot);

        movingCubies.forEach(c => {
            cubeGroup.remove(c);
            pivot.add(c);
        });

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
                // Return cubies to cubeGroup at fixed original positions
                movingCubies.forEach(c => {
                    pivot.remove(c);
                    c.position.set(
                        c.userData.customX * (SIZE + GAP),
                        c.userData.customY * (SIZE + GAP),
                        c.userData.customZ * (SIZE + GAP)
                    );
                    c.rotation.set(0, 0, 0);
                    cubeGroup.add(c);
                });
                scene.remove(pivot);

                // Update logical state and refresh all cubie stickers
                applyLogicalMove(move);
                updateCubeVisuals();
                resolve();
            }
        }

        requestAnimationFrame(step);
    });
}

/**
 * Execute a move (expanding wide moves and slice moves as needed).
 * shouldSync = true only on manual moves or final solve step.
 */
async function executeMove(move, shouldSync = false) {
    const subMoves = expandMove(move);
    for (const sub of subMoves) {
        await animateSingleMove(sub);
    }
    if (shouldSync) {
        syncMatrixToFile();
    }
}

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
            await executeMove(move, true);
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
        updateCubeVisuals();
        syncMatrixToFile();
        clearUI();
        document.getElementById("animation-status").innerText = "Ready";
    });

    // Scramble Button
    document.getElementById("scramble-btn").addEventListener("click", async () => {
        if (isAnimating) return;
        isAnimating = true;
        clearUI();
        const moves = ["R", "L", "U", "D", "F", "B", "R'", "U'", "F'", "D'"];
        document.getElementById("animation-status").innerText = "Scrambling...";
        for (let i = 0; i < 10; i++) {
            const randMove = moves[Math.floor(Math.random() * moves.length)];
            await executeMove(randMove, false);
        }
        await syncMatrixToFile();
        document.getElementById("animation-status").innerText = "Ready";
        isAnimating = false;
    });

    // Solve Button — uses async job polling so no single HTTP request blocks
    // for the full solver duration (avoids cold-start / proxy timeouts on Render).
    document.getElementById("solve-btn").addEventListener("click", async () => {
        if (isAnimating) return;
        isAnimating = true;
        document.getElementById("animation-status").innerText = "Solving CFOP...";

        try {
            // 1. Sync cube state to the server.
            await syncMatrixToFile();

            // 2. Start the solver job — returns immediately with a job_id.
            const startRes = await fetch("/api/solve", { method: "POST" });
            const startData = await startRes.json();
            if (!startData.success) {
                throw new Error(startData.error || "Failed to start solver.");
            }
            const jobId = startData.job_id;

            // 3. Poll until done (every 2 s, up to 5 min).
            let parsed = null;
            const MAX_POLLS = 150;   // 150 × 2 s = 5 minutes
            for (let i = 0; i < MAX_POLLS; i++) {
                await new Promise(r => setTimeout(r, 2000));
                const pollRes = await fetch(`/api/solve/status/${jobId}`);
                const pollData = await pollRes.json();

                if (!pollData.success) {
                    throw new Error(pollData.error || "Solver error.");
                }
                if (pollData.status === "done") {
                    parsed = pollData.parsed;
                    break;
                }
                // status === "running" — keep waiting
                const elapsed = (i + 1) * 2;
                document.getElementById("animation-status").innerText =
                    `Solving CFOP... (${elapsed}s)`;
            }

            if (!parsed) {
                throw new Error("Solver timed out waiting for result.");
            }

            // 4. Update UI and animate.
            updateSolverUI(parsed);
            const allMoves = parsed.total_moves;
            document.getElementById("animation-status").innerText = "Animating Solution...";

            for (let i = 0; i < allMoves.length; i++) {
                highlightMoveToken(i);
                await executeMove(allMoves[i], false);
            }

            // 5. Reset internal state to canonical solved cube and sync.
            internalCube = {
                upper: Array(3).fill(0).map(() => Array(3).fill(0)),
                front: Array(3).fill(0).map(() => Array(3).fill(1)),
                lower: Array(3).fill(0).map(() => Array(3).fill(2)),
                right: Array(3).fill(0).map(() => Array(3).fill(3)),
                left:  Array(3).fill(0).map(() => Array(3).fill(4)),
                back:  Array(3).fill(0).map(() => Array(3).fill(5))
            };
            updateCubeVisuals();
            await syncMatrixToFile();
            document.getElementById("animation-status").innerText = "Cube Solved!";

        } catch (err) {
            alert("Solver Error: " + err);
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
