import os
import sys
import time
import uuid
import threading
import numpy as np
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

# Absolute path to this file's directory — all relative paths resolve from here.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)   # ensures imports from cfop modules work in-process

FACE_ORDER = ["upper", "front", "lower", "right", "left", "back"]
MATRIX_FILE = os.path.join(APP_DIR, "matrix.txt")

# ---------------------------------------------------------------------------
# In-process job store (replaces subprocess approach)
# ---------------------------------------------------------------------------
_jobs: dict[str, dict] = {}   # job_id -> {"status": "running"|"done"|"error", ...}
_jobs_lock = threading.Lock()

# Lazy-load the solver modules once (avoids repeated import overhead).
# We load them on first use inside the worker thread so startup stays fast.
_solvers_loaded = False
_solvers_lock = threading.Lock()

def _load_solvers():
    global _solvers_loaded
    global Cube, cube_from_matrices, read_matrices_c, solve_cross, to_standard
    global state_from_matrices, solve_state, simplify, apply_solution_to_matrices
    global solve_oll, solve_pll, _expand, is_pll_solved
    global CrossSolver, OLLSolver, PLLSolver
    global CROSS_MODEL, OLL_MODEL, PLL_MODEL

    with _solvers_lock:
        if _solvers_loaded:
            return
        from notations import Cube
        from c import (cube_from_matrices, read_matrices as read_matrices_c,
                       solve_cross, to_standard)
        from f import (state_from_matrices, solve_state, simplify,
                       apply_solution_to_matrices)
        from o import solve_oll
        from p import solve_pll
        from oll_algorithms import _expand
        from pll_verify import is_pll_solved
        from cross_solver import CrossSolver
        from oll_solver import OLLSolver
        from pll_solver import PLLSolver

        CROSS_MODEL = os.path.join(APP_DIR, "models", "cross_net_dropout03.pt")
        OLL_MODEL   = os.path.join(APP_DIR, "models", "oll_model.pt")
        PLL_MODEL   = os.path.join(APP_DIR, "models", "pll_model.pt")

        # Pre-instantiate solvers so model weights are loaded once.
        app._cross_solver = CrossSolver(CROSS_MODEL)
        app._oll_solver   = OLLSolver(OLL_MODEL)
        app._pll_solver   = PLLSolver(PLL_MODEL)

        _solvers_loaded = True   # must be last — marks the block as complete


# ---------------------------------------------------------------------------
# Cube helpers
# ---------------------------------------------------------------------------
def format_matrices_to_text(cube):
    lines = []
    for face_name in FACE_ORDER:
        face_arr = getattr(cube, face_name)
        for row in face_arr:
            lines.append(" ".join(str(val) for val in row))
    return "\n".join(lines) + "\n"


def save_cube_to_matrix_file(cube):
    with open(MATRIX_FILE, "w") as f:
        f.write(format_matrices_to_text(cube))


# ---------------------------------------------------------------------------
# Background solver worker
# ---------------------------------------------------------------------------
def _run_solver(job_id: str, matrix_text: str):
    """Run the full CFOP pipeline in a background thread and store the result."""
    try:
        _load_solvers()

        cube = cube_from_matrices(matrix_text)
        total_moves = []
        start_total = time.time()

        # Align Y so front centre is colour 1 (orange)
        align_y = []
        while cube.front[1, 1] != 1 and len(align_y) < 4:
            cube.Y()
            align_y.append("y")
        total_moves.extend(align_y)

        # --- Cross ---
        t0 = time.time()
        cross_moves = solve_cross(cube, solver=app._cross_solver)
        cross_time = time.time() - t0
        if cross_moves is None:
            raise RuntimeError("Failed to solve Cross stage.")
        for m in cross_moves:
            getattr(cube, m)()
        cube.X(); cube.X()
        cross_std = [to_standard(m) for m in cross_moves] + ["x2"]
        total_moves.extend(cross_std)

        # --- F2L ---
        f2l_matrices = [getattr(cube, n) for n in FACE_ORDER]
        f2l_state = state_from_matrices(f2l_matrices)
        t0 = time.time()
        raw_f2l = solve_state(f2l_state)
        f2l_time = time.time() - t0
        if raw_f2l is None:
            raise RuntimeError("No F2L solution found.")
        f2l_solution = simplify(raw_f2l)
        f2l_moves = f2l_solution.split() if f2l_solution else []
        updated_mats = apply_solution_to_matrices(f2l_matrices, f2l_solution)
        for name, mat in zip(FACE_ORDER, updated_mats):
            setattr(cube, name, mat)
        cube.X(); cube.X()
        f2l_moves.append("x2")
        total_moves.extend(f2l_moves)

        # --- OLL ---
        t0 = time.time()
        oll_moves = solve_oll(cube, solver=app._oll_solver)
        oll_time = time.time() - t0
        if oll_moves is None:
            raise RuntimeError("Failed to solve OLL stage.")
        for m in oll_moves:
            for name in _expand(m):
                getattr(cube, name)()
        total_moves.extend(oll_moves)

        # --- PLL ---
        t0 = time.time()
        pll_moves = solve_pll(cube, solver=app._pll_solver)
        pll_time = time.time() - t0
        if pll_moves is None:
            raise RuntimeError("Failed to solve PLL stage.")
        for m in pll_moves:
            for name in _expand(m):
                getattr(cube, name)()
        if not is_pll_solved(cube):
            raise RuntimeError("PLL verification failed.")
        total_moves.extend(pll_moves)

        total_time = time.time() - start_total

        parsed = {
            "cross_moves": cross_std,
            "cross_time":  f"{cross_time:.4f}s",
            "f2l_moves":   f2l_moves,
            "f2l_time":    f"{f2l_time:.4f}s",
            "oll_moves":   oll_moves,
            "oll_time":    f"{oll_time:.4f}s",
            "pll_moves":   pll_moves,
            "pll_time":    f"{pll_time:.4f}s",
            "total_moves": total_moves,
            "total_time":  f"{total_time:.4f}s",
        }

        with _jobs_lock:
            _jobs[job_id] = {"status": "done", "parsed": parsed}

    except Exception as exc:
        with _jobs_lock:
            _jobs[job_id] = {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/update_matrix", methods=["POST"])
def update_matrix():
    """Receive face matrices from frontend and save to matrix.txt."""
    data = request.json
    matrices = data.get("matrices")
    if not matrices:
        return jsonify({"success": False, "error": "No matrix data provided"}), 400

    # Import Cube lazily so cold-start of / is instant
    from notations import Cube
    cube = Cube()
    for name in FACE_ORDER:
        if name in matrices:
            setattr(cube, name, np.array(matrices[name], dtype=int))

    save_cube_to_matrix_file(cube)
    return jsonify({"success": True})


@app.route("/api/solve", methods=["POST"])
def solve():
    """
    Start the solver in a background thread and return a job_id immediately.
    The client polls /api/solve/status/<job_id> to check progress.
    This keeps the HTTP request short (<1 s) regardless of solver duration,
    avoiding cold-start timeouts on Render's free tier.
    """
    if not os.path.exists(MATRIX_FILE):
        # Write a default solved cube so cfop has something to read
        from notations import Cube
        save_cube_to_matrix_file(Cube())

    try:
        with open(MATRIX_FILE) as f:
            matrix_text = f.read()
    except OSError as e:
        return jsonify({"success": False, "error": f"Could not read matrix: {e}"}), 500

    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {"status": "running"}

    thread = threading.Thread(target=_run_solver, args=(job_id, matrix_text), daemon=True)
    thread.start()

    return jsonify({"success": True, "job_id": job_id})


@app.route("/api/solve/status/<job_id>", methods=["GET"])
def solve_status(job_id: str):
    """Poll this endpoint until status is 'done' or 'error'."""
    with _jobs_lock:
        job = _jobs.get(job_id)

    if job is None:
        return jsonify({"success": False, "error": "Unknown job ID"}), 404

    if job["status"] == "running":
        return jsonify({"success": True, "status": "running"})

    if job["status"] == "error":
        # Clean up after reading
        with _jobs_lock:
            _jobs.pop(job_id, None)
        return jsonify({"success": False, "error": job["error"]}), 500

    # done
    with _jobs_lock:
        _jobs.pop(job_id, None)
    return jsonify({"success": True, "status": "done", "parsed": job["parsed"]})


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
