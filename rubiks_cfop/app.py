import os
import subprocess
import time
import sys
import numpy as np
from flask import Flask, render_template, jsonify, request
from notations import Cube

app = Flask(__name__)

FACE_ORDER = ["upper", "front", "lower", "right", "left", "back"]

def format_matrices_to_text(cube):
    """Format cube faces as text suitable for matrix.txt."""
    lines = []
    for face_name in FACE_ORDER:
        face_arr = getattr(cube, face_name)
        for row in face_arr:
            lines.append(" ".join(str(val) for val in row))
    return "\n".join(lines) + "\n"

def save_cube_to_matrix_file(cube):
    """Write current cube state to matrix.txt."""
    text = format_matrices_to_text(cube)
    with open("matrix.txt", "w") as f:
        f.write(text)

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

    cube = Cube()
    for name in FACE_ORDER:
        if name in matrices:
            setattr(cube, name, np.array(matrices[name], dtype=int))

    save_cube_to_matrix_file(cube)
    return jsonify({"success": True})

@app.route("/api/solve", methods=["POST"])
def solve():
    """Run cfop.py script on matrix.txt and return the solve moves and statistics."""
    # Ensure matrix.txt is present
    if not os.path.exists("matrix.txt"):
        cube = Cube()
        save_cube_to_matrix_file(cube)

    try:
        start_t = time.time()
        result = subprocess.run(
            [sys.executable, "cfop.py", "matrix.txt"],
            capture_output=True,
            text=True,
            timeout=30
        )
        elapsed = time.time() - start_t

        output = result.stdout
        if result.returncode != 0:
            return jsonify({
                "success": False,
                "error": result.stderr or "CFOP Solver failed."
            }), 500

        # Parse output from cfop.py
        parsed = parse_cfop_output(output)
        return jsonify({
            "success": True,
            "parsed": parsed,
            "raw_output": output
        })

    except subprocess.TimeoutExpired:
        return jsonify({"success": False, "error": "Solver timed out"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def parse_cfop_output(output):
    """Extract individual stage moves, times, and total moves from cfop.py output."""
    res = {
        "cross_moves": [], "cross_time": "0s",
        "f2l_moves": [], "f2l_time": "0s",
        "oll_moves": [], "oll_time": "0s",
        "pll_moves": [], "pll_time": "0s",
        "total_moves": [],
        "total_time": "0s"
    }

    lines = output.splitlines()
    for line in lines:
        line_s = line.strip()
        if line_s.startswith("Cross ("):
            # e.g., Cross (8 moves, 0.0123s): B' U F L U' U' L' x2
            parts = line_s.split("): ")
            if len(parts) > 1:
                res["cross_moves"] = parts[1].split()
            time_part = line_s[line_s.find(",")+1:line_s.find("):")].strip()
            res["cross_time"] = time_part

        elif line_s.startswith("F2L   ("):
            parts = line_s.split("): ")
            if len(parts) > 1:
                res["f2l_moves"] = parts[1].split()
            time_part = line_s[line_s.find(",")+1:line_s.find("):")].strip()
            res["f2l_time"] = time_part

        elif line_s.startswith("OLL   ("):
            parts = line_s.split("): ")
            if len(parts) > 1:
                res["oll_moves"] = parts[1].split()
            time_part = line_s[line_s.find(",")+1:line_s.find("):")].strip()
            res["oll_time"] = time_part

        elif line_s.startswith("PLL   ("):
            parts = line_s.split("): ")
            if len(parts) > 1:
                res["pll_moves"] = parts[1].split()
            time_part = line_s[line_s.find(",")+1:line_s.find("):")].strip()
            res["pll_time"] = time_part

        elif line_s.startswith("FULL SOLUTION:"):
            res["total_moves"] = line_s.replace("FULL SOLUTION:", "").strip().split()

        elif line_s.startswith("TOTAL TIME TAKEN:"):
            res["total_time"] = line_s.replace("TOTAL TIME TAKEN:", "").strip()

    return res

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)