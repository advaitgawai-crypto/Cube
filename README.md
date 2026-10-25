# Rubik's Cube CFOP Solver

A Python Flask web application that solves a Rubik's Cube using the CFOP method (Cross, F2L, OLL, PLL). The server provides a simple UI for entering cube face matrices and returns a sequence of moves that solves the cube.

Built by [Advait Gawai](https://github.com/advaitgawai-crypto) and [Om Patil](https://www.linkedin.com/in/om-patil-913518394/).

---

##  Motivation

DeepCubeA runs one A* search over the whole cube. Its cost grows exponentially with solution depth, roughly O(12^n) where n is the number of moves. Splitting the solve into four stages (Cross, F2L, OLL, PLL) makes each search smaller. Each stage has its own learned heuristic, and A* uses it to find moves for that stage.

---

##  Performance

### Current results

| Metric | Value |
|--------|-------|
| Average solve time | ~0.74 s |
| Fastest solve | ~0.47 s |
| Average solution length | ~86 moves |

### Comparison with DeepCubeA

| Metric | This solver (CFOP) | DeepCubeA (reference) |
|--------|--------------------|-----------------------|
| Average solve time | ~0.74 s | ~1.2 s |
| Average solution length | ~86 moves | ~20–28 moves |

**Trade-off:** This solver is faster, but its solutions are much longer. Staged solving trades solution length for speed.

> The two solvers may use different move metrics (quarter-turn or half-turn) and different scramble sets. Confirm that the benchmark matches before comparing directly.

---

##  Project Structure

```
Cube/
├─ rubiks_cfop/               # Main package
│   ├─ app.py                # Flask entry-point, background job handling
│   ├─ c.py, f.py, o.py, p.py, ...  # Solver modules
│   ├─ models/               # Pre-trained PyTorch models for Cross, OLL, PLL
│   ├─ static/               # CSS/JS assets for the UI
│   ├─ templates/            # HTML template(s)
│   └─ ...
├─ .gitignore
├─ requirements.txt (optional)
└─ README.md                  # ← You are here
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/advaitgawai-crypto/Cube
   cd Cube/rubiks_cfop
   ```
2. **Create a virtual environment** (recommended) and activate it:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate   # PowerShell
   ```
3. **Install the required Python packages**:
   ```bash
   pip install -r requirements.txt
   ```
   If a `requirements.txt` file is missing, install the core dependencies manually:
   ```bash
   pip install flask numpy torch
   ```
4. **(Optional) Verify the model files.** The `models/` directory already contains the pre-trained weights the solver needs.
this is also temporarily deployed on https://cfop-core.onrender.com (use pc to access)
   
##  Running the Application

```bash
python app.py
```

- The server starts on `http://127.0.0.1:5000`.
- Open that URL in a browser to enter face matrices in the UI.
- A background thread runs the full CFOP pipeline. The JSON API is at `POST /api/update_matrix`.

## 📡 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Serves the HTML UI (`templates/index.html`). |
| `POST` | `/api/update_matrix` | Accepts a JSON payload with a `matrices` object (face → 3×3 array). Saves the cube state to `matrix.txt`, starts the solver, and returns the move list with timing information. |

### Example Request

```json
{
  "matrices": {
    "upper": [[0,1,0],[2,0,2],[1,2,1]],
    "front": [[...]],
    "...": [[...]]
  }
}
```

##  How It Works

1. **Lazy loading.** Solver modules and PyTorch models load on first use, which keeps Flask startup fast.
2. **Background job store.** Each solve request gets a `job_id`. The solver runs in a separate thread and updates a shared `_jobs` dictionary. The UI polls for completion.
3. **CFOP pipeline:**
   - Align the cube so the front centre is colour 1.
   - Solve the Cross with a neural-network heuristic and A* search.
   - Solve F2L with a custom state-search algorithm.
   - Solve OLL and PLL with separate models.
   - Verify the final state.
