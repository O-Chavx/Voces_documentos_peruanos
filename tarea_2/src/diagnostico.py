"""Ejecuta un notebook celda por celda en un proceso normal de Python, para diagnosticar caídas.

    .\\entorno\\.venv\\Scripts\\python.exe -X faulthandler src\\diagnostico.py 01_extraccion.ipynb [desde_celda]

Si el proceso se cae dentro de una librería en C (Paddle, PyTorch...), faulthandler imprime
dónde. Las salidas no se guardan en el notebook: solo sirve para encontrar el problema.
"""
import faulthandler, json, sys, time, traceback, os
faulthandler.enable(all_threads=True)
os.environ.setdefault("MPLBACKEND", "Agg")
nb = json.load(open(sys.argv[1], encoding="utf-8"))
desde = int(sys.argv[2]) if len(sys.argv) > 2 else 0
ns = {"__name__": "__main__", "display": lambda *a, **k: print(*[str(x)[:300] for x in a])}
celdas = [c for c in nb["cells"] if c["cell_type"] == "code"]
for i, c in enumerate(celdas):
    src = "".join(c["source"])
    src = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith(("%", "!")))
    if i < desde and i != 0:
        continue
    print(f"\n===== CELDA {i}: {src.strip().splitlines()[0][:70]!r}", flush=True)
    t0 = time.time()
    try:
        exec(compile(src, f"celda_{i}", "exec"), ns)
    except Exception:
        traceback.print_exc()
        print(f"===== ERROR de Python en la celda {i}", flush=True)
        break
    print(f"===== celda {i} OK ({time.time() - t0:.0f} s)", flush=True)
print("===== FIN", flush=True)
