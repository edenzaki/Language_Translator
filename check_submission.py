# check_submission.py: checks submission.zip the way the portal does. DO NOT EDIT.
#     python check_submission.py submission.zip
# It unpacks the zip into an empty folder and, in a fresh Python process that can only see
# that folder, imports model.py, builds Translator(), loads weights.pt and translates.
import os, subprocess, sys, tempfile, zipfile

MAX_PARAMS = 30_000_000
REQUIRED = ["model.py", "vocab.json", "weights.pt"]

CHILD = """
import os, sys, time
import torch
failed = []
def ok(msg): print("  ok    " + msg, flush=True)
def bad(msg): print("  FAIL  " + msg, flush=True); failed.append(msg)
try:
    from model import Translator                     # only the three files are in this folder
    import model
    assert os.path.dirname(os.path.abspath(model.__file__)) == os.getcwd(), model.__file__
    ok("model.py imports on its own and defines Translator")
except Exception as e:
    bad(f"importing Translator from model.py alone failed: {type(e).__name__}: {e}"); sys.exit(1)
try:
    tr = Translator()
except Exception as e:
    bad(f"Translator() with no arguments failed: {type(e).__name__}: {e}"); sys.exit(1)
n = sum(p.numel() for p in torch.nn.Module.parameters(tr))
(ok if n <= %d else bad)(f"{n:,} parameters (limit %s)")
try:
    sd = torch.load("weights.pt", map_location="cpu", weights_only=True)
    tr.load_state_dict(sd, strict=True)
    ok(f"weights.pt loads into a fresh Translator with strict=True ({os.path.getsize('weights.pt') / 1e6:.1f} MB)")
except Exception as e:
    bad(f"weights.pt does not load into Translator() with strict=True: {type(e).__name__}: {e}"); sys.exit(1)
tr.eval()
sents = ["Where is the station?", "Tom is my best friend.", "I don't like cold weather.",
         "Can you help me with my homework?", "We will leave early tomorrow morning."]
def check_out(out, where):
    if not isinstance(out, list) or len(out) != len(sents) or not all(isinstance(s, str) for s in out):
        bad(f"{where}: translate() must return a list of {len(sents)} strings; got "
            f"{type(out).__name__} of length {len(out) if hasattr(out, '__len__') else '?'}"); return False
    ok(f"{where}: {len(out)} sentences in, {len(out)} strings out"); return True
try:
    out = tr.translate(sents)
except Exception as e:
    bad(f"translate() failed on the CPU: {type(e).__name__}: {e}"); sys.exit(1)
if check_out(out, "CPU"):
    for e, s in zip(sents[:3], out[:3]): print(f"          {e!r} -> {s!r}")
if torch.cuda.is_available():
    try:
        tr = tr.cuda()
        t0 = time.time(); outg = tr.translate(sents); dt = time.time() - t0
    except Exception as e:
        bad(f"translate() failed on the GPU: {type(e).__name__}: {e}"); sys.exit(1)
    if check_out(outg, "GPU"):
        t0 = time.time(); tr.translate(sents * 20); per100 = time.time() - t0
        ok(f"100 sentences take {per100:.1f} s on this GPU ({torch.cuda.get_device_name(0)}); "
           f"the grader translates about 1,000, so roughly {per100 * 10:.0f} s there (limit 15 min)")
else:
    print("  note  no GPU in this session: skipped the GPU translation and the timing")
sys.exit(1 if failed else 0)
""" % (MAX_PARAMS, f"{MAX_PARAMS:,}")


def main(zpath):
    if not os.path.exists(zpath):
        print(f"FAILED: {zpath} not found"); return 1
    with zipfile.ZipFile(zpath) as z:
        names = sorted(z.namelist())
        if names != REQUIRED:
            print(f"FAILED: the zip must hold exactly {', '.join(REQUIRED)}, at the top level. It holds {names}")
            return 1
        tmp = tempfile.mkdtemp(prefix="hw4check_")
        z.extractall(tmp)
    sizes = {n: os.path.getsize(os.path.join(tmp, n)) for n in REQUIRED}
    print(f"checking {zpath} (model.py {sizes['model.py']:,} bytes, vocab.json {sizes['vocab.json'] / 1e6:.2f} MB, "
          f"weights.pt {sizes['weights.pt'] / 1e6:.1f} MB)", flush=True)
    if sizes["model.py"] > 256 * 1024 or sizes["vocab.json"] > 8 * 1024 * 1024 or sizes["weights.pt"] > 160 * 1024 * 1024:
        print("FAILED: limits are model.py 256 KB, vocab.json 8 MB, weights.pt 160 MB"); return 1
    r = subprocess.run([sys.executable, "-c", CHILD], cwd=tmp)
    print("PASSED" if r.returncode == 0 else "FAILED: fix the lines marked FAIL above, rebuild the zip, check again")
    return r.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "submission.zip"))
