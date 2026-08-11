import glob, json, os, re
files = sorted(glob.glob("/storage/afarbin/jetreg/data/jets_*.npz"),
               key=lambda p: int(re.search(r"_0000(\d+)", p).group(1)))
assert len(files) == 17, f"expected 17 shards, found {len(files)}"
manifest = {"args": {"note": "reconstructed after node reboot 2026-08-10"},
            "files": [{"input": os.path.basename(f), "output": os.path.basename(f),
                       "cuts": {}} for f in files]}
json.dump(manifest, open("/storage/afarbin/jetreg/data/manifest.json", "w"), indent=2)
open("/storage/afarbin/jetreg/data/ALL17_READY", "w").write("ok")
print("manifest 17 + ALL17_READY flag written")
