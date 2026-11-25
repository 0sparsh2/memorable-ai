"""Push a training job in training/kaggle/<job>/ as a private Kaggle GPU kernel.

  python training/kaggle/push.py qgen [--sources user/kernel,...]

Writes kernel-metadata.json into a build dir (never committed: it names your
Kaggle account) and runs `kaggle kernels push`. Needs the kaggle CLI logged in.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def username():
    out = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True).stdout
    m = re.search(r"username:\s*(\S+)", out)
    if not m:
        raise SystemExit("kaggle CLI is not logged in")
    return m.group(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job")
    ap.add_argument("--sources", default="", help="kernel outputs to mount, comma separated (user/slug)")
    ap.add_argument("--datasets", default="", help="datasets to mount, comma separated (user/slug)")
    ap.add_argument("--gpu", default="NvidiaTeslaT4")
    ap.add_argument("--cpu", action="store_true", help="no accelerator (API-only jobs such as v8_answers)")
    ap.add_argument("--slug", default="", help="kernel name; default memorable-<job> (a resumed session needs its own)")
    args = ap.parse_args()
    src = os.path.join(HERE, args.job)
    script = os.path.join(src, f"{args.job}.py")
    user = username()
    build = tempfile.mkdtemp(prefix=f"kaggle_{args.job}_")
    shutil.copy(script, build)
    slug = args.slug or f"memorable-{args.job.replace('_', '-')}"
    meta = {
        "id": f"{user}/{slug}",
        "title": slug,
        "code_file": f"{args.job}.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": not args.cpu,
        "enable_internet": True,
        "dataset_sources": [d if "/" in d else f"{user}/{d}" for d in args.datasets.split(",") if d],
        "kernel_sources": [s if "/" in s else f"{user}/{s}" for s in args.sources.split(",") if s],
        "competition_sources": [],
        "model_sources": [],
    }
    if not args.cpu:
        meta["machine_shape"] = args.gpu
    json.dump(meta, open(os.path.join(build, "kernel-metadata.json"), "w"), indent=2)
    subprocess.run(["kaggle", "kernels", "push", "-p", build], check=True)
    print(f"pushed {meta['id']} from {build}")


if __name__ == "__main__":
    main()
