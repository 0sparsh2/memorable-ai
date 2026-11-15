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
    ap.add_argument("--gpu", default="NvidiaTeslaT4")
    args = ap.parse_args()
    src = os.path.join(HERE, args.job)
    script = os.path.join(src, f"{args.job}.py")
    user = username()
    build = tempfile.mkdtemp(prefix=f"kaggle_{args.job}_")
    shutil.copy(script, build)
    meta = {
        "id": f"{user}/memorable-{args.job.replace('_', '-')}",
        "title": f"memorable-{args.job.replace('_', '-')}",
        "code_file": f"{args.job}.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": args.gpu,
        "dataset_sources": [],
        "kernel_sources": [s if "/" in s else f"{user}/{s}" for s in args.sources.split(",") if s],
        "competition_sources": [],
        "model_sources": [],
    }
    json.dump(meta, open(os.path.join(build, "kernel-metadata.json"), "w"), indent=2)
    subprocess.run(["kaggle", "kernels", "push", "-p", build], check=True)
    print(f"pushed {meta['id']} from {build}")


if __name__ == "__main__":
    main()
