import os
from pathlib import Path

import requests
from tqdm import tqdm


REPO_URL = "https://hf-mirror.com/datasets/Angelou0516/pancreas-ct/resolve/main"
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw_hf_direct"


def case_ids():
    return [case for case in range(1, 83) if case not in (25, 70)]


def download_file(session, rel_path, dst_path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{REPO_URL}/{rel_path}"

    last_error = None
    for attempt in range(1, 6):
        try:
            with session.get(url, stream=True, timeout=(120, 300), allow_redirects=True) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                if dst_path.exists() and total > 0 and dst_path.stat().st_size == total:
                    return
                with open(dst_path, "wb") as f, tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=rel_path,
                    leave=False,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
            last_error = None
            break
        except requests.RequestException as exc:
            last_error = exc
            print(f"Retry {attempt}/5 failed for {rel_path}: {exc}")

    if last_error is not None:
        raise last_error


def main():
    session = requests.Session()
    session.trust_env = False

    data_files = []
    for case in case_ids():
        data_files.append(
            (
                f"images/PANCREAS_{case:04d}/none_pancreas.nii.gz",
                RAW_DIR / "images" / f"PANCREAS_{case:04d}" / "none_pancreas.nii.gz",
            )
        )
        data_files.append(
            (
                f"labels/label{case:04d}.nii.gz",
                RAW_DIR / "labels" / f"label{case:04d}.nii.gz",
            )
        )

    for rel_path, dst_path in tqdm(data_files, desc="files"):
        download_file(session, rel_path, dst_path)

    print(f"Downloaded dataset to {RAW_DIR}")


if __name__ == "__main__":
    main()
