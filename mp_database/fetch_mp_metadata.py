#!/usr/bin/env python3
"""Fetch Materials Project summary metadata for an existing CIF inventory.

The API key is read from ``MP_API_KEY`` or an interactive hidden prompt.  It is
never accepted as a command-line option and is never written to the output.
Only summary documents are downloaded; existing CIF files are left untouched.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ID_RE = re.compile(r"(mp-\d+)")
ALPHA_ID_RE = re.compile(r"^(mp)-([a-z]{8})$")
DEFAULT_ENDPOINT = "https://api.materialsproject.org/materials/summary/"
FIELDS = [
    "material_id",
    "formula_pretty",
    "chemsys",
    "energy_above_hull",
    "formation_energy_per_atom",
    "is_stable",
    "theoretical",
    "deprecated",
    "band_gap",
    "density",
    "volume",
    "nsites",
    "last_updated",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def material_id_from_name(value: Any) -> str | None:
    match = ID_RE.search(str(value))
    return match.group(1) if match else None


def normalize_api_material_id(value: Any) -> str:
    """Map the API's fixed-width base-26 MPID form back to the legacy numeric ID.

    Recent API deployments may serialize ``mp-149`` as ``mp-aaaaaaft``.  The
    query still accepts the legacy ID used in the CIF inventory, so decoding the
    returned suffix gives a deterministic one-to-one join key.
    """
    text = str(value)
    if ID_RE.fullmatch(text):
        return text
    match = ALPHA_ID_RE.fullmatch(text)
    if not match:
        return text
    number = 0
    for character in match.group(2):
        number = number * 26 + (ord(character) - ord("a"))
    return f"{match.group(1)}-{number}"


def load_inventory(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "cif_file" not in frame.columns:
        raise ValueError("inventory must contain a cif_file column")
    if "material_id" not in frame.columns:
        frame["material_id"] = frame["cif_file"].map(material_id_from_name)
    else:
        parsed = frame["cif_file"].map(material_id_from_name)
        frame["material_id"] = frame["material_id"].fillna(parsed).astype(str)
    if frame["material_id"].isna().any() or frame["material_id"].eq("None").any():
        bad = frame.loc[frame["material_id"].isna() | frame["material_id"].eq("None"), "cif_file"]
        raise ValueError(f"could not parse MP IDs from: {bad.head(5).tolist()}")
    duplicated = frame["material_id"].duplicated(keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, ["material_id", "cif_file"]].head(10).to_dict("records")
        raise ValueError(f"material_id is not one-to-one in the inventory: {examples}")
    return frame


def make_session(api_key: str) -> requests.Session:
    retry = Retry(
        total=6,
        read=6,
        connect=6,
        status=6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        backoff_factor=1.0,
        respect_retry_after_header=True,
    )
    session = requests.Session()
    session.headers.update(
        {
            "x-api-key": api_key,
            "user-agent": "ML-Electrolyte-metadata-fetch/1.0",
            "accept": "application/json",
        }
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def query_batch(
    session: requests.Session,
    endpoint: str,
    ids: list[str],
    timeout: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    params = {
        "material_ids": ",".join(ids),
        "_fields": ",".join(FIELDS),
        "_limit": len(ids),
    }
    response = session.get(endpoint, params=params, timeout=timeout)
    if not response.ok:
        # Do not include request headers (and therefore never the API key) in errors.
        detail = response.text[:1000].replace("\n", " ")
        raise RuntimeError(f"MP API returned HTTP {response.status_code}: {detail}")
    payload = response.json()
    data = payload.get("data")
    if not isinstance(data, list):
        raise RuntimeError("MP API response does not contain a data list")
    return data, payload.get("meta", {})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True, help="CSV containing cif_file")
    parser.add_argument("--output", type=Path, required=True, help="metadata CSV to create")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--force", action="store_true", help="replace an existing output")
    args = parser.parse_args()

    if args.batch_size < 1 or args.batch_size > 200:
        raise ValueError("--batch-size must be between 1 and 200")
    if args.output.exists() and not args.force:
        raise FileExistsError(f"refusing to replace existing output: {args.output}")

    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        api_key = getpass.getpass("Materials Project API key: ")
    if not api_key:
        raise SystemExit("no Materials Project API key supplied")

    inventory = load_inventory(args.inventory)
    ids = sorted(inventory["material_id"].tolist())
    session = make_session(api_key)
    documents: dict[str, dict[str, Any]] = {}
    meta_samples: list[dict[str, Any]] = []
    batches = list(chunks(ids, args.batch_size))
    for number, batch in enumerate(batches, start=1):
        data, meta = query_batch(session, args.endpoint, batch, args.timeout)
        meta_samples.append(meta)
        for document in data:
            material_id = normalize_api_material_id(document.get("material_id", ""))
            if material_id in documents:
                raise RuntimeError(f"duplicate document returned for {material_id}")
            documents[material_id] = document
        print(
            f"batch {number}/{len(batches)}: requested={len(batch)}, "
            f"returned={len(data)}, cumulative={len(documents)}",
            flush=True,
        )
        if number < len(batches):
            time.sleep(0.05)
    session.close()

    rows = []
    for item in inventory[["material_id", "cif_file"]].to_dict("records"):
        document = documents.get(item["material_id"], {})
        row = {
            "material_id": item["material_id"],
            "api_material_id": document.get("material_id"),
            "cif_file": item["cif_file"],
            "mp_metadata_status": "ok" if document else "not_returned",
        }
        for field in FIELDS:
            if field != "material_id":
                row[field] = document.get(field)
        rows.append(row)
    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    output.to_csv(temporary, index=False)
    temporary.replace(args.output)

    missing = sorted(set(ids) - set(documents))
    provenance = {
        "schema_version": 1,
        "posthoc_target_aware": False,
        "source": "Materials Project summary API",
        "endpoint": args.endpoint,
        "fields": FIELDS,
        "inventory": str(args.inventory),
        "inventory_sha256": sha256_file(args.inventory),
        "requested_material_ids": len(ids),
        "returned_material_ids": len(documents),
        "missing_material_ids": missing,
        "api_meta_samples": meta_samples[:1] + meta_samples[-1:] if meta_samples else [],
        "api_key_persisted": False,
        "api_alpha_mpid_decoded_for_join": True,
    }
    provenance_path = args.output.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.output}: {len(output)} rows, {len(missing)} not returned")


if __name__ == "__main__":
    main()
