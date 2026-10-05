from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from run_h001_backtest import (
    DATA_END_DATE,
    DATA_START_DATE,
    FRED_PATH,
    TICKERS,
    TIINGO_DIR,
)


OUTPUT_PATH = Path(
    "results/h001/data_manifest.json"
)

RETRIEVAL_METADATA_PATH = Path(
    "data/raw/h001_retrieval_metadata.json"
)

PACKAGE_NAMES = [
    "numpy",
    "pandas",
    "requests",
    "python-dotenv",
    "matplotlib",
    "pytest",
]


def file_sha256(
    path: Path,
) -> str:
    """Return the SHA-256 digest of a file."""

    digest = sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def file_modified_at_utc(
    path: Path,
) -> str:
    """Return filesystem modification time in UTC."""

    modified = datetime.fromtimestamp(
        path.stat().st_mtime,
        tz=timezone.utc,
    )

    return modified.isoformat()


def package_versions() -> dict[str, str]:
    """Return versions of the main runtime packages."""

    result = {}

    for package in PACKAGE_NAMES:
        try:
            result[package] = version(
                package
            )

        except PackageNotFoundError:
            result[package] = (
                "not-installed"
            )

    return result


def run_git_command(
    *args: str,
) -> str | None:
    """Run a read-only git command when available."""

    try:
        completed = subprocess.run(
            [
                "git",
                *args,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
    ):
        return None

    return completed.stdout.strip()


def git_metadata() -> dict[str, object]:
    """Describe the working tree at manifest generation time."""

    status = run_git_command(
        "status",
        "--porcelain",
    )

    return {
        "head_at_generation":
            run_git_command(
                "rev-parse",
                "HEAD",
            ),

        "branch_at_generation":
            run_git_command(
                "branch",
                "--show-current",
            ),

        "working_tree_dirty_at_generation":
            None
            if status is None
            else bool(status),
    }


def load_retrieval_metadata() -> dict:
    """
    Load recorded provider retrieval metadata.

    Legacy snapshots may legitimately have no retrieval record.
    """

    if not RETRIEVAL_METADATA_PATH.exists():
        return {
            "schema_version": 1,
            "files": {},
        }

    metadata = json.loads(
        RETRIEVAL_METADATA_PATH.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(metadata, dict):
        raise ValueError(
            "Retrieval metadata must be a JSON object."
        )

    files = metadata.get(
        "files"
    )

    if not isinstance(files, dict):
        raise ValueError(
            "Retrieval metadata must contain a files object."
        )

    return metadata


def retrieval_for_file(
    path: Path,
    retrieval_metadata: dict,
) -> dict[str, object]:
    """
    Return retrieval metadata for one snapshot.

    Missing records are treated explicitly as legacy snapshots.
    """

    record = retrieval_metadata[
        "files"
    ].get(
        path.as_posix()
    )

    if record is None:
        return {
            "status":
                "not_recorded_for_legacy_snapshot",

            "retrieved_at_utc":
                None,
        }

    return {
        "status":
            "recorded",

        "retrieved_at_utc":
            record.get(
                "retrieved_at_utc"
            ),

        "requested_start":
            record.get(
                "requested_start"
            ),

        "requested_end":
            record.get(
                "requested_end"
            ),

        "rows_at_retrieval":
            record.get(
                "rows"
            ),
    }


def load_etf_metadata(
    ticker: str,
    retrieval_metadata: dict,
) -> dict[str, object]:
    """Build reproducibility metadata for one ETF snapshot."""

    path = (
        TIINGO_DIR
        / f"{ticker}.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing ETF snapshot: {path}"
        )

    df = pd.read_csv(
        path
    )

    required_columns = {
        "date",
        "adjClose",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{ticker} snapshot is missing columns: "
            f"{sorted(missing_columns)}"
        )

    dates = pd.to_datetime(
        df["date"],
        utc=True,
        errors="raise",
    ).dt.tz_localize(None)

    return {
        "provider":
            "Tiingo",

        "symbol":
            ticker,

        "path":
            path.as_posix(),

        "sha256":
            file_sha256(
                path
            ),

        "rows":
            len(df),

        "actual_start":
            dates.min().date().isoformat(),

        "actual_end":
            dates.max().date().isoformat(),

        "date_column":
            "date",

        "price_column":
            "adjClose",

        "local_file_modified_at_utc":
            file_modified_at_utc(
                path
            ),

        "retrieval":
            retrieval_for_file(
                path=path,
                retrieval_metadata=(
                    retrieval_metadata
                ),
            ),
    }


def load_fred_metadata(
    retrieval_metadata: dict,
) -> dict[str, object]:
    """Build reproducibility metadata for the risk-free snapshot."""

    if not FRED_PATH.exists():
        raise FileNotFoundError(
            f"Missing FRED snapshot: {FRED_PATH}"
        )

    df = pd.read_csv(
        FRED_PATH
    )

    required_columns = {
        "date",
        "value",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "FRED snapshot is missing columns: "
            f"{sorted(missing_columns)}"
        )

    dates = pd.to_datetime(
        df["date"],
        errors="raise",
    )

    return {
        "provider":
            "Federal Reserve Economic Data (FRED)",

        "series":
            "DGS3MO",

        "path":
            FRED_PATH.as_posix(),

        "sha256":
            file_sha256(
                FRED_PATH
            ),

        "rows":
            len(df),

        "actual_start":
            dates.min().date().isoformat(),

        "actual_end":
            dates.max().date().isoformat(),

        "date_column":
            "date",

        "value_column":
            "value",

        "local_file_modified_at_utc":
            file_modified_at_utc(
                FRED_PATH
            ),

        "retrieval":
            retrieval_for_file(
                path=FRED_PATH,
                retrieval_metadata=(
                    retrieval_metadata
                ),
            ),
    }


def retrieval_summary(
    etf_snapshots: list[dict[str, object]],
    risk_free_snapshot: dict[str, object],
) -> dict[str, object]:
    """Summarize retrieval-record coverage."""

    snapshots = [
        *etf_snapshots,
        risk_free_snapshot,
    ]

    recorded = sum(
        snapshot[
            "retrieval"
        ][
            "status"
        ] == "recorded"
        for snapshot in snapshots
    )

    total = len(
        snapshots
    )

    if recorded == 0:
        status = (
            "not_recorded_for_legacy_snapshots"
        )

    elif recorded == total:
        status = (
            "recorded_for_all_snapshots"
        )

    else:
        status = (
            "partially_recorded"
        )

    return {
        "status":
            status,

        "recorded_snapshots":
            recorded,

        "total_snapshots":
            total,

        "metadata_source":
            RETRIEVAL_METADATA_PATH.as_posix(),

        "note":
            (
                "Provider retrieval timestamps are reported "
                "only when they were explicitly recorded at "
                "download time. Filesystem modification times "
                "are reported separately and are not treated "
                "as provider retrieval timestamps."
            ),
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    retrieval_metadata = (
        load_retrieval_metadata()
    )

    etf_snapshots = [
        load_etf_metadata(
            ticker=ticker,
            retrieval_metadata=(
                retrieval_metadata
            ),
        )
        for ticker in TICKERS
    ]

    risk_free_snapshot = (
        load_fred_metadata(
            retrieval_metadata=(
                retrieval_metadata
            )
        )
    )

    manifest = {
        "experiment":
            "H001",

        "manifest_schema_version":
            1,

        "manifest_generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "requested_data_range": {
            "start":
                DATA_START_DATE.date().isoformat(),

            "end":
                DATA_END_DATE.date().isoformat(),
        },

        "data_conventions": {
            "etf_price_field":
                "adjClose",

            "etf_calendar_alignment":
                (
                    "All ETF snapshots must share "
                    "exactly the same observed-session calendar."
                ),

            "risk_free_series":
                "DGS3MO",

            "risk_free_alignment":
                (
                    "FRED is left-joined to the ETF calendar "
                    "and forward-filled."
                ),

            "risk_free_return_model":
                (
                    "Previous observed annual percentage yield "
                    "divided by 100 and 252."
                ),
        },

        "retrieval_metadata":
            retrieval_summary(
                etf_snapshots=etf_snapshots,
                risk_free_snapshot=(
                    risk_free_snapshot
                ),
            ),

        "etf_snapshots":
            etf_snapshots,

        "risk_free_snapshot":
            risk_free_snapshot,

        "environment": {
            "python":
                sys.version.split()[0],

            "packages":
                package_versions(),
        },

        "git":
            git_metadata(),
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Saved manifest: {OUTPUT_PATH}"
    )

    print(
        f"ETF snapshots: "
        f"{len(etf_snapshots)}"
    )

    print(
        "Risk-free snapshot: DGS3MO"
    )

    print(
        "Retrieval metadata: "
        f"{manifest['retrieval_metadata']['status']}"
    )


if __name__ == "__main__":
    main()