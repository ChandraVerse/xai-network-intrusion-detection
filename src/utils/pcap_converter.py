"""PCAP to CICFlowMeter CSV converter wrapper.

Wraps the CICFlowMeter CLI to convert raw PCAP files into
78-feature CSV records compatible with the XAI-NIDS pipeline.

Usage:
    from src.utils.pcap_converter import convert_pcap

    out_csv = convert_pcap(
        pcap_path="captures/network.pcap",
        out_dir="data/raw/live",
    )

Requires:
    CICFlowMeter installed and on PATH.
    See: https://github.com/ahlashkari/CICFlowMeter
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

_CFM_CMD = os.environ.get("CICFLOWMETER_CMD", "CICFlowMeter")


def _find_output_csv(out_dir: Path, stem: str) -> Path | None:
    """Return the first CSV in out_dir whose name contains *stem*."""
    candidates = sorted(out_dir.glob(f"*{stem}*.csv"))
    if not candidates:
        candidates = sorted(out_dir.glob("*.csv"))
    return candidates[-1] if candidates else None


def convert_pcap(
    pcap_path: str | Path,
    out_dir: str | Path,
    timeout: int = 300,
) -> Path:
    """Run CICFlowMeter on *pcap_path* and return the output CSV path.

    Args:
        pcap_path: Path to the input PCAP file.
        out_dir: Directory where CICFlowMeter writes the output CSV.
        timeout: Maximum seconds to wait for CICFlowMeter to finish.

    Returns:
        Path to the generated CSV file.

    Raises:
        FileNotFoundError: If pcap_path does not exist.
        RuntimeError: If CICFlowMeter exits with a non-zero status.
        FileNotFoundError: If the expected output CSV cannot be found.
    """
    pcap_path = Path(pcap_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

    cmd = [_CFM_CMD, str(pcap_path), str(out_dir)]
    log.info("Running CICFlowMeter: %s", " ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"CICFlowMeter failed (exit {result.returncode}):\n{result.stderr}"
        )

    log.info("CICFlowMeter stdout:\n%s", result.stdout)

    out_csv = _find_output_csv(out_dir, pcap_path.stem)
    if out_csv is None:
        raise FileNotFoundError(
            f"No CSV output found in {out_dir} after running CICFlowMeter."
        )

    log.info("Converted CSV written to: %s", out_csv)
    return out_csv


def batch_convert_pcap(
    pcap_dir: str | Path,
    out_dir: str | Path,
    timeout: int = 300,
) -> list[Path]:
    """Convert all PCAP files in *pcap_dir* to CSV.

    Args:
        pcap_dir: Directory containing .pcap or .pcapng files.
        out_dir: Destination directory for CSV output files.
        timeout: Per-file timeout in seconds.

    Returns:
        List of paths to the generated CSV files.
    """
    pcap_dir = Path(pcap_dir)
    pcap_files = sorted(pcap_dir.glob("*.pcap")) + sorted(pcap_dir.glob("*.pcapng"))
    if not pcap_files:
        log.warning("No PCAP files found in %s", pcap_dir)
        return []

    results = []
    for i, pcap in enumerate(pcap_files, 1):
        log.info("[%d/%d] Converting %s", i, len(pcap_files), pcap.name)
        try:
            csv_path = convert_pcap(pcap, out_dir, timeout=timeout)
            results.append(csv_path)
        except Exception as exc:
            log.error("Failed to convert %s: %s", pcap.name, exc)

    log.info("Batch complete: %d/%d converted", len(results), len(pcap_files))
    return results


def mock_convert_pcap(
    pcap_path: str | Path,
    out_dir: str | Path,
) -> Path:
    """Write a minimal stub CSV for CI/testing without CICFlowMeter installed."""
    pcap_path = Path(pcap_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{pcap_path.stem}_stub.csv"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, dir=out_dir
    ) as tmp:
        tmp.write("Label,Flow Duration\nBENIGN,12345\n")
        tmp_path = Path(tmp.name)
    tmp_path.rename(out_csv)
    log.info("Mock CSV written to: %s", out_csv)
    return out_csv
