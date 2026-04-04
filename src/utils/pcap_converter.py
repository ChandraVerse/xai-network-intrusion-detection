"""PCAP to CICFlowMeter CSV converter wrapper.

Wraps the CICFlowMeter CLI to convert raw PCAP files into
78-feature CSV records compatible with the XAI-NIDS pipeline.

Prerequisites:
    - Java 8+ installed and on PATH
    - CICFlowMeter JAR downloaded from:
      https://github.com/ahlashkari/CICFlowMeter
    - Set env var: CICFLOWMETER_JAR=/path/to/CICFlowMeter.jar

Usage (CLI):
    python src/utils/pcap_converter.py \\
        --input  /path/to/capture.pcap \\
        --output data/raw/
"""

import argparse
import logging
import os
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_JAR = os.environ.get(
    "CICFLOWMETER_JAR",
    "tools/CICFlowMeter.jar",
)


def convert(
    pcap_path: str | Path,
    output_dir: str | Path,
    jar_path: str | Path = DEFAULT_JAR,
) -> Path:
    """Convert a PCAP file to a CICFlowMeter feature CSV.

    Args:
        pcap_path:  Path to the .pcap input file.
        output_dir: Directory where the output CSV will be written.
        jar_path:   Path to the CICFlowMeter JAR file.

    Returns:
        Path to the generated CSV file.

    Raises:
        FileNotFoundError: If pcap_path or jar_path does not exist.
        subprocess.CalledProcessError: If CICFlowMeter exits with an error.
    """
    pcap_path = Path(pcap_path)
    output_dir = Path(output_dir)
    jar_path = Path(jar_path)

    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP not found: {pcap_path}")
    if not jar_path.exists():
        raise FileNotFoundError(
            f"CICFlowMeter JAR not found at {jar_path}.\n"
            "Set CICFLOWMETER_JAR env var or pass --jar argument."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "java", "-jar", str(jar_path),
        str(pcap_path),
        str(output_dir),
    ]
    log.info("Running CICFlowMeter: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log.error("CICFlowMeter stderr:\n%s", result.stderr)
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )

    log.info("CICFlowMeter stdout:\n%s", result.stdout)

    # CICFlowMeter names the output after the input PCAP
    expected_csv = output_dir / (pcap_path.stem + "_ISCX.csv")
    if expected_csv.exists():
        log.info("Output CSV → %s", expected_csv)
        return expected_csv

    # Fallback: find any CSV in output_dir newer than when we started
    csvs = sorted(output_dir.glob("*.csv"))
    if csvs:
        log.info("Output CSV → %s", csvs[-1])
        return csvs[-1]

    raise FileNotFoundError(
        f"CICFlowMeter completed but no CSV found in {output_dir}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert PCAP to CICFlowMeter CSV")
    p.add_argument("--input",  required=True, help="Path to input .pcap file")
    p.add_argument("--output", default="data/raw/",  help="Output directory")
    p.add_argument("--jar",    default=DEFAULT_JAR,  help="Path to CICFlowMeter JAR")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = convert(args.input, args.output, args.jar)
    print(f"Generated: {out}")


if __name__ == "__main__":
    main()
