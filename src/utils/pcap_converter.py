"""
pcap_converter.py
-----------------
Convert raw PCAP files to CICFlowMeter-style 78-feature CSVs.

This script is a wrapper that calls CICFlowMeter's jar (if installed)
or provides guidance on how to run it manually.

CICFlowMeter download: https://github.com/CanadianInstituteForCybersecurity/CICFlowMeter
"""
import argparse
import os
import subprocess
import sys

CICFLOW_JAR = os.environ.get(
    "CICFLOWMETER_JAR",
    "/opt/CICFlowMeter/CICFlowMeter.jar"
)


def convert_pcap(pcap_path: str, output_dir: str, jar_path: str = CICFLOW_JAR) -> str:
    """
    Run CICFlowMeter on a PCAP file and output a feature CSV.

    Parameters
    ----------
    pcap_path  : path to the input .pcap file
    output_dir : directory where the output CSV will be written
    jar_path   : path to the CICFlowMeter JAR

    Returns the path to the output CSV.
    """
    if not os.path.exists(jar_path):
        print(
            "\n[pcap_converter] CICFlowMeter JAR not found at:", jar_path,
            "\nDownload from: https://github.com/CanadianInstituteForCybersecurity/CICFlowMeter",
            "\nOr set the CICFLOWMETER_JAR environment variable.",
            "\n\nManual usage:",
            "\n  java -jar CICFlowMeter.jar <pcap_file> <output_dir>",
        )
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    cmd = ["java", "-jar", jar_path, pcap_path, output_dir]
    print(f"  [pcap_converter] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("  [pcap_converter] STDERR:", result.stderr)
        raise RuntimeError(f"CICFlowMeter failed for {pcap_path}")

    # CICFlowMeter writes <pcap_basename>_YYYYMMDD_HHmmss.csv
    csv_files = [
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith(".csv")
    ]
    if not csv_files:
        raise FileNotFoundError(f"No CSV output found in {output_dir}")

    latest = max(csv_files, key=os.path.getctime)
    print(f"  [pcap_converter] Output CSV: {latest}")
    return latest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert PCAP to CICFlowMeter 78-feature CSV"
    )
    parser.add_argument("--pcap",   required=True, help="Input .pcap file")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()
    convert_pcap(args.pcap, args.output)
