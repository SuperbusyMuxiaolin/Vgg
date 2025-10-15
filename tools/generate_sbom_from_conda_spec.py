"""
Generate CycloneDX SBOM (JSON and XML) from a conda-style requirements.txt (name=version=build).

This script parses a conda "export"-style requirements file (not pip-style),
builds a minimal CycloneDX v1.6 SBOM including all packages as components,
and writes both JSON and XML outputs under sbom/.

Usage:
  python tools/generate_sbom_from_conda_spec.py [--input requirements.txt] [--outdir sbom]

Notes:
  - Lines starting with '#' are ignored.
  - Each package line is expected as: <name>=<version>=<build>
  - We emit a purl using the "conda" type when possible: pkg:conda/<name>@<version>?build=<build>
    (Channel is not recorded in this file; if needed, you can post-edit or extend parsing.)
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
import uuid
from typing import List, Dict, Any, Optional


CDX_SPEC_VERSION = "1.6"


def _now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def parse_conda_export_lines(lines: List[str]) -> List[Dict[str, str]]:
    """Parse conda export lines of the form name=version=build.

    Returns list of dicts with keys: name, version, build.
    Skips comments and blank lines.
    """
    pkgs: List[Dict[str, str]] = []
    pattern = re.compile(r"^(?P<name>[^=\s#]+)=(?P<version>[^=\s#]+)=(?P<build>[^\s#]+)$")
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = pattern.match(line)
        if not m:
            # Not the expected format; ignore gracefully.
            continue
        d = m.groupdict()
        pkgs.append({"name": d["name"], "version": d["version"], "build": d["build"]})
    return pkgs


def to_cyclonedx_components(packages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    comps: List[Dict[str, Any]] = []
    for p in packages:
        name = p.get("name", "").strip()
        version = p.get("version", "").strip()
        build = p.get("build", "").strip()
        # Construct a purl for conda when possible (spec may vary; keep minimal)
        purl = None
        if name and version:
            # purl type 'conda' is accepted by Package URL spec.
            # Include build as a qualifier when available.
            qualifiers = f"?build={build}" if build else ""
            purl = f"pkg:conda/{name}@{version}{qualifiers}"

        comp: Dict[str, Any] = {
            "type": "library",
            "name": name,
            "version": version or None,
        }
        if purl:
            comp["purl"] = purl

        # Add properties to preserve original build string and ecosystem
        props = []
        if build:
            props.append({"name": "cdx:conda:build", "value": build})
        props.append({"name": "cdx:ecosystem", "value": "conda"})
        if props:
            comp["properties"] = props

        comps.append(comp)
    return comps


def build_cyclonedx_bom(components: List[Dict[str, Any]], project_name: str = "Vgg") -> Dict[str, Any]:
    bom_serial_number = f"urn:uuid:{uuid.uuid4()}"
    bom: Dict[str, Any] = {
        "bomFormat": "CycloneDX",
        "specVersion": CDX_SPEC_VERSION,
        "version": 1,
        "metadata": {
            "timestamp": _now_iso(),
            "tools": [
                {
                    "vendor": "Generated",
                    "name": "generate_sbom_from_conda_spec.py",
                    "version": "1.0.0",
                }
            ],
            "component": {
                "type": "application",
                "name": project_name,
            },
        },
        "components": components,
        "serialNumber": bom_serial_number,
    }
    return bom


def write_json(bom: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bom, f, indent=2, ensure_ascii=False)


def write_xml(bom: Dict[str, Any], path: str) -> None:
    """Write a minimal CycloneDX XML representation matching the JSON content."""
    import xml.etree.ElementTree as ET

    ns = {
        "bom": "http://cyclonedx.org/schema/bom/1.6",
    }
    ET.register_namespace("", ns["bom"])  # default namespace

    bom_el = ET.Element(ET.QName(ns["bom"], "bom"), attrib={"version": "1", "serialNumber": bom.get("serialNumber", "")})

    # metadata
    md = bom.get("metadata", {})
    md_el = ET.SubElement(bom_el, ET.QName(ns["bom"], "metadata"))
    if ts := md.get("timestamp"):
        ts_el = ET.SubElement(md_el, ET.QName(ns["bom"], "timestamp"))
        ts_el.text = ts
    tools = md.get("tools", [])
    if tools:
        tools_el = ET.SubElement(md_el, ET.QName(ns["bom"], "tools"))
        for t in tools:
            tool_el = ET.SubElement(tools_el, ET.QName(ns["bom"], "tool"))
            if v := t.get("vendor"):
                v_el = ET.SubElement(tool_el, ET.QName(ns["bom"], "vendor"))
                v_el.text = v
            if n := t.get("name"):
                n_el = ET.SubElement(tool_el, ET.QName(ns["bom"], "name"))
                n_el.text = n
            if ver := t.get("version"):
                ver_el = ET.SubElement(tool_el, ET.QName(ns["bom"], "version"))
                ver_el.text = ver
    comp_md = md.get("component") or {}
    if comp_md:
        comp_el = ET.SubElement(md_el, ET.QName(ns["bom"], "component"), attrib={"type": comp_md.get("type", "application")})
        name_el = ET.SubElement(comp_el, ET.QName(ns["bom"], "name"))
        name_el.text = comp_md.get("name", "application")

    # components
    comps_el = ET.SubElement(bom_el, ET.QName(ns["bom"], "components"))
    for c in bom.get("components", []):
        c_el = ET.SubElement(comps_el, ET.QName(ns["bom"], "component"), attrib={"type": c.get("type", "library")})
        n_el = ET.SubElement(c_el, ET.QName(ns["bom"], "name"))
        n_el.text = c.get("name", "")
        if c.get("version"):
            v_el = ET.SubElement(c_el, ET.QName(ns["bom"], "version"))
            v_el.text = str(c["version"])  # ensure string
        if purl := c.get("purl"):
            purl_el = ET.SubElement(c_el, ET.QName(ns["bom"], "purl"))
            purl_el.text = purl
        # properties
        props = c.get("properties") or []
        if props:
            props_el = ET.SubElement(c_el, ET.QName(ns["bom"], "properties"))
            for pr in props:
                prop_el = ET.SubElement(props_el, ET.QName(ns["bom"], "property"), attrib={"name": pr.get("name", "")})
                prop_el.text = pr.get("value", "")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tree = ET.ElementTree(bom_el)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate CycloneDX SBOM from conda-style requirements.txt")
    parser.add_argument("--input", "-i", default="requirements.txt", help="Path to conda export-style requirements file")
    parser.add_argument("--outdir", "-o", default=os.path.join("sbom"), help="Output directory for SBOM files")
    parser.add_argument("--project-name", default="Vgg", help="Project/application name for SBOM metadata")
    args = parser.parse_args(argv)

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        return 2

    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()

    packages = parse_conda_export_lines(lines)
    if not packages:
        print("[WARN] No packages parsed from input. SBOM will contain zero components.")

    components = to_cyclonedx_components(packages)
    bom = build_cyclonedx_bom(components, project_name=args.project_name)

    os.makedirs(args.outdir, exist_ok=True)
    json_out = os.path.join(args.outdir, "bom.cdx.json")
    xml_out = os.path.join(args.outdir, "bom.cdx.xml")

    write_json(bom, json_out)
    write_xml(bom, xml_out)

    print(f"SBOM generated:\n  JSON: {json_out}\n  XML : {xml_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
