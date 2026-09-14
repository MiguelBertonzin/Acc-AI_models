#!/usr/bin/env python3
"""Monta um IP-XACT somente-Verilog usando o RTL FIFO final do Vitis HLS."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


FIFO_RE = re.compile(r"resnet8_resource_fifo_opt_fifo_w\d+_d\d+_[A-Za-z0-9_]+")
STALE_SCAFFOLD_FILES = (
    "vivado.log",
    "vivado.jou",
    "xilinx_com_hls_resnet8_resource_fifo_opt_1_0.zip",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corrige o IP HLS stale substituindo-o pelo syn/verilog final."
    )
    parser.add_argument("--scaffold", type=Path, required=True)
    parser.add_argument("--syn-rtl", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--version", default="1.1")
    parser.add_argument("--archive", type=Path)
    return parser.parse_args()


def load_namespaces(component: Path) -> dict[str, str]:
    namespaces: dict[str, str] = {}
    for _, pair in ET.iterparse(component, events=("start-ns",)):
        prefix, uri = pair
        namespaces[prefix or ""] = uri
    for prefix, uri in namespaces.items():
        try:
            ET.register_namespace(prefix, uri)
        except ValueError:
            pass
    return namespaces


def rewrite_component(component: Path, version_text: str, rtl_dir: Path) -> None:
    namespaces = load_namespaces(component)
    tree = ET.parse(component)
    root = tree.getroot()
    spirit_uri = namespaces.get("spirit", root.tag.split("}")[0].lstrip("{"))

    def q(name: str) -> str:
        return f"{{{spirit_uri}}}{name}"

    version = root.find(q("version"))
    if version is None:
        raise RuntimeError("spirit:version principal não encontrado")
    version.text = version_text

    model = root.find(q("model"))
    if model is not None:
        views = model.find(q("views"))
        if views is not None:
            for view in list(views):
                texts = [(node.text or "").strip().lower() for node in view.iter()]
                if any("vhdl" in text for text in texts):
                    views.remove(view)

    file_sets = root.find(q("fileSets"))
    if file_sets is None:
        raise RuntimeError("spirit:fileSets não encontrado")

    rtl_files = sorted(path for path in rtl_dir.iterdir() if path.suffix in {".v", ".dat"})
    if not rtl_files:
        raise RuntimeError(f"Nenhum RTL encontrado em {rtl_dir}")

    for file_set in list(file_sets):
        set_name = (file_set.findtext(q("name")) or "").lower()
        if "vhdl" in set_name:
            file_sets.remove(file_set)
            continue

        if "verilog" not in set_name:
            continue

        for file_node in list(file_set.findall(q("file"))):
            name = (file_node.findtext(q("name")) or "").replace("\\", "/")
            if name.startswith("hdl/verilog/"):
                file_set.remove(file_node)

        for path in rtl_files:
            file_node = ET.SubElement(file_set, q("file"))
            name_node = ET.SubElement(file_node, q("name"))
            name_node.text = f"hdl/verilog/{path.name}"
            if path.suffix == ".v":
                type_node = ET.SubElement(file_node, q("fileType"))
                type_node.text = "verilogSource"
            else:
                type_node = ET.SubElement(file_node, q("userFileType"))
                type_node.text = "mif"

    tree.write(component, encoding="UTF-8", xml_declaration=True)


def main() -> None:
    args = parse_args()
    scaffold = args.scaffold.resolve()
    syn_rtl = args.syn_rtl.resolve()
    out = args.out.resolve()

    component_source = scaffold / "component.xml"
    top_source = syn_rtl / "resnet8_resource_fifo_opt.v"
    for required in (component_source, top_source):
        if not required.is_file():
            raise FileNotFoundError(required)

    top_text = top_source.read_text(errors="replace")
    if re.search(r"fifo_w\d+_d4096", top_text):
        raise RuntimeError("O top syn/verilog ainda instancia FIFO d4096")

    if out.exists():
        raise FileExistsError(f"Saída já existe; recusa em sobrescrever: {out}")

    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(scaffold, out)

    removed_scaffold_artifacts = []
    xil_dir = out / ".Xil"
    if xil_dir.exists():
        shutil.rmtree(xil_dir)
        removed_scaffold_artifacts.append(".Xil/")
    for name in STALE_SCAFFOLD_FILES:
        stale = out / name
        if stale.exists():
            stale.unlink()
            removed_scaffold_artifacts.append(name)

    packaged_rtl = out / "hdl/verilog"
    if packaged_rtl.exists():
        shutil.rmtree(packaged_rtl)
    shutil.copytree(syn_rtl, packaged_rtl)

    vhdl_dir = out / "hdl/vhdl"
    if vhdl_dir.exists():
        shutil.rmtree(vhdl_dir)

    removed_d4096 = []
    for path in sorted(packaged_rtl.glob("*fifo_w*_d4096*.v")):
        removed_d4096.append(path.name)
        path.unlink()

    component = out / "component.xml"
    rewrite_component(component, args.version, packaged_rtl)

    packaged_top = packaged_rtl / top_source.name
    packaged_text = packaged_top.read_text(errors="replace")
    referenced_fifos = sorted(set(FIFO_RE.findall(packaged_text)))
    available_modules = {
        match.group(1)
        for path in packaged_rtl.glob("*.v")
        for match in re.finditer(r"\bmodule\s+(\S+)", path.read_text(errors="replace"))
    }
    missing_fifos = sorted(set(referenced_fifos) - available_modules)
    if missing_fifos:
        raise RuntimeError(f"Módulos FIFO ausentes no pacote: {missing_fifos}")
    if re.search(r"fifo_w\d+_d4096", packaged_text):
        raise RuntimeError("O top empacotado ainda instancia d4096")

    manifest = {
        "vlnv": f"xilinx.com:hls:resnet8_resource_fifo_opt:{args.version}",
        "source_scaffold": str(scaffold),
        "source_syn_rtl": str(syn_rtl),
        "component_sha256": sha256(component),
        "top_sha256": sha256(packaged_top),
        "top_matches_syn_rtl": sha256(packaged_top) == sha256(top_source),
        "top_instantiates_d4096": False,
        "removed_scaffold_artifacts": removed_scaffold_artifacts,
        "removed_residual_d4096_modules": removed_d4096,
        "referenced_fifo_modules": referenced_fifos,
        "verilog_file_count": len(list(packaged_rtl.glob("*.v"))),
        "data_file_count": len(list(packaged_rtl.glob("*.dat"))),
    }
    (out / "ip_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    if args.archive:
        archive = args.archive.resolve()
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive_base = archive.with_suffix("")
        generated = Path(
            shutil.make_archive(str(archive_base), "zip", root_dir=out.parent, base_dir=out.name)
        )
        if generated != archive:
            if archive.exists():
                raise FileExistsError(archive)
            generated.rename(archive)
        manifest["archive"] = str(archive)
        manifest["archive_sha256"] = sha256(archive)
        (out / "ip_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
