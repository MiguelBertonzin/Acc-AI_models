"""Strict IP audit permitting only Vitis simulation-only deadlock additions."""
import hashlib
import re
import zipfile

def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def audit_ip(out, top):
    ip=out/f"{top}_prj/solution1/impl/ip"
    syn=out/f"{top}_prj/solution1/syn/verilog"
    package=ip/f"xilinx_com_hls_{top}_1_0.zip"
    allowed={f"{top}_hls_deadlock_{suffix}" for suffix in (
        "kernel_monitor_top.vh","detector.vh","report_unit.vh","detection_unit.v","idx0_monitor.v")}
    extra=[]
    same=[]
    normalized=[]
    def nonempty(text):
        return [line for line in text.splitlines() if line.strip()]
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None
        files=[p for p in (ip/"hdl/verilog").iterdir() if p.is_file()]
        assert files and (ip/"component.xml").is_file()
        for source in syn.iterdir():
            if source.is_file() and source.suffix in {".v",".vh",".dat"}:
                assert (ip/"hdl/verilog"/source.name).is_file(),source.name
        normal_sources=[]
        for p in files:
            assert z.read(p.relative_to(ip).as_posix())==p.read_bytes(),p.name
            original=syn/p.name
            if not original.exists():
                assert p.name in allowed, f"Unexpected additional HDL: {p.name}"
                extra.append(p.name)
                continue
            if p.read_bytes()==original.read_bytes():
                same.append(p.name)
                if p.suffix==".v": normal_sources.append(p.read_text())
                continue
            assert p.name==f"{top}.v",f"Unexpected changed HDL: {p.name}"
            text=p.read_text()
            for variable,suffix in (("find_df_deadlock","deadlock_detector.vh"),("find_kernel_block","deadlock_kernel_monitor_top.vh")):
                include=f"{top}_hls_{suffix}"
                pattern=(r"reg\s+"+variable+r"\s*=\s*0;\s*// synthesis translate_off\s*`include \""+
                         re.escape(include)+r"\"\s*// synthesis translate_on")
                text,count=re.subn(pattern,"",text)
                assert count==1,(p.name,variable,count)
                assert variable not in text,(p.name,variable)
            assert nonempty(text)==nonempty(original.read_text()),f"Datapath differs: {p.name}"
            normal_sources.append(text)
            normalized.append(p.name)
        # Auxiliary .v modules must not be instantiated in the synthesis RTL.
        for name in extra:
            if name.endswith(".v"):
                modules=re.findall(r"^\s*module\s+(\w+)",(ip/"hdl/verilog"/name).read_text(),re.MULTILINE)
                assert modules,name
                for module in modules:
                    assert all(not re.search(r"\b"+re.escape(module)+r"\b",src) for src in normal_sources),module
    return {"status":"passed","package":str(package.relative_to(out)),"sha256":sha(package),
        "comparison":"exact RTL except two recognized simulation-only include blocks and unused monitor flags in top",
        "byte_identical_files":same,"normalized_top_files":normalized,
        "simulation_auxiliary_files":extra,"synthesis_datapath_matches":True}
