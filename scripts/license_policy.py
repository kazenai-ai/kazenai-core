#!/usr/bin/env python3
# Canonical copy: kazenai-control-topology/scripts/license_policy.py (tests in
# tests/test_license_policy.py). Each repo carries an identical copy in scripts/.
"""KazenAI license gate and third-party notice generator.

One shared file, copied into each repository's ``scripts/``. Requires Python
3.11+ and ``license-expression`` (AboutCode, Apache-2.0); nothing else.

  check    Evaluate a Syft JSON SBOM against licence-gate/license-policy.toml. SPDX
           expressions are evaluated (OR = best allowed alternative, AND = every
           operand, WITH = base licence plus exception). Exit 1 on any denial.
  notices  Write third-party notices from the packages actually resolved: a
           Python environment (by lock names) or a Node project's production
           dependency graph. Each package keeps its own licence and NOTICE text.

Package class comes from the purl (pkg:deb / pkg:apk / pkg:rpm / binary = OS),
never from SBOM dependency edges. See docs in licence-gate/license-policy.toml.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from license_expression import (
    AND,
    OR,
    ExpressionError,
    LicenseSymbol,
    LicenseWithExceptionSymbol,
    get_spdx_licensing,
)

LICENSING = get_spdx_licensing()

# Severity order. OR takes the minimum, AND the maximum.
CATEGORIES = ("permissive", "weak_copyleft", "lgpl", "gpl", "agpl", "source_available", "proprietary", "unknown")
SEVERITY = {name: i for i, name in enumerate(CATEGORIES)}
OS_TYPES = {"deb", "apk", "rpm", "alpm", "binary", "linux-kernel"}


# --------------------------------------------------------------------------- policy


@dataclass
class Policy:
    categories: dict[str, set[str]]
    linking_exceptions: set[str]
    aliases: dict[str, str]
    first_party: list[dict[str, Any]]
    exceptions: list[dict[str, Any]] = field(default_factory=list)
    build_only: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, policy_path: Path, exceptions_path: Path | None) -> "Policy":
        data = tomllib.loads(policy_path.read_text(encoding="utf-8"))
        cats = {name: {k.lower() for k in data["categories"].get(name, [])} for name in CATEGORIES}
        aliases = {k.lower(): v for k, v in data.get("aliases", {}).items()}
        exc: list[dict[str, Any]] = []
        if exceptions_path and exceptions_path.is_file():
            exc = tomllib.loads(exceptions_path.read_text(encoding="utf-8")).get("exception", [])
        return cls(
            categories=cats,
            linking_exceptions={e.lower() for e in data.get("linking_exceptions", [])},
            aliases=aliases,
            first_party=data.get("first_party", []),
            exceptions=exc,
            build_only=data.get("build_only", []),
        )

    def category_of_key(self, key: str) -> str:
        k = key.lower()
        for name in CATEGORIES:
            if k in self.categories[name]:
                return name
        if k.startswith("licenseref-"):
            return "proprietary"
        for prefix, name in (("agpl-", "agpl"), ("lgpl-", "lgpl"), ("gpl-", "gpl")):
            if k.startswith(prefix):
                return name
        return "unknown"


# --------------------------------------------------------------------------- expressions


def normalize(raw: str, policy: Policy) -> str | None:
    """Return a parseable SPDX expression for a raw licence string, else None."""
    text = (raw or "").strip()
    if not text:
        return None
    alias = policy.aliases.get(text.lower())
    if alias:
        return alias
    try:
        parsed = LICENSING.parse(text, validate=False)
    except ExpressionError:
        return None
    if parsed is None:
        return None
    # For "X WITH Y" only the base licence X is categorised; evaluate() handles Y.
    keys = [s.license_symbol.key if isinstance(s, LicenseWithExceptionSymbol) else s.key
            for s in LICENSING.license_symbols(parsed, decompose=False)]
    unknown = [k for k in keys if policy.category_of_key(k) == "unknown"]
    if unknown:
        return None
    return str(parsed)


def evaluate(expression: str, policy: Policy) -> tuple[str, list[str]]:
    """Category of an SPDX expression and the licence keys that set it."""
    parsed = LICENSING.parse(expression, validate=False)

    def walk(node) -> tuple[str, list[str]]:
        if isinstance(node, LicenseWithExceptionSymbol):
            base = policy.category_of_key(node.license_symbol.key)
            exc = node.exception_symbol.key.lower()
            if base == "gpl" and exc in policy.linking_exceptions:
                base = "lgpl"
            return base, [str(node)]
        if isinstance(node, LicenseSymbol):
            return policy.category_of_key(node.key), [node.key]
        results = [walk(arg) for arg in node.args]
        if isinstance(node, OR):
            best = min(results, key=lambda r: SEVERITY[r[0]])
            return best
        if isinstance(node, AND):
            worst = max(results, key=lambda r: SEVERITY[r[0]])
            keys = [k for r in results for k in r[1]]
            return worst[0], keys
        return "unknown", [str(node)]

    return walk(parsed)


# --------------------------------------------------------------------------- inputs


def ecosystem(artifact: dict[str, Any]) -> str:
    purl = artifact.get("purl") or ""
    m = re.match(r"pkg:([a-z0-9.+-]+)/", purl)
    if m:
        return {"pypi": "pypi", "npm": "npm", "deb": "deb", "apk": "apk", "rpm": "rpm", "golang": "go"}.get(m.group(1), m.group(1))
    return artifact.get("type") or "unknown"


def is_os_package(artifact: dict[str, Any]) -> bool:
    return ecosystem(artifact) in OS_TYPES or (artifact.get("type") in OS_TYPES)


def lock_digests(paths: Iterable[Path]) -> dict[tuple[str, str, str], set[str] | str]:
    """(ecosystem, name, version) -> sha256 set (pip lock) or 'link' (npm local link)."""
    out: dict[tuple[str, str, str], Any] = {}
    for path in paths:
        if path.name == "package-lock.json":
            data = json.loads(path.read_text(encoding="utf-8"))
            root_name = ((data.get("packages") or {}).get("") or {}).get("name") or data.get("name")
            if root_name:
                out[("npm", root_name.lower(), "*")] = "link"  # the project itself
            for loc, meta in (data.get("packages") or {}).items():
                if not loc:
                    continue
                name = meta.get("name") or loc.split("node_modules/")[-1]
                if meta.get("link"):
                    out[("npm", name.lower(), "*")] = "link"
                    target = data["packages"].get(meta.get("resolved", ""), {})
                    out[("npm", (target.get("name") or name).lower(), target.get("version", "*"))] = "link"
            continue
        current: tuple[str, str, str] | None = None
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;\\]+)", line)
            if m:
                current = ("pypi", re.sub(r"[-_.]+", "-", m.group(1)).lower(), m.group(2))
                out.setdefault(current, set())
            for h in re.findall(r"--hash=sha256:([0-9a-f]{64})", line):
                if current:
                    out[current].add(h)
    return out


def version_matches(version: str, spec: str) -> bool:
    spec = (spec or "*").strip()
    if spec in ("*", ""):
        return True

    def key(v: str) -> tuple:
        return tuple(int(p) if p.isdigit() else p for p in re.split(r"[.\-+]", v))

    for clause in (c.strip() for c in spec.split(",")):
        m = re.match(r"^(==|>=|<=|>|<|!=)?\s*(.+)$", clause)
        op, ref = (m.group(1) or "=="), m.group(2)
        try:
            a, b = key(version), key(ref)
            ok = {"==": a == b, ">=": a >= b, "<=": a <= b, ">": a > b, "<": a < b, "!=": a != b}[op]
        except TypeError:
            ok = (version == ref) if op == "==" else False
        if not ok:
            return False
    return True


# --------------------------------------------------------------------------- check


def check(args: argparse.Namespace) -> int:
    policy = Policy.load(Path(args.policy), Path(args.exceptions) if args.exceptions else None)
    locks = lock_digests(Path(p) for p in (args.lock or []))
    today = dt.date.fromisoformat(args.today) if args.today else dt.date.today()
    findings: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for sbom_path in args.syft:
        sbom = json.loads(Path(sbom_path).read_text(encoding="utf-8"))
        for art in sbom.get("artifacts", []):
            eco, name, version = ecosystem(art), art.get("name", ""), art.get("version", "")
            ident = (eco, name.lower(), version)
            if ident in seen:
                continue
            seen.add(ident)
            raws = sorted({(l.get("spdxExpression") or l.get("value") or "").strip() for l in art.get("licenses", [])} - {""})
            os_pkg = is_os_package(art)
            finding = {"ecosystem": eco, "name": name, "version": version, "os_package": os_pkg,
                       "declared": raws, "expression": None, "category": None, "decision": None, "notes": []}

            normalized = [normalize(r, policy) for r in raws]
            if raws and all(normalized):
                distinct = sorted(set(normalized))
                expression = distinct[0] if len(distinct) == 1 else " AND ".join(f"({e})" for e in distinct)
                if len(distinct) > 1:
                    finding["notes"].append("multiple licence statements combined with AND")
                category, _keys = evaluate(expression, policy)
            else:
                expression, category = None, "unknown"
            if raws == ["UNLICENSED"]:
                expression, category = "UNLICENSED", "proprietary"

            # First party: name + ecosystem, plus a verified digest or local link.
            fp = next((f for f in policy.first_party
                       if f["name"].lower() == name.lower() and f.get("ecosystem", eco) == eco), None)
            if fp and category in ("proprietary", "unknown"):
                lock_entry = locks.get((eco, re.sub(r"[-_.]+", "-", name).lower() if eco == "pypi" else name.lower(), version)) \
                    or locks.get((eco, name.lower(), "*"))
                want = (fp.get("sha256") or "").lower()
                if fp.get("local_link") and lock_entry == "link":
                    category, expression = "first_party", expression or "LicenseRef-KazenAI-Proprietary"
                    finding["notes"].append(f"first-party local workspace package ({fp.get('repository')})")
                elif want and isinstance(lock_entry, set) and want in lock_entry:
                    category, expression = "first_party", expression or "LicenseRef-KazenAI-Proprietary"
                    finding["notes"].append(f"first-party, digest {want[:16]} verified against lock ({fp.get('repository')})")
                else:
                    finding["notes"].append("first-party name matched but digest/link not verified: treated as third-party")

            # Reviewed exceptions replace the expression with a verified one.
            exc = next((e for e in policy.exceptions
                        if fnmatch.fnmatchcase(name.lower(), e["name"].lower()) and e.get("ecosystem", eco) in (eco, "any")
                        and version_matches(version, e.get("versions", "*"))), None)
            if exc and category not in ("first_party",):
                if dt.date.fromisoformat(str(exc["review_by"])) < today:
                    finding["notes"].append(f"exception expired {exc['review_by']}")
                    category = "expired_exception"
                else:
                    expression = exc["spdx"]
                    category, _ = evaluate(expression, policy)
                    finding["notes"].append(f"exception: {exc['reason']} (approver {exc['approver']}, review by {exc['review_by']})")

            finding["expression"], finding["category"] = expression, category
            finding["decision"] = decide(category, os_pkg, args.context)
            build = next((b for b in policy.build_only if fnmatch.fnmatchcase(name.lower(), b["name"].lower())
                          and b.get("ecosystem", eco) == eco), None)
            if build and finding["decision"] != "allow" and category not in ("agpl", "source_available"):
                finding["decision"] = "allow"
                finding["notes"].append(f"build-only, not distributed: {build['reason']}")
            findings.append(finding)

    denied = [f for f in findings if f["decision"] == "deny"]
    conditional = [f for f in findings if f["decision"] == "conditional"]
    report = {
        "context": args.context,
        "sboms": args.syft,
        "summary": {
            "components": len(findings),
            "allowed": sum(f["decision"] == "allow" for f in findings),
            "conditional": len(conditional),
            "denied": len(denied),
        },
        "denied": denied,
        "conditional": conditional,
        "components": findings,
    }
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"license gate [{args.context}]: {report['summary']}")
    for f in denied:
        print(f"  DENY  {f['ecosystem']}:{f['name']}@{f['version']} {f['category']} {f['expression'] or f['declared']} {f['notes']}")
    for f in conditional:
        print(f"  COND  {f['ecosystem']}:{f['name']}@{f['version']} {f['category']} {f['expression'] or f['declared']}")
    return 1 if denied else 0


def decide(category: str, os_pkg: bool, context: str) -> str:
    if category in ("permissive", "first_party"):
        return "allow"
    if category in ("agpl", "source_available", "expired_exception"):
        return "deny"
    if category == "weak_copyleft":
        return "conditional"  # unmodified; notice must give the source location
    if os_pkg and context == "hosted-image":
        # Hosted-only images are not distributed. GPL/LGPL/unknown OS packages are
        # recorded as self-hosted gate items (corresponding source + notices).
        return "conditional" if category in ("lgpl", "gpl", "unknown") else "deny"
    return "deny"  # lgpl/gpl/proprietary/unknown application packages


# --------------------------------------------------------------------------- notices

PY_HELPER = r"""
import importlib.metadata as md, json, re, sys
want = set(json.loads(sys.stdin.read()))
norm = lambda n: re.sub(r"[-_.]+", "-", n).lower()
out = []
for dist in md.distributions():
    name = dist.metadata["Name"] or ""
    if norm(name) not in want:
        continue
    files = {}
    for f in dist.files or []:
        base = f.name.lower()
        if any(base.startswith(p) for p in ("license", "licence", "copying", "notice", "authors")) and "dist-info" in str(f.parent):
            # Read the located file; a missing licence file must fail, not vanish.
            files[str(f).split("dist-info/", 1)[-1]] = open(f.locate(), encoding="utf-8", errors="replace").read()
    lic = dist.metadata.get("License-Expression") or dist.metadata.get("License") or ""
    classifiers = [c.split("::")[-1].strip() for c in (dist.metadata.get_all("Classifier") or []) if c.startswith("License")]
    out.append({"name": name, "version": dist.version, "license": lic.strip().splitlines()[0] if lic.strip() else "",
                "classifiers": classifiers, "home": dist.metadata.get("Home-page") or "", "files": files})
print(json.dumps(out))
"""


def python_packages(args: argparse.Namespace) -> list[dict[str, Any]]:
    names: set[str] = set()
    for lock in args.lock or []:
        for (eco, name, _v) in lock_digests([Path(lock)]):
            if eco == "pypi":
                names.add(name)
    exclude = {re.sub(r"[-_.]+", "-", n).lower() for n in (args.exclude or [])}
    names -= exclude
    if args.docker_image:
        cmd = ["docker", "run", "--rm", "-i", "--entrypoint", "python", args.docker_image, "-I", "-c", PY_HELPER]
    else:
        cmd = [args.python, "-I", "-c", PY_HELPER]
    proc = subprocess.run(cmd, input=json.dumps(sorted(names)), capture_output=True, text=True, check=True)
    pkgs = json.loads(proc.stdout)
    missing = names - {re.sub(r"[-_.]+", "-", p["name"]).lower() for p in pkgs}
    conditional = set()
    for lock in args.lock or []:
        for line in Path(lock).read_text(encoding="utf-8").splitlines():
            m = re.match(r"^([A-Za-z0-9_.\-]+)==[^\s;\\]+\s*;", line)
            if m:  # environment marker, e.g. ; sys_platform == 'win32'
                conditional.add(re.sub(r"[-_.]+", "-", m.group(1)).lower())
    unexpected = sorted(missing - conditional)
    if unexpected:
        raise SystemExit(f"locked packages not installed in the target environment: {unexpected}")
    return pkgs


def node_packages(root: Path) -> list[dict[str, Any]]:
    lock = json.loads((root / "package-lock.json").read_text(encoding="utf-8"))
    out = []
    for loc, meta in sorted((lock.get("packages") or {}).items()):
        if not loc or meta.get("dev") or meta.get("devOptional") or meta.get("link") or not loc.startswith("node_modules/") and "/node_modules/" not in loc:
            continue
        if meta.get("optional"):
            # Platform-specific optional packages differ per OS/CPU, so list them
            # from lock metadata only: the notices stay identical on every machine.
            out.append({"name": loc.split("node_modules/")[-1], "version": meta.get("version", ""),
                        "license": str(meta.get("license") or ""), "classifiers": [], "home": "",
                        "files": {}, "optional": True})
            continue
        pkg_dir = root / loc
        if not pkg_dir.is_dir():
            raise SystemExit(f"{loc} is a production dependency but is not installed; run npm ci first")
        pj = json.loads((pkg_dir / "package.json").read_text(encoding="utf-8"))
        lic = meta.get("license") or pj.get("license") or pj.get("licenses") or ""
        if isinstance(lic, dict):
            lic = lic.get("type", "")
        if isinstance(lic, list):
            lic = " OR ".join(x.get("type", "") if isinstance(x, dict) else str(x) for x in lic)
        files = {}
        for f in sorted(pkg_dir.iterdir()):
            if f.is_file() and re.match(r"^(licen[cs]e|copying|notice|authors)", f.name, re.I):
                files[f.name] = f.read_text(encoding="utf-8", errors="replace")
        repo = pj.get("repository")
        if isinstance(repo, dict):
            repo = repo.get("url", "")
        out.append({"name": pj.get("name") or loc.split("node_modules/")[-1], "version": pj.get("version", meta.get("version", "")),
                    "license": str(lic), "classifiers": [], "home": pj.get("homepage") or repo or "", "files": files})
    return out


def render_notices(title: str, pkgs: list[dict[str, Any]], policy: Policy, source_note: str, extra: str) -> str:
    lines = [
        f"THIRD-PARTY NOTICES — {title}",
        "",
        "This file is generated from the packages actually resolved for this artifact",
        f"({source_note}). Each package keeps its own licence text, copyright lines",
        "and NOTICE file; identical licence texts are deliberately not merged.",
        "KazenAI's own code is under the licence in the repository's LICENSE file.",
        "",
    ]
    seen: set[tuple[str, str]] = set()
    for p in sorted(pkgs, key=lambda x: (x["name"].lower(), x["version"])):
        key = (p["name"].lower(), p["version"])
        if key in seen:
            continue
        seen.add(key)
        declared = p["license"] or ", ".join(p["classifiers"]) or "not declared"
        expr = normalize(p["license"], policy) if p["license"] else None
        category = evaluate(expr, policy)[0] if expr else "unknown"
        lines += ["=" * 78, f"{p['name']} {p['version']}", f"License: {declared}"]
        if p["home"]:
            lines.append(f"Project: {p['home']}")
        if category == "weak_copyleft":
            src = (f"https://pypi.org/project/{p['name']}/{p['version']}/#files" if not p["home"] or "npm" not in p["home"]
                   else p["home"])
            lines.append(f"Source code (MPL/weak copyleft, unmodified): {src}")
        lines.append("")
        if p.get("optional"):
            lines += ["(Optional platform-specific package, listed from the lockfile; its licence text",
                      " is in the package distribution for the platform where it is installed.)", ""]
        elif p["files"]:
            for fname, text in sorted(p["files"].items()):
                lines += [f"--- {fname} ---", text.rstrip(), ""]
        else:
            lines += ["(The package ships no licence file; the declared licence above applies.)", ""]
    if extra:
        lines += ["=" * 78, "ADDITIONAL NOTICES (code and assets not tracked by a package manager)", "", extra.rstrip(), ""]
    return "\n".join(lines) + "\n"


def notices(args: argparse.Namespace) -> int:
    policy = Policy.load(Path(args.policy), None)
    if args.node_root:
        pkgs = node_packages(Path(args.node_root))
        source = f"production dependency graph of {Path(args.node_root).name}/package-lock.json"
    else:
        pkgs = python_packages(args)
        locks = ", ".join(Path(l).name for l in args.lock or [])
        source = f"runtime closure pinned in {locks}"
    extra = Path(args.extra).read_text(encoding="utf-8") if args.extra and Path(args.extra).is_file() else ""
    text = render_notices(args.title, pkgs, policy, source, extra)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"wrote {args.out}: {len(pkgs)} packages, sha256 {hashlib.sha256(text.encode()).hexdigest()[:16]}")
    return 0


def validate(args: argparse.Namespace) -> int:
    """Strict CycloneDX JSON schema validation (needs cyclonedx-python-lib[json-validation])."""
    from cyclonedx.schema import SchemaVersion
    from cyclonedx.validation.json import JsonStrictValidator

    versions = {"1.5": SchemaVersion.V1_5, "1.6": SchemaVersion.V1_6}
    failed = 0
    for path in args.cdx:
        text = Path(path).read_text(encoding="utf-8")
        spec = json.loads(text).get("specVersion", "1.5")
        error = JsonStrictValidator(versions.get(spec, SchemaVersion.V1_5)).validate_str(text)
        print(f"{'INVALID' if error else 'valid  '} {path} (CycloneDX {spec})" + (f": {error}" if error else ""))
        failed += bool(error)
    return 1 if failed else 0


# --------------------------------------------------------------------------- cli


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="evaluate a Syft JSON SBOM")
    c.add_argument("--syft", action="append", required=True, help="Syft JSON SBOM (repeatable)")
    c.add_argument("--context", choices=["hosted-image", "library", "frontend", "service-env"], required=True)
    c.add_argument("--policy", default="licence-gate/license-policy.toml")
    c.add_argument("--exceptions", default="licence-gate/license-exceptions.toml")
    c.add_argument("--lock", action="append", help="requirements.lock / package-lock.json for first-party digests")
    c.add_argument("--report", help="write a JSON report")
    c.add_argument("--today", help="override today's date (tests)")
    n = sub.add_parser("notices", help="generate third-party notices")
    n.add_argument("--title", required=True)
    n.add_argument("--out", required=True)
    n.add_argument("--policy", default="licence-gate/license-policy.toml")
    n.add_argument("--lock", action="append", help="requirements.lock (Python): notices cover these packages")
    n.add_argument("--exclude", action="append", help="first-party package names to leave out")
    n.add_argument("--python", default=sys.executable, help="interpreter of the environment to read")
    n.add_argument("--docker-image", help="read the Python environment inside this image instead")
    n.add_argument("--node-root", help="Node project root (production deps from package-lock.json)")
    n.add_argument("--extra", help="hand-maintained notices for untracked code/assets")
    v = sub.add_parser("validate", help="strict CycloneDX schema validation")
    v.add_argument("cdx", nargs="+", help="CycloneDX JSON files")
    args = ap.parse_args(argv)
    return {"check": check, "notices": notices, "validate": validate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
