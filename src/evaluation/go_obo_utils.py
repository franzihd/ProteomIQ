"""
Parses data/go/go-basic.obo into GO term names, definitions and namespaces.
"""
import re
from pathlib import Path

GO_OBO_PATH = Path("data/go/go-basic.obo")

DEF_LINE_PATTERN = re.compile(r'^def:\s*"(.*)"\s*\[.*\]\s*$')


def load_go_term_info(obo_path: Path = GO_OBO_PATH) -> dict[str, dict]:
    """Returns {go_id: {"name": str, "def": str, "namespace": str}} for
    every non-obsolete term in the OBO file. "def" is "" if the term has
    no def: line (rare, but not assumed absent -- checked, not assumed)."""
    terms: dict[str, dict] = {}
    current: dict = {}
    in_term = False

    def flush():
        if current.get("id") and not current.get("is_obsolete"):
            terms[current["id"]] = {
                "name": current.get("name", ""),
                "def": current.get("def", ""),
                "namespace": current.get("namespace", ""),
            }

    with open(obo_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "[Term]":
                if in_term:
                    flush()
                in_term = True
                current = {}
            elif line == "" and in_term:
                flush()
                in_term = False
                current = {}
            elif in_term:
                if line.startswith("id: "):
                    current["id"] = line[4:]
                elif line.startswith("name: "):
                    current["name"] = line[6:]
                elif line.startswith("namespace: "):
                    current["namespace"] = line[11:]
                elif line.startswith("is_obsolete:"):
                    current["is_obsolete"] = True
                elif line.startswith("def:"):
                    m = DEF_LINE_PATTERN.match(line)
                    current["def"] = m.group(1) if m else ""
        if in_term:
            flush()

    return terms


if __name__ == "__main__":
    # Quick self-check when run directly: confirm names AND defs parse for
    # one term per aspect.
    info = load_go_term_info()
    print(f"Parsed {len(info)} non-obsolete GO terms with name/def/namespace")
    for go_id in ["GO:0003677", "GO:0008150", "GO:0005575"]:
        t = info.get(go_id)
        if t:
            print(f"\n{go_id} ({t['namespace']})")
            print(f"  name: {t['name']}")
            print(f"  def:  {t['def'][:150]}")
