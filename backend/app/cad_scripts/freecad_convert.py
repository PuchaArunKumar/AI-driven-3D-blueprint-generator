"""FreeCAD headless script: mesh -> tessellated B-rep -> STEP/IGES/BREP.

Run by :mod:`app.providers.cad.freecad` as::

    FreeCADCmd freecad_convert.py -- --input model.stl --output model.step

Important: the shape this produces is a *faceted* B-rep built from the mesh
triangles.  It opens in CAD software and can be measured, but it is not a
parametric solid with feature history - see the note in the README.
"""

from __future__ import annotations

import argparse
import json
import sys

import FreeCAD  # noqa: F401  - provided by the FreeCAD runtime
import Mesh
import Part


def parse_args(argv: list[str]) -> argparse.Namespace:
    argv = argv[argv.index("--") + 1:] if "--" in argv else argv[1:]
    parser = argparse.ArgumentParser(prog="freecad_convert")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--sew", action="store_true", default=True)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args(list(sys.argv))
    result: dict[str, object] = {"ok": False}
    try:
        document = FreeCAD.newDocument("Conversion")
        Mesh.insert(args.input, document.Name)
        mesh_objects = [obj for obj in document.Objects if hasattr(obj, "Mesh")]
        if not mesh_objects:
            raise RuntimeError("no mesh was imported")

        shape = Part.Shape()
        topology = mesh_objects[0].Mesh.Topology
        shape.makeShapeFromMesh(topology, args.tolerance)
        if args.sew:
            shape.sewShape()

        try:
            shape = Part.makeSolid(shape)
            result["solid"] = True
        except Exception:
            # An open mesh cannot become a solid; the shell still exports.
            result["solid"] = False

        part = document.addObject("Part::Feature", "ConvertedMesh")
        part.Shape = shape
        document.recompute()
        Part.export([part], args.output)

        result.update(
            ok=True,
            faces=len(shape.Faces),
            is_valid=bool(shape.isValid()),
            freecad_version=".".join(FreeCAD.Version()[:3]),
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    print("RESULT_JSON:" + json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
