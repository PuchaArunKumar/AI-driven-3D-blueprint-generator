"""Blender background script: import a mesh, tidy it, save a .blend.

Run by :mod:`app.providers.cad.blender` as::

    blender --background --factory-startup --python process_model.py -- \
        --input model.glb --output model.blend [--export out.obj ...]

Everything after ``--`` is parsed here; Blender itself ignores it.  The script
prints a single ``RESULT_JSON:{...}`` line so the caller can parse the outcome
without scraping Blender's console banner.

Written against the Blender 4.x/5.x operator names: the legacy
``import_mesh.stl`` / ``export_scene.obj`` operators used by the original
research notebook were removed in Blender 4.0.
"""

from __future__ import annotations

import argparse
import json
import sys

import bpy  # provided by the Blender runtime


def parse_args(argv: list[str]) -> argparse.Namespace:
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    parser = argparse.ArgumentParser(prog="process_model")
    parser.add_argument("--input", required=True, help="mesh file to import")
    parser.add_argument("--output", default="", help="path for the saved .blend")
    parser.add_argument("--export", action="append", default=[],
                        help="additional file to export (repeatable)")
    parser.add_argument("--material", default="GeneratedMaterial")
    parser.add_argument("--no-normalise", action="store_true",
                        help="keep the imported transform as-is")
    return parser.parse_args(argv)


def reset_scene() -> None:
    """Start from a genuinely empty scene, with no default cube or lamp."""
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_mesh(path: str) -> None:
    """Import ``path`` using the operator matching its extension."""
    lowered = path.lower()
    if lowered.endswith((".glb", ".gltf")):
        bpy.ops.import_scene.gltf(filepath=path)
    elif lowered.endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=path)
    elif lowered.endswith(".stl"):
        # Blender 4.0+ replaced import_mesh.stl with this operator.
        if hasattr(bpy.ops.wm, "stl_import"):
            bpy.ops.wm.stl_import(filepath=path)
        else:  # pragma: no cover - Blender 3.x fallback
            bpy.ops.import_mesh.stl(filepath=path)
    elif lowered.endswith(".ply"):
        if hasattr(bpy.ops.wm, "ply_import"):
            bpy.ops.wm.ply_import(filepath=path)
        else:  # pragma: no cover - Blender 3.x fallback
            bpy.ops.import_mesh.ply(filepath=path)
    else:
        raise ValueError(f"unsupported import format: {path}")


def mesh_objects() -> list[bpy.types.Object]:
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def repair_and_normalise(objects: list[bpy.types.Object], normalise: bool) -> dict[str, int]:
    """Weld vertices, drop loose geometry, recalculate normals, apply transforms."""
    import bmesh  # imported here so the module loads outside Blender for linting

    totals = {"vertices": 0, "faces": 0, "removed_vertices": 0}
    for obj in objects:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        before = len(bm.verts)

        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
        bmesh.ops.dissolve_degenerate(bm, dist=1e-6, edges=bm.edges)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        # Delete vertices that belong to no face - they break exporters.
        loose = [vert for vert in bm.verts if not vert.link_faces]
        if loose:
            bmesh.ops.delete(bm, geom=loose, context="VERTS")

        totals["removed_vertices"] += before - len(bm.verts)
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

        totals["vertices"] += len(mesh.vertices)
        totals["faces"] += len(mesh.polygons)

    if normalise and objects:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = objects[0]
        # Bake rotation/scale into the mesh data and centre the origin.
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    return totals


def apply_vertex_colours(objects: list[bpy.types.Object], name: str) -> int:
    """Wire each mesh's colour attribute into its material's Base Colour.

    The reconstruction stores colour per vertex, and the glTF importer always
    creates *some* material, so an "only if it has no material" check silently
    skips every imported mesh - which is why the saved .blend used to open grey
    while the same model looked correct in the web viewer.

    This instead guarantees the link exists, creating the material only when
    one is genuinely missing. Returns how many objects were coloured.
    """
    coloured = 0
    for obj in objects:
        colour_attributes = getattr(obj.data, "color_attributes", None)

        if not obj.data.materials:
            obj.data.materials.append(bpy.data.materials.new(name=name))

        material = obj.data.materials[0]
        if material is None:
            material = bpy.data.materials.new(name=name)
            obj.data.materials[0] = material
        material.use_nodes = True

        tree = material.node_tree
        principled = next(
            (node for node in tree.nodes if node.type == "BSDF_PRINCIPLED"), None
        )
        if principled is None:
            continue

        principled.inputs["Roughness"].default_value = 0.45
        if not colour_attributes:
            continue

        base_colour = principled.inputs["Base Color"]
        already = any(
            link.from_node.type in {"VERTEX_COLOR", "ATTRIBUTE"}
            for link in base_colour.links
        )
        if not already:
            # Replace whatever drives Base Colour with the mesh's own colours.
            for link in list(base_colour.links):
                tree.links.remove(link)
            attribute = tree.nodes.new("ShaderNodeVertexColor")
            attribute.layer_name = colour_attributes[0].name
            attribute.location = (principled.location.x - 320,
                                  principled.location.y + 120)
            tree.links.new(attribute.outputs["Color"], base_colour)

        # Solid viewport shading reads diffuse_color rather than the node tree,
        # so set it whether or not the link was already there - otherwise the
        # file still opens grey in the default viewport.
        try:
            data = colour_attributes[0].data
            samples = min(len(data), 512)
            if samples:
                total = [0.0, 0.0, 0.0]
                for index in range(samples):
                    value = data[index].color
                    total[0] += value[0]
                    total[1] += value[1]
                    total[2] += value[2]
                material.diffuse_color = (
                    total[0] / samples, total[1] / samples, total[2] / samples, 1.0,
                )
        except Exception:  # viewport colour is cosmetic; never fail the export
            pass

        coloured += 1
    return coloured


def export_to(path: str) -> None:
    """Export the whole scene to ``path`` using the format's operator."""
    lowered = path.lower()
    if lowered.endswith(".glb"):
        bpy.ops.export_scene.gltf(filepath=path, export_format="GLB")
    elif lowered.endswith(".gltf"):
        bpy.ops.export_scene.gltf(filepath=path, export_format="GLTF_SEPARATE")
    elif lowered.endswith(".obj"):
        bpy.ops.wm.obj_export(filepath=path)
    elif lowered.endswith(".stl"):
        bpy.ops.wm.stl_export(filepath=path)
    elif lowered.endswith(".ply"):
        bpy.ops.wm.ply_export(filepath=path)
    elif lowered.endswith(".fbx"):
        bpy.ops.export_scene.fbx(filepath=path)
    else:
        raise ValueError(f"unsupported export format: {path}")


def main() -> int:
    args = parse_args(list(sys.argv))
    result: dict[str, object] = {"ok": False, "exports": []}
    try:
        reset_scene()
        import_mesh(args.input)
        objects = mesh_objects()
        if not objects:
            raise RuntimeError("no mesh objects were imported")

        stats = repair_and_normalise(objects, normalise=not args.no_normalise)
        result["coloured_objects"] = apply_vertex_colours(objects, args.material)

        for target in args.export:
            export_to(target)
            result["exports"].append(target)

        if args.output:
            bpy.ops.wm.save_as_mainfile(filepath=args.output)
            result["blend"] = args.output

        result.update(ok=True, objects=len(objects), **stats)
        result["blender_version"] = bpy.app.version_string
    except Exception as exc:  # reported back to the caller as JSON
        result["error"] = f"{type(exc).__name__}: {exc}"

    print("RESULT_JSON:" + json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
