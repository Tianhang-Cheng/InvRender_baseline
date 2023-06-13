#!/usr/bin/env python3
import os
import sys
import argparse
from math import *
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.append(str(Path(__file__).absolute().parent.parent))




def create_camera(name, data):
    """Creates a camera object from an npz file."""
    from utils import blender_utils as bu
    R = data['R']
    t = data['t']
    cvcam = bu.CVCamera(K=data['K'],
                     R=R,
                     t=t,
                     height_px=data['height'],
                     width_px=data['width'])

    bpy.ops.object.camera_add()
    cam = bpy.context.object
    cvcam.apply_to_blender(cam)
    cam['image_name'] = data['name']
    cam['image_id'] = int(data['image_id'])
    cam['original_intrinsics'] = data['K'].reshape(-1)
    cam['image_height'] = int(data['height'])
    cam['image_width'] = int(data['width'])
    cam.name = name
    return cam


def create_image_plane(name, image, K, width, height, scale=1.0):
    """Creates a mesh object from an npz file and sets up the pbr material nodes
    Args:
        name: Name of the new mesh
        image: np.ndarray with shape (H,W,3) and dtype np.uint8
        scale: scale the image plane
    
    """
    K[1,2] = height - K[1,2] # mirror y
    inv_K = np.linalg.inv(K)
    vertices = np.array([
        [0,0,1],
        [width,0,1],
        [width,height,1],
        [0,height,1],
    ], dtype=np.float32)
    vertices = (inv_K @ vertices.T).T
    vertices[:,2] = 0
    vertices *= scale


    h,w,_ = image.shape

    # vertices = np.array([
    #     [-w,-h,0],
    #     [w,-h,0],
    #     [w,h,0],
    #     [-w,h,0],
    # ], dtype=np.float32)
    # diag = np.sqrt(2)*np.sqrt(w**2 + h**2)
    # vertices *= scale/diag

    mesh = bpy.data.meshes.new(name=name)

    mesh.vertices.add(4)
    mesh.vertices.foreach_set("co", vertices.ravel())

    mesh.loops.add(4)  # this is the size of the index array not the number of triangles
    mesh.loops.foreach_set("vertex_index", [0,1,2,3])

    mesh.polygons.add(1)
    mesh.polygons.foreach_set("loop_start", [0])
    mesh.polygons.foreach_set("loop_total", [4])

    uvmap = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float32)
    uv_layer = mesh.uv_layers.new()
    uv_layer.data.foreach_set('uv', uvmap.ravel())

    mesh.update()
    mesh.validate()

    obj = bpy.data.objects.new(name, mesh)

    collection = bpy.context.collection
    collection.objects.link(obj)

    # setup materials
    # Deselect all
    bpy.ops.object.select_all(action='DESELECT')

    mat = bpy.data.materials.new("material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    mat_node = nodes['Material Output']
    bsdf_node = nodes.new('ShaderNodeBsdfPrincipled')
    mat.node_tree.links.new(bsdf_node.outputs['BSDF'],
                            mat_node.inputs['Surface'])
    bsdf_node.inputs['Alpha'].default_value = 0.5

    filename = 'albedo.jpg'
    tex = image.astype(np.float32)/255
    tex = np.concatenate([tex, np.ones_like(tex[...,:1])], axis=-1)
    tex = np.flip(tex, axis=0) # flip v axis
    img = bpy.data.images.new(filename, width=tex.shape[1], height=tex.shape[0])
    img.filepath = filename
    img.colorspace_settings.name = 'sRGB'
    img.pixels = tex.ravel()
    img.file_format = 'JPEG'
    node_tex = nodes.new("ShaderNodeTexImage")
    node_tex.image = img
    mat.node_tree.links.new(node_tex.outputs['Color'], bsdf_node.inputs['Base Color'])
        
    # viewport settings for the material
    mat.blend_method = 'BLEND'
    mat.shadow_method = 'NONE'

    obj.select_set(True)
    obj.location = (0,0,-scale)

    return obj


def create_board(name, image, width, height):
    """Creates a mesh object from an npz file and sets up the pbr material nodes
    Args:
        name: Name of the new mesh
        image: np.ndarray with shape (H,W,3) and dtype np.uint8
        width: width of the board in meter
        height: height of the board in meter
    """
    w_2 = width/2
    h_2 = height/2
    vertices = np.array([
        [-w_2,-h_2,0],
        [w_2,-h_2,0],
        [w_2,h_2,0],
        [-w_2,h_2,0],
    ], dtype=np.float32)

    mesh = bpy.data.meshes.new(name=name)

    mesh.vertices.add(4)
    mesh.vertices.foreach_set("co", vertices.ravel())

    mesh.loops.add(4)  # this is the size of the index array not the number of triangles
    mesh.loops.foreach_set("vertex_index", [0,1,2,3])

    mesh.polygons.add(1)
    mesh.polygons.foreach_set("loop_start", [0])
    mesh.polygons.foreach_set("loop_total", [4])

    uvmap = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float32)
    uv_layer = mesh.uv_layers.new()
    uv_layer.data.foreach_set('uv', uvmap.ravel())

    mesh.update()
    mesh.validate()

    obj = bpy.data.objects.new(name, mesh)

    collection = bpy.context.collection
    collection.objects.link(obj)

    # setup materials
    # Deselect all
    bpy.ops.object.select_all(action='DESELECT')

    mat = bpy.data.materials.new("material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    mat_node = nodes['Material Output']
    bsdf_node = nodes.new('ShaderNodeBsdfPrincipled')
    mat.node_tree.links.new(bsdf_node.outputs['BSDF'],
                            mat_node.inputs['Surface'])

    filename = 'albedo.jpg'
    tex = image.astype(np.float32)/255
    tex = np.concatenate([tex, np.ones_like(tex[...,:1])], axis=-1)
    tex = np.flip(tex, axis=0) # flip v axis
    img = bpy.data.images.new(filename, width=tex.shape[1], height=tex.shape[0])
    img.filepath = filename
    img.colorspace_settings.name = 'sRGB'
    img.pixels = tex.ravel()
    img.file_format = 'JPEG'
    node_tex = nodes.new("ShaderNodeTexImage")
    node_tex.image = img
    mat.node_tree.links.new(node_tex.outputs['Color'], bsdf_node.inputs['Base Color'])
        
    obj.select_set(True)
    return obj


def create_points(name, points, colors=None):
    """Creates a mesh object from an npz file and sets up the pbr material nodes
    Args:
        name: Name of the new mesh
        points: (N,3) float array
        colors: (N,3) uint8 array
    
    """
    vertices = points

    mesh = bpy.data.meshes.new(name=name)

    mesh.vertices.add(vertices.shape[0])
    mesh.vertices.foreach_set("co", vertices.ravel())

    if colors is not None:
        vertex_colors = mesh.vertex_colors.new(name='base_color')
        base_color = np.concatenate([colors.astype(np.float32)/255, np.ones((vertices.shape[0], 1), dtype=np.float32)], axis=-1)
        vertex_colors.data.foreach_set('color', base_color.ravel())

    mesh.update()
    mesh.validate()

    obj = bpy.data.objects.new(name, mesh)

    collection = bpy.context.collection
    collection.objects.link(obj)

    # setup materials
    # Deselect all
    bpy.ops.object.select_all(action='DESELECT')

    mat = bpy.data.materials.new("material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    mat_node = nodes['Material Output']
    bsdf_node = nodes.new('ShaderNodeBsdfPrincipled')
    mat.node_tree.links.new(bsdf_node.outputs['BSDF'],
                            mat_node.inputs['Surface'])

    obj.select_set(True)
    return obj




def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "This script calls blender and loads a sparse COLMAP reconstruction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,)

    parser.add_argument("path", type=Path, help="Path to the COLMAP model with the images.bin file.")
    # parser.add_argument("output_path", type=Path, help="The output path")
    parser.add_argument("--tmpdir", type=Path, help="The temporary directory to use for preparing data for blender.")
    parser.add_argument("--blender",
                        type=str,
                        default='blender',
                        help="blender command. script was tested with 3.4.1")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if '--' in sys.argv:
        args, _ = parser.parse_known_args(sys.argv[sys.argv.index("--") + 1:])
    else:
        args = parser.parse_args()
    print(args)
    return args


def main_python():
    import subprocess
    import pycolmap
    import cv2
    import fitz # pymupdf
    from tqdm import tqdm
    from PIL import Image
    from io import BytesIO

    args = parse_args()

    cmd = [args.blender, '-P', __file__]
    cmd.append('--')

    for x in sys.argv[1:]:
        cmd.append(x)

    with TemporaryDirectory() as tmpdir:
        cmd += ['--tmpdir', tmpdir]

        recon = pycolmap.Reconstruction(args.path/'recon'/'0')
        images = {'image_id': [], 'K': [], 'R': [], 't': [], 'name': [], 'height': [], 'width': []}
        registered_image_names = set()
        for im_id, im in tqdm(recon.images.items()):
            cam = recon.cameras[im.camera_id]
            registered_image_names.add(im.name)
            images['K'].append(cam.calibration_matrix())
            images['R'].append(pycolmap.qvec_to_rotmat(im.qvec))
            images['t'].append(im.tvec)
            images['image_id'].append(im_id)
            images['name'].append(im.name)
            images['height'].append(cam.height)
            images['width'].append(cam.width)

            colors = cv2.imread(str(args.path/'images'/im.name))
            max_wh = max(colors.shape)
            target_size = tuple((np.asarray(colors.shape[:2])/max_wh*256).astype(int)[::-1])
            colors = cv2.resize(colors, target_size)
            images[f'.images_{im.name}'] = colors[...,[2,1,0]]
            # if len(images['image_id']) == 2:
            #     break

        # get images that have not been registered
        image_paths = [x for x in (args.path/'images').glob('*.jpg') if x.name not in registered_image_names]
        for im_path in tqdm(image_paths):
            cam = recon.cameras[1]
            colors = cv2.imread(str(im_path))
            images['K'].append(cam.calibration_matrix())
            images['R'].append(np.eye(3).astype(np.float32))
            images['t'].append(np.zeros((3,), dtype=np.float32))
            images['image_id'].append(-1)
            images['name'].append(im_path.name)
            images['height'].append(colors.shape[0])
            images['width'].append(colors.shape[1])
            max_wh = max(colors.shape)
            target_size = tuple((np.asarray(colors.shape[:2])/max_wh*256).astype(int)[::-1])
            colors = cv2.resize(colors, target_size)
            images[f'.images_{im_path.name}'] = colors[...,[2,1,0]]
        

        np.savez(Path(tmpdir)/'images.npz', **images)

        points, colors = zip(*[(p.xyz, p.color) for p in recon.points3D.values()])
        points = np.stack(points)
        colors = np.stack(colors)
        np.savez(Path(tmpdir)/'points.npz', points=points, colors=colors)

        # create texture for the april tag board
        pdfpath = Path(__file__).resolve().parent.parent/'calibration'/'targets'/'atag_board.pdf'
        print(pdfpath)
        pdf = fitz.open(pdfpath)
        page = pdf.load_page(0)
        pixmap = page.get_pixmap()
        pngbytes = pixmap.tobytes()
        bytesio = BytesIO(pngbytes)
        im = Image.open(bytesio)
        arr = np.asarray(im)
        board_tex = cv2.resize(arr, (512,512))
        point_in_meter = 0.0003528
        np.savez(Path(tmpdir)/'board.npz', tex=board_tex, size=(page.mediabox.width*point_in_meter, page.mediabox.height*point_in_meter))


        print('==================================================================')
        print('Calling blender with')
        print(' '.join(cmd))
        print('==================================================================')
        status = subprocess.check_call(cmd, shell=False)
        print(status)


def main_blender():
    """This is the main function if the script is called with blender"""
    print('main_blender', flush=True)
    import bpy
    from bpy import context as C
    import utils.blender_utils as bu
    bpy.context.preferences.view.show_splash = False

    if bpy.app.version < (3,4,1):
        raise Exception("This script requires at least Blender 3.4.1")

    args = parse_args()

    bu.SavePosePriors.register_op()
    bu.UtilsPanel.register_panel()

    bu.delete_all_objects()

    images = np.load(args.tmpdir/'images.npz')

    for i, (im_name, idx) in enumerate(sorted(zip(images['name'],np.arange(len(images['image_id']))))):
        print(i, im_name)
        d = {k: v[idx] for k,v in images.items() if not k.startswith('.')}
        cam = create_camera(f'image_{i}', d)
        cam['image_path'] = str(args.path.resolve()/'images'/d['name'])
        scale = 0.02
        cam.data.display_size = scale
        cam.data.clip_start = 1e-6

        m = bpy.context.scene.timeline_markers.new(f'F_{i}', frame=i)
        m.camera = cam

        implane = create_image_plane(f'implane_{i}', images[f'.images_{d["name"]}'], d['K'], d['width'], d['height'], scale)
        implane.parent = cam
        bpy.context.scene.render.resolution_x = d['width']
        bpy.context.scene.render.resolution_y = d['height']

    board = np.load(args.tmpdir/'board.npz')
    board_obj = create_board('board', board['tex'], board['size'][0], board['size'][1])

    points = np.load(args.tmpdir/'points.npz')
    points_obj = create_points('points', points['points'], points['colors'])
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = len(images['image_id'])-1


if __name__ == '__main__':
    try:
        import bpy
        is_blender = True
    except:
        is_blender = False

    if is_blender:
        main_blender()
    else:
        sys.exit(main_python())
