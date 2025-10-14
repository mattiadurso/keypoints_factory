import numpy as np
import os
import h5py
import glob
import pycolmap
from pathlib import Path
from scipy.spatial.transform import Rotation as R_scipy


def build_colmap_from_h5(base_path, output_dir):
    """
    Build COLMAP format model files from h5 calibration and depth files.

    Args:
        base_path: Base path containing calibration/ and depth_maps/ folders
        output_dir: Output directory for COLMAP files
    """
    os.makedirs(output_dir, exist_ok=True)

    calibration_dir = os.path.join(base_path, "calibration")
    depth_dir = os.path.join(base_path, "depth_maps")

    # Get all calibration files
    calib_files = sorted(glob.glob(os.path.join(calibration_dir, "calibration_*.h5")))

    # Extract image identifiers from calibration files
    image_ids = [Path(f).stem.replace("calibration_", "") for f in calib_files]

    # 1. Write cameras.txt
    with open(f"{output_dir}/cameras.txt", "w") as cam_file:
        cam_file.write("# Camera list with one line of data per camera:\n")
        cam_file.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")

        for i, (calib_file, img_id) in enumerate(zip(calib_files, image_ids)):
            with h5py.File(calib_file, "r") as f:
                K = f["K"][:]

            # Get depth map to extract image dimensions
            depth_file = os.path.join(depth_dir, f"{img_id}.h5")
            with h5py.File(depth_file, "r") as d:
                depth = d["depth"][:]
                h, w = depth.shape[:2]

            # Extract intrinsics
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            cam_file.write(f"{i+1} PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")

    # 2. Write images.txt
    with open(f"{output_dir}/images.txt", "w") as img_file:
        img_file.write("# Image list with two lines of data per image:\n")
        img_file.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        img_file.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for i, (calib_file, img_id) in enumerate(zip(calib_files, image_ids)):
            with h5py.File(calib_file, "r") as f:
                R = f["R"][:]
                T = f["T"][:] if "T" in f else f["t"][:]  # Handle both 'T' and 't'

                # Convert rotation matrix to quaternion (qw, qx, qy, qz)
                if "q" in f:
                    q = f["q"][:]
                    # Check if quaternion needs reordering (some formats use qx,qy,qz,qw)
                    if len(q) == 4:
                        quat = q  # Assume already in qw,qx,qy,qz format
                    else:
                        quat = q
                else:
                    # Convert from rotation matrix
                    quat = R_scipy.from_matrix(R).as_quat()[
                        [3, 0, 1, 2]
                    ]  # Convert to qw,qx,qy,qz

            image_name = f"{img_id}.jpg"  # Adjust extension as needed

            img_file.write(f"{i+1} {quat[0]} {quat[1]} {quat[2]} {quat[3]} ")
            img_file.write(f"{T[0]} {T[1]} {T[2]} {i+1} {image_name}\n")
            img_file.write("\n")  # Empty line for 2D points

    # 3. Write points3D.txt (from depth maps)
    with open(f"{output_dir}/points3D.txt", "w") as pts_file:
        pts_file.write("# 3D point list with one line of data per point:\n")
        pts_file.write(
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        )

    print(f"COLMAP model saved to {output_dir}")
    print(f"Processed {len(calib_files)} images")


#####################################################################################
# Process all scenes in the phototourism dataset
#####################################################################################
path = Path(
    "/home/mattia/Desktop/Repos/wrapper_factory/benchmarks_2D/imc/data/phototourism/"
)

for scene in os.listdir(path):
    # Example usage:
    base_path = path / scene / "set_100"
    output_dir = path / scene / "set_100/colmap/sparse"

    build_colmap_from_h5(base_path, output_dir)

    images_path = base_path / "images"
    rec_path = base_path / "colmap/sparse"

    rec = pycolmap.Reconstruction(rec_path)

    temp_path = base_path / "temp"
    os.makedirs(temp_path, exist_ok=True)
    os.system(f"mv {images_path}/* {temp_path}/")

    for image in sorted(os.listdir(temp_path)):
        # image from temp_path to images_path
        os.system(f"mv {temp_path}/{image} {images_path}/{image}")
        camera = rec.find_image_with_name(image).camera

        # load camera intrinsics
        camera_model = camera.model.name
        params = [
            str(p) for p in camera.params
        ]  # convert to strings for COLMAP command

        # extract features using COLMAP
        os.system(
            f'colmap feature_extractor \
            --database_path {base_path}/colmap/database.db \
            --image_path {images_path} \
            --ImageReader.camera_model {camera_model} \
            --ImageReader.camera_params "{",".join(params)}" \
            '
        )
    os.rmdir(temp_path)
    os.system(
        f"colmap exhaustive_matcher --database_path {base_path}/colmap/database.db"
    )
    os.system(
        f"colmap point_triangulator \
            --input_path {base_path}/colmap/sparse \
            --database_path {base_path}/colmap/database.db \
            --image_path {images_path} \
            --output_path {base_path}/colmap/sparse \
            --clear_points 0 \
            --refine_intrinsics 0 \
            --Mapper.ba_refine_focal_length 0 \
            --Mapper.ba_refine_extra_params 0 \
            --Mapper.ba_refine_principal_point 0 \
            --Mapper.ba_refine_sensor_from_rig 0 \
            --Mapper.ba_local_max_num_iterations 1 \
            --Mapper.ba_global_max_num_iterations 1 \
            --Mapper.ba_global_function_tolerance 1.0 \
            --Mapper.ba_local_function_tolerance 1.0 \
            "
    )
