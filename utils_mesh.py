import numpy as np
from skimage import measure
from scipy import ndimage

def generate_tumor_mesh_obj(segmentation_mask, volume=None):
    """
    Enhanced mesher that generates:
    1. A semi-transparent "Brain Shell".
    2. Smoothed, solid tumor sub-regions.
    """
    obj_lines = []
    obj_lines.append("# TumorVision Advanced Clinical Reconstruction")
    
    current_vert_offset = 1
    found_any = False

    # 1. GENERATE BRAIN SHELL (Envelope)
    # This captures the entire brain/head volume to provide context.
    
    try:
        shell_mask = None
        
        if volume is not None:
            # Use actual MRI intensity to find brain bounds
            # volume is likely Z-scored. Zeros are background.
            # We treat non-zero (or above small threshold) as brain.
            # Squeeze to ensure 3D (128, 128, 128)
            vol_3d = np.squeeze(volume)
            if vol_3d.ndim > 3:
                # If channels, take mean or max
                vol_3d = np.mean(vol_3d, axis=-1)
                
            shell_mask = (np.abs(vol_3d) > 0.01).astype(np.uint8)
        else:
            # Fallback to older method (bounding box of data?)
            # or just don't render shell if no volume
            shell_mask = (segmentation_mask > -1).astype(np.uint8) 

        # Apply dilation/closing to make it a solid "egg" shape
        shell_mask = ndimage.binary_closing(shell_mask, iterations=3)
        # Fill holes
        shell_mask = ndimage.binary_fill_holes(shell_mask)
        
        # Use higher step_size for shell to keep it low-poly and fast
        verts, faces, normals, values = measure.marching_cubes(shell_mask, level=0.5, step_size=3)
        
        obj_lines.append("g BrainShell")
        for v in verts: obj_lines.append(f"v {v[0]} {v[1]} {v[2]}")
        for n in normals: obj_lines.append(f"vn {n[0]} {n[1]} {n[2]}")
        for f in faces:
            v1, v2, v3 = f + current_vert_offset
            obj_lines.append(f"f {v1}//{v1} {v2}//{v2} {v3}//{v3}")
        current_vert_offset += len(verts)
        found_any = True
    except Exception as e:
        print(f"Shell generation error: {e}")
        pass

    # 2. GENERATE TUMOR REGIONS
    class_names = {
        1: "Necrotic",
        2: "Edema",
        3: "Enhancing"
    }

    for class_val, name in class_names.items():
        mask = (segmentation_mask == class_val).astype(np.uint8)
        
        if np.sum(mask) > 5:
            # VOLUMETRIC SMOOTHING
            # We apply morphological closing to fill small gaps and make it "filled"
            # as requested by the user.
            mask = ndimage.binary_closing(mask, iterations=1)
            # Dilation helps connect fragmented pieces
            mask = ndimage.binary_dilation(mask, iterations=1)
            
            try:
                verts, faces, normals, values = measure.marching_cubes(mask, level=0.5)
                
                obj_lines.append(f"g {name}")
                for v in verts: obj_lines.append(f"v {v[0]} {v[1]} {v[2]}")
                for n in normals: obj_lines.append(f"vn {n[0]} {n[1]} {n[2]}")
                for f in faces:
                    v1, v2, v3 = f + current_vert_offset
                    obj_lines.append(f"f {v1}//{v1} {v2}//{v2} {v3}//{v3}")
                current_vert_offset += len(verts)
                found_any = True
            except:
                continue

    return "\n".join(obj_lines) if found_any else None
