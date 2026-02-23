#!/usr/bin/env python3

from pathlib import Path
import nd2
import numpy as np

exp = "20250912"
base = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground")
nd2_path = next((base / "raw_image_data" / "YX1" / exp).glob("*.nd2"))

print(f"Processing: {nd2_path}")
print(f"File size: {nd2_path.stat().st_size / (1024**4):.2f} TB")

with nd2.ND2File(str(nd2_path)) as f:
    print("ND2 sizes:", f.sizes)
    print("Channel info:")

    # Get metadata for first frame
    try:
        md = f.frame_metadata(0)
        print(f"Channel name: '{md.channels[0].channel.name}'")
        print(f"Channel index: {md.channels[0].channel.index}")
        print(f"Modality flags: {md.channels[0].microscope.modalityFlags}")
    except Exception as e:
        print(f"Error getting metadata: {e}")

    # Extract a sample image using dask to avoid loading everything
    print("\nExtracting sample image...")
    try:
        arr = f.to_dask()   # lazy, doesn't load everything yet
        print(f"Array shape: {arr.shape}")
        print(f"Array dtype: {arr.dtype}")

        # Get a 2D slice from the middle of the dataset
        # Try different positions to see if there's actual data
        positions_to_try = [
            (0, 0),      # First timepoint, first position
            (0, 47),     # First timepoint, middle position
            (56, 47),    # Middle timepoint, middle position
        ]

        for i, (t, p) in enumerate(positions_to_try):
            print(f"\nTrying position T={t}, P={p}:")

            # Extract max projection across Z
            slice2d = arr[t, p, :, :, :].max(axis=0)  # (Y, X)
            img = slice2d.compute()  # pulls ONLY this slice into memory

            print(f"  Image shape: {img.shape}")
            print(f"  Image dtype: {img.dtype}")
            print(f"  Min value: {img.min()}")
            print(f"  Max value: {img.max()}")
            print(f"  Mean value: {img.mean():.2f}")
            print(f"  Non-zero pixels: {np.count_nonzero(img)}/{img.size}")

            # Save the image
            output_path = f"/net/trapnell/vol1/home/mdcolon/proj/morphseq/sample_t{t}_p{p}.png"

            # Normalize to 0-255 for PNG
            if img.max() > img.min():
                img_norm = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            else:
                img_norm = img.astype(np.uint8)

            import imageio
            imageio.imwrite(output_path, img_norm)
            print(f"  Saved: {output_path}")

            # If we found non-empty data, we can stop
            if np.count_nonzero(img) > 0:
                print(f"  ✅ Found non-empty data at T={t}, P={p}")
                break
        else:
            print("  ⚠️  All tested positions appear to be empty")

    except Exception as e:
        print(f"Error extracting image: {e}")
        import traceback
        traceback.print_exc()

print("\nDone!")