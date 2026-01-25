# test_run.py
# -*- coding: utf-8 -*-

import os
from api_nano_nano import perform_head_swap

def main():
    target_image_path = r"test_data\target.jpg"
    face_image_path   = r"test_data\test 7.jpg"
    output_filename   = r"test_output\final3_3.4.jpg"

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    out = perform_head_swap(
        target_image_path=target_image_path,
        face_image_path=face_image_path,
        output_filename=output_filename
    )

    if out:
        print(f"\n✅ Done! Output: {out}")
    else:
        print("\n❌ Failed. Check logs.")

if __name__ == "__main__":
    main()
