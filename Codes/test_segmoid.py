# -*- coding: utf-8 -*-

from api_segmiod import perform_head_swap

def main():
    # l=["target","slide_01","slide_03","slide_07"]
    l=["target"]
    # n=["test2","test8","test9","test10","test 7"]
    n=["test2"]
    for i in l:
        for j in n:
             target_image = fr"test_data/{i}.jpg"
             face_image   = rf"test_data/{j}.jpg"
             output_image = fr"test_output/sigmoid/{i} {j}.png"
             print("🚀 Running Segmind FLUX.2 Pro head swap...")
             out = perform_head_swap(target_image, face_image, output_image)

        if out:
            print("✅ DONE:", out)
        else:
            print("❌ FAILED")

if __name__ == "__main__":
    main()
