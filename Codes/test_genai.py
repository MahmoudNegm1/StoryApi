# runner.py
# =========================
# Simple runner for Gemini Nano Banana Pro head swap
# =========================

from api_nano_gemi import perform_head_swap
# 👆 اسم الملف اللي فيه الكود بتاعك
# لو اسم الملف مختلف عدله هنا بس

def main():
    l=["slide_10","slide_01","slide_03","slide_07"]
    for i in l:
        target_image = fr"test_data/{i}.jpg"
        face_image   = r"test_data/img01.png"
        output_image = fr"test_output/img01{i}.png"

        print("🚀 Running Gemini Nano Banana Pro head swap...")

        result = perform_head_swap(
        target_image_path=target_image,
        face_image_path=face_image,
        output_filename=output_image
    )

        if result:
         print("🎉 DONE:", result)
        else:
            print("❌ FAILED")

if __name__ == "__main__":
    main()
