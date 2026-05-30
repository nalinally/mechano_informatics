import cv2
import os
from common import *

def get_image_files(dir):
    files = os.listdir(dir)
    return [f for f in files if "jpeg" in f or "png" in f or "jpg" in f or "bmp" in f]


def main():
    img_files = get_image_files(original_img_dir)

    print(f"{len(img_files)} img files found.\n{img_files}")

    for img_file in img_files:
        
        input_file = original_img_dir + img_file
        output_file = processed_img_dir + ".".join(img_file.split(".")[:-1]) + processed_suffix + "." + img_file.split(".")[-1]
        
        print(f"processing {input_file}...")
        
        img = cv2.imread(input_file, 0)

        img = cv2.resize(img, (img_size[0], img_size[1]))

        ret2, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

        print(f"writing to {output_file}.")
        cv2.imwrite(output_file, img)
        # cv2.imshow("otsu", img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        
        print(f"writed to {output_file}.")

if __name__=="__main__":
    main()