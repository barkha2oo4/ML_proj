---
jupyter:
  colab:
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown id="zzI3trnHzrzX"}
You are tasked with creating a photo stylization program using OpenCV.
Your program should take in a single image and produce an artistic
output where the main subject is highlighted and the background is
stylized. Requirements:

1.  Input & Output

-   The program should read an image file.
-   It should save and display the final result.

1.  Processing Steps Your solution should include the following:

-   Image Enhancement: Improve global contrast using CLAHE.
-   Mask Creation: Automatically identify and isolate the main subject
    by generating a mask (largest contour or foreground).
-   Stylization:
-   Apply edge-preserving smoothing (bilateral filter) on the subject.
-   Desaturate and slightly blur the background.
-   Overlay stylized edges (from Canny edge detection) as an artistic
    outline.
-   Blending: Combine subject and background smoothly using the mask.

1.  Final Output

-   Display a montage showing:
-   Original image
-   Subject mask preview
-   Enhanced image
-   Final stylized result Constraints:
-   You must use only OpenCV and NumPy.
-   No camera input or pretrained models (e.g., face detection).
-   The solution should run in under one minute per image. Deliverables:
-   A Python script named selective_stylizer.py.
-   When run, it should generate and display a result image saved as
    stylized_result.jpg. Example (expected effect):
-   The background should appear muted and blurred.
-   The subject should appear smooth and vibrant.
-   Artistic edge outlines should highlight details.
:::

::: {.cell .code execution_count="2" id="0EwfYCfGzone"}
``` python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```
:::

::: {.cell .code execution_count="10" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":740}" id="Rpgl49Ca0Xo6" outputId="9b2d8daa-c049-472e-999c-42462b27f056"}
``` python
img_path1='/content/maskman.jpg'
img_path2='/content/scenary.jpg'
img1=cv2.imread(img_path1)
img2=cv2.imread(img_path2)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
height, width = img2.shape[:2]
img1 = cv2.resize(img1,(width,height),interpolation=cv2.INTER_AREA)
plt.imshow(img1)
plt.title('Mask Man')
plt.axis('off')
plt.show()
print()
print()

img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
plt.imshow(img2)
plt.title(' Guy Sitting under a Tree')
plt.axis('off')
plt.show()
print()
cv2.imwrite('img1.jpg', img1)
cv2.imwrite('img2.jpg',img2)
```

::: {.output .display_data}
![](48c8b1555dad54ac1d12870b629f162e5d8e7a23.png)
:::

::: {.output .stream .stdout}
:::

::: {.output .display_data}
![](fbb3b60faae2b99daaf5ac77a48ed753afe02838.png)
:::

::: {.output .stream .stdout}
:::

::: {.output .execute_result execution_count="10"}
    True
:::
:::

::: {.cell .code execution_count="18" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="sN3vl8KP1g5b" outputId="596943e2-7700-48d0-8d85-5f679799f819"}
``` python
# image enhancement using CLAHE
img1=cv2.imread(img_path1,0)
img2=cv2.imread(img_path2,0)
equ = cv2.equalizeHist(img1)
print("--"*10 + "\n IMG 1   \n"+ "--"*10)
print(img1)
print("--"*10 + "\n EQU   \n"+ "--"*10)
print(equ)
print("--"*10)
equ1 = cv2.equalizeHist(img2)
print("--"*10 + "\n IMG 2   \n"+ "--"*10)
print(img2)
print("--"*10 + "\n EQU   \n"+ "--"*10)
print(equ1)
```

::: {.output .stream .stdout}
    --------------------
     IMG 1   
    --------------------
    [[26 26 26 ... 21 22 22]
     [26 26 26 ... 21 22 22]
     [26 26 26 ... 22 22 22]
     ...
     [30 30 30 ... 29 28 28]
     [30 30 30 ... 29 28 28]
     [30 30 30 ... 29 28 28]]
    --------------------
     EQU   
    --------------------
    [[33 33 33 ... 23 24 24]
     [33 33 33 ... 23 24 24]
     [33 33 33 ... 24 24 24]
     ...
     [57 57 57 ... 48 42 42]
     [57 57 57 ... 48 42 42]
     [57 57 57 ... 48 42 42]]
    --------------------
    --------------------
     IMG 2   
    --------------------
    [[198 104 108 ... 110 116 125]
     [193  99 104 ...  97 103 112]
     [191  98 102 ...  97 103 112]
     ...
     [  1   1   1 ...   0   0  22]
     [  1   1   1 ...   0   2  16]
     [  1   1   1 ...   0   2  16]]
    --------------------
     EQU   
    --------------------
    [[237  51  53 ...  55  61  77]
     [222  48  51 ...  48  50  57]
     [214  48  50 ...  48  50  57]
     ...
     [ 22  22  22 ...   0   0  42]
     [ 22  22  22 ...   0  26  41]
     [ 22  22  22 ...   0  26  41]]
:::
:::

::: {.cell .code execution_count="26" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":881}" id="x4p9fb7A6fu8" outputId="fcec8068-975c-4ce2-9f64-3111f1d7ab02"}
``` python
plt.hist(img1.flat, bins=100, range=(0, 255))
plt.hist(equ.flat, bins=100, range=(0, 255))

```

::: {.output .execute_result execution_count="26"}
    (array([ 420.,  352.,  130.,  482.,  197.,  419.,  213.,  336.,  166.,
             381.,  402.,  213.,  394.,    0.,  466.,    0.,  658.,    0.,
             740.,    0.,    0.,    0., 1122.,    0.,    0.,    0.,    0.,
            1669.,    0.,    0.,    0., 1097.,    0.,    0., 1127.,    0.,
               0.,    0.,    0.,    0., 1500.,    0.,    0.,    0.,    0.,
               0.,    0., 2246.,    0.,    0.,    0.,    0., 1630.,    0.,
               0.,    0.,    0.,    0.,    0.,    0., 2450.,    0.,    0.,
            1053.,    0.,    0.,  795.,  426.,  308.,  265.,  271.,  290.,
             344.,  258.,  482.,  262.,  360.,  212.,  418.,  226.,  396.,
             382.,  237.,  315.,  229.,  384.,  274.,  316.,  258.,  408.,
             329.,  270.,  347.,  242.,  380.,  221.,  398.,  233.,  360.,
             321.]),
     array([  0.  ,   2.55,   5.1 ,   7.65,  10.2 ,  12.75,  15.3 ,  17.85,
             20.4 ,  22.95,  25.5 ,  28.05,  30.6 ,  33.15,  35.7 ,  38.25,
             40.8 ,  43.35,  45.9 ,  48.45,  51.  ,  53.55,  56.1 ,  58.65,
             61.2 ,  63.75,  66.3 ,  68.85,  71.4 ,  73.95,  76.5 ,  79.05,
             81.6 ,  84.15,  86.7 ,  89.25,  91.8 ,  94.35,  96.9 ,  99.45,
            102.  , 104.55, 107.1 , 109.65, 112.2 , 114.75, 117.3 , 119.85,
            122.4 , 124.95, 127.5 , 130.05, 132.6 , 135.15, 137.7 , 140.25,
            142.8 , 145.35, 147.9 , 150.45, 153.  , 155.55, 158.1 , 160.65,
            163.2 , 165.75, 168.3 , 170.85, 173.4 , 175.95, 178.5 , 181.05,
            183.6 , 186.15, 188.7 , 191.25, 193.8 , 196.35, 198.9 , 201.45,
            204.  , 206.55, 209.1 , 211.65, 214.2 , 216.75, 219.3 , 221.85,
            224.4 , 226.95, 229.5 , 232.05, 234.6 , 237.15, 239.7 , 242.25,
            244.8 , 247.35, 249.9 , 252.45, 255.  ]),
     <BarContainer object of 100 artists>)
:::

::: {.output .display_data}
![](84a0206e0fc54a70856d158bc9ea191a11464b54.png)
:::
:::

::: {.cell .code execution_count="27" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":847}" id="nd_p47UN8RLb" outputId="ce450a14-712a-4c11-936b-4b347acc577b"}
``` python
clahe = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(8,8))
cl_img = clahe.apply(img1)
plt.hist(cl_img.flat, bins=100, range=(100, 255))
```

::: {.output .execute_result execution_count="27"}
    (array([145., 147.,  84., 123.,  69., 146.,  64., 108.,  56., 115., 103.,
             59., 122.,  52., 130.,  52., 125.,  57., 135.,  65., 115., 110.,
             53., 106.,  76., 110.,  55., 104.,  55., 136., 116.,  57., 127.,
             50.,  89.,  60., 122.,  54., 114.,  57., 113., 104.,  61., 101.,
             48., 117.,  67., 133.,  63., 112., 114.,  43., 100.,  61., 113.,
             51.,  94.,  53., 100.,  38.,  87.,  84.,  41.,  87.,  53.,  75.,
             46.,  78.,  41.,  87.,  78.,  45.,  99.,  42.,  67.,  40.,  68.,
             37.,  64.,  22.,  64.,  62.,  22.,  47.,  19.,  66.,  20.,  39.,
             11.,  33.,  42.,  24.,  32.,  16.,  25.,   8.,  19.,  10.,   8.,
             10.]),
     array([100.  , 101.55, 103.1 , 104.65, 106.2 , 107.75, 109.3 , 110.85,
            112.4 , 113.95, 115.5 , 117.05, 118.6 , 120.15, 121.7 , 123.25,
            124.8 , 126.35, 127.9 , 129.45, 131.  , 132.55, 134.1 , 135.65,
            137.2 , 138.75, 140.3 , 141.85, 143.4 , 144.95, 146.5 , 148.05,
            149.6 , 151.15, 152.7 , 154.25, 155.8 , 157.35, 158.9 , 160.45,
            162.  , 163.55, 165.1 , 166.65, 168.2 , 169.75, 171.3 , 172.85,
            174.4 , 175.95, 177.5 , 179.05, 180.6 , 182.15, 183.7 , 185.25,
            186.8 , 188.35, 189.9 , 191.45, 193.  , 194.55, 196.1 , 197.65,
            199.2 , 200.75, 202.3 , 203.85, 205.4 , 206.95, 208.5 , 210.05,
            211.6 , 213.15, 214.7 , 216.25, 217.8 , 219.35, 220.9 , 222.45,
            224.  , 225.55, 227.1 , 228.65, 230.2 , 231.75, 233.3 , 234.85,
            236.4 , 237.95, 239.5 , 241.05, 242.6 , 244.15, 245.7 , 247.25,
            248.8 , 250.35, 251.9 , 253.45, 255.  ]),
     <BarContainer object of 100 artists>)
:::

::: {.output .display_data}
![](ecbaa5ea2ea98927c59f8773aad1921fb69f52e7.png)
:::
:::

::: {.cell .code execution_count="21" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":881}" id="GhMNoC-b69yq" outputId="316bb251-65c1-4884-8fe6-6f7b5ee79987"}
``` python
plt.hist(img2.flat, bins=100, range=(0, 255))
plt.hist(equ1.flat, bins=100, range=(0, 255))
```

::: {.output .execute_result execution_count="21"}
    (array([1519.,    0.,    0.,    0.,    0.,    0.,    0.,    0., 4138.,
               0.,  796.,  739.,  524.,  398.,  683.,  449.,  587.,  389.,
             493.,  451.,  566.,  423.,  447.,  647.,  234.,  271.,    0.,
            1340.,  355.,  588.,  528.,  476.,  276.,  654.,  354.,  413.,
             827.,  203.,  451.,  424.,  833.,  382.,  450.,  455.,  488.,
             739.,  375.,  321.,  314.,  691.,  802.,  261.,  575.,  325.,
             403.,  440.,  776.,  452.,    0.,  685.,  697.,  565.,  518.,
             453.,  537.,  447.,    0.,  979.,    0.,  823.,  644.,  343.,
             718.,    0.,  673.,  437.,  429.,  474.,  910.,    0.,  842.,
             684.,  405.,  552.,    0.,  448.,    0.,  991.,  648.,  628.,
             373.,  611.,  622.,    0.,  850.,  480.,  607.,  211.,  784.,
             527.]),
     array([  0.  ,   2.55,   5.1 ,   7.65,  10.2 ,  12.75,  15.3 ,  17.85,
             20.4 ,  22.95,  25.5 ,  28.05,  30.6 ,  33.15,  35.7 ,  38.25,
             40.8 ,  43.35,  45.9 ,  48.45,  51.  ,  53.55,  56.1 ,  58.65,
             61.2 ,  63.75,  66.3 ,  68.85,  71.4 ,  73.95,  76.5 ,  79.05,
             81.6 ,  84.15,  86.7 ,  89.25,  91.8 ,  94.35,  96.9 ,  99.45,
            102.  , 104.55, 107.1 , 109.65, 112.2 , 114.75, 117.3 , 119.85,
            122.4 , 124.95, 127.5 , 130.05, 132.6 , 135.15, 137.7 , 140.25,
            142.8 , 145.35, 147.9 , 150.45, 153.  , 155.55, 158.1 , 160.65,
            163.2 , 165.75, 168.3 , 170.85, 173.4 , 175.95, 178.5 , 181.05,
            183.6 , 186.15, 188.7 , 191.25, 193.8 , 196.35, 198.9 , 201.45,
            204.  , 206.55, 209.1 , 211.65, 214.2 , 216.75, 219.3 , 221.85,
            224.4 , 226.95, 229.5 , 232.05, 234.6 , 237.15, 239.7 , 242.25,
            244.8 , 247.35, 249.9 , 252.45, 255.  ]),
     <BarContainer object of 100 artists>)
:::

::: {.output .display_data}
![](f56d9f98b6a5215db3b495e9aae3ea20f37a7207.png)
:::
:::

::: {.cell .code execution_count="28" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":847}" id="Byu35iiB8bQO" outputId="a247dded-943f-4aa3-a34e-cd03c2411a92"}
``` python
clahe = cv2.createCLAHE(clipLimit =2.0, tileGridSize=(8,8))
cl_img1 = clahe.apply(img2)
plt.hist(cl_img.flat, bins=100, range=(100, 255))
```

::: {.output .execute_result execution_count="28"}
    (array([145., 147.,  84., 123.,  69., 146.,  64., 108.,  56., 115., 103.,
             59., 122.,  52., 130.,  52., 125.,  57., 135.,  65., 115., 110.,
             53., 106.,  76., 110.,  55., 104.,  55., 136., 116.,  57., 127.,
             50.,  89.,  60., 122.,  54., 114.,  57., 113., 104.,  61., 101.,
             48., 117.,  67., 133.,  63., 112., 114.,  43., 100.,  61., 113.,
             51.,  94.,  53., 100.,  38.,  87.,  84.,  41.,  87.,  53.,  75.,
             46.,  78.,  41.,  87.,  78.,  45.,  99.,  42.,  67.,  40.,  68.,
             37.,  64.,  22.,  64.,  62.,  22.,  47.,  19.,  66.,  20.,  39.,
             11.,  33.,  42.,  24.,  32.,  16.,  25.,   8.,  19.,  10.,   8.,
             10.]),
     array([100.  , 101.55, 103.1 , 104.65, 106.2 , 107.75, 109.3 , 110.85,
            112.4 , 113.95, 115.5 , 117.05, 118.6 , 120.15, 121.7 , 123.25,
            124.8 , 126.35, 127.9 , 129.45, 131.  , 132.55, 134.1 , 135.65,
            137.2 , 138.75, 140.3 , 141.85, 143.4 , 144.95, 146.5 , 148.05,
            149.6 , 151.15, 152.7 , 154.25, 155.8 , 157.35, 158.9 , 160.45,
            162.  , 163.55, 165.1 , 166.65, 168.2 , 169.75, 171.3 , 172.85,
            174.4 , 175.95, 177.5 , 179.05, 180.6 , 182.15, 183.7 , 185.25,
            186.8 , 188.35, 189.9 , 191.45, 193.  , 194.55, 196.1 , 197.65,
            199.2 , 200.75, 202.3 , 203.85, 205.4 , 206.95, 208.5 , 210.05,
            211.6 , 213.15, 214.7 , 216.25, 217.8 , 219.35, 220.9 , 222.45,
            224.  , 225.55, 227.1 , 228.65, 230.2 , 231.75, 233.3 , 234.85,
            236.4 , 237.95, 239.5 , 241.05, 242.6 , 244.15, 245.7 , 247.25,
            248.8 , 250.35, 251.9 , 253.45, 255.  ]),
     <BarContainer object of 100 artists>)
:::

::: {.output .display_data}
![](ecbaa5ea2ea98927c59f8773aad1921fb69f52e7.png)
:::
:::

::: {.cell .code execution_count="31" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":218}" id="74Wnev_99zeW" outputId="b1cf464d-8e06-45fa-fc34-826e4ba4c00c"}
``` python
#Image MAsking with largest Contour
def create_largest_contour_mask(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Error: No contours found.")
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    return masked_img, mask
a = create_largest_contour_mask(img_path1)
b = create_largest_contour_mask(img_path2)

# Display the masked images
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(a[0], cv2.COLOR_BGR2RGB))
plt.title('Masked Image 1')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(b[0], cv2.COLOR_BGR2RGB))
plt.title('Masked Image 2')
plt.axis('off')

plt.show()
```

::: {.output .display_data}
![](eebe0a26b6952b393be53d3b8cf272a9b923ea5d.png)
:::
:::

::: {.cell .code execution_count="46" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":755}" id="MPFfovVRA9g5" outputId="f111e942-2341-4909-a399-27f1282413b9"}
``` python
# Stylization and Blending

def stylize_and_blend(original_img, mask):
    # Ensure original_img is BGR for processing
    if len(original_img.shape) == 3 and original_img.shape[2] == 3:
        img_bgr = original_img.copy()
    else:
        img_bgr = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)


    # 1. Subject Stylization (Bilateral Filter)
    subject_smooth = cv2.bilateralFilter(img_bgr, 9, 75, 75)

    # 2. Background Stylization (Desaturation and Blur)
    background = cv2.bitwise_and(img_bgr, img_bgr, mask=cv2.bitwise_not(mask))
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    background_blurred = cv2.GaussianBlur(background_gray, (5, 5), 0)
    background_stylized = cv2.cvtColor(background_blurred, cv2.COLOR_GRAY2BGR) # Convert back to BGR for blending

    # 3. Stylized Edges (Canny Edge Detection)
    edges = cv2.Canny(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), 100, 200)
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) # Convert edges to color

    # 4. Blending
    # Subject is the smoothed image where the mask is white
    subject_part = cv2.bitwise_and(subject_smooth, subject_smooth, mask=mask)
    # Background is the stylized background where the mask is black
    background_part = cv2.bitwise_and(background_stylized, background_stylized, mask=cv2.bitwise_not(mask))

    # Combine subject and background
    blended_img = cv2.add(subject_part, background_part)

    # Overlay edges - use the mask to only add edges where the subject is
    edge_overlay = cv2.bitwise_and(edges_color, edges_color, mask=mask)
    final_result = cv2.addWeighted(blended_img, 1, edge_overlay, 0.5, 0) # Adjust weight for edge visibility

    return final_result

# Process and display both images
img_paths = [img_path1, img_path2]
masks = [a[1], b[1]] # Assuming 'a' and 'b' contain (masked_img, mask) from the previous step

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for i in range(2):
    original_img = cv2.imread(img_paths[i])
    if original_img is None:
        print(f"Error: Could not load original image {img_paths[i]}.")
        continue

    mask = masks[i]
    stylized_img = stylize_and_blend(original_img, mask)

    axs[i, 0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    axs[i, 0].set_title(f'Original Image {i+1}')
    axs[i, 0].axis('off')

    axs[i, 1].imshow(cv2.cvtColor(stylized_img, cv2.COLOR_BGR2RGB))
    axs[i, 1].set_title(f'Stylized Image {i+1}')
    axs[i, 1].axis('off')

plt.tight_layout()
plt.show()

# Save the stylized result for the first image as required
cv2.imwrite('stylized_result.jpg', cv2.cvtColor(stylized_img1, cv2.COLOR_BGR2RGB))
print("Stylized image saved as stylized_result.jpg")
```

::: {.output .display_data}
![](b1d3800b0c1de133d0c1ec659b832ced2f9d0beb.png)
:::

::: {.output .stream .stdout}
    Stylized image saved as stylized_result.jpg
:::
:::

::: {.cell .code execution_count="43" id="H0YTAyeQGg7V"}
``` python
```
:::

::: {.cell .code id="DtpHSgOkGLde"}
``` python
```
:::
