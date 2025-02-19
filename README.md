# 🖼️ Image Segmentation by Region Growing

This project implements **image segmentation using the Region Growing algorithm**. It segments an image based on intensity values in **Lab** and **Luv** color spaces.

## 📌 Project Overview

The algorithm **identifies homogeneous regions** in an image by growing connected pixel groups with similar intensity values. The goal is to **highlight and separate meaningful objects in an image**.

### 🔹 Features Implemented:
- Load and **preprocess images** (Gaussian filtering to reduce noise).
- Convert images to **Lab** and **Luv** color spaces.
- Segment images using **Region Growing** in different channels (**a, b, u, v**).
- Display segmented images using **color mapping**.

## 🏗️ System Architecture

### 🛠️ Modules Implemented:
1️⃣ **Image Loading:** Loads images and applies **Gaussian filter**.  
2️⃣ **Color Space Conversion:** Converts images to **Lab** and **Luv** models.  
3️⃣ **Region Growing Algorithm:**  
   - Initializes region labels for **each channel (a, b, u, v)**.
   - Uses **queue-based expansion** for region growth.
   - Adds neighboring pixels based on **intensity difference threshold**.
   - Updates **region mean intensity dynamically**.
4️⃣ **Result Visualization:** 
   - Maps segmented regions to colors using **COLORMAP_RAINBOW**.
   - Displays processed images.

## 🛠️ Technologies Used

| Component          | Technology |
|--------------------|------------|
| Programming Language | C++ |
| Image Processing   | OpenCV |
| UI & Visualization | OpenCV GUI |

## ▶️ Running the Project

1️⃣ **Clone the repository:**
   ```sh
   git clone https://github.com/raresm2003/Image-Segmentation-by-Region-Growing.git
   cd image-segmentation-by-region-growing
   ```

2️⃣ **Run the application:**

## 🚀 Future Improvements

- Support for **additional color spaces (HSV, YCbCr)**.
- Implement **adaptive thresholding** using **Machine Learning**.
- Add **multi-scale segmentation** for improved detail.

## 👨‍💻 Contributors

**[Miclea Rareș](https://github.com/raresm2003)**  

---

### ⭐ If you like this project, give it a star! ⭐  
