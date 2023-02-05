import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import glob

chess_images = sorted(glob.glob('chess_images/chess_img?.jpg')) # spremanje putanje svih slika unutar jednog polja

chess_image = cv.imread(chess_images[4]) # odabir slike
gray_chess_image = cv.cvtColor(chess_image, cv.COLOR_BGR2GRAY) # konverzija slike u grayscale

retval, detected_corners = cv.findChessboardCorners(image = gray_chess_image, patternSize = (9,6)) # pronalazak kutova među bijelim i crnim poljima šahovske ploče, dimenzija je 9x6
image_with_corners = cv.drawChessboardCorners(chess_image, (9,6), detected_corners, retval) # slika s označenim kutovima

plt.figure(figsize = (12, 8))
plt.subplot(1, 2, 1)
plt.imshow(gray_chess_image, cmap='gray') # prikaži grayscale sliku
plt.title('Chess Image')
plt.subplot(1, 2, 2)
plt.imshow(image_with_corners) # prikaži sliku s označenim kutovima
plt.title('Chess Image With Drawn Corners');
plt.show();

# priprema objektnih točaka
object_grid = np.zeros((9*6, 3), np.float32) 
object_grid[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

points_3D = [] # 3D točke
points_2D = [] # 2D točke

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001) # završetak kriterija

# izdvajanje slika iz mape chess_images
for image in chess_images:
    chess_image = cv.imread(image)
    gray_chess_image = cv.cvtColor(chess_image, cv.COLOR_BGR2GRAY)
    retval, detected_corners = cv.findChessboardCorners(gray_chess_image, (9, 6))

    # ako je odgovarajući broj kutova pronađen, precizirati će se koordinate piksela
    if retval:
        points_3D.append(object_grid)        
        detected_corners_2 = cv.cornerSubPix(gray_chess_image, detected_corners, (11, 11), (-1, -1), criteria)
        points_2D.append(detected_corners_2)

retval, matrix, distortion, rotation_vectors, translation_vectors = cv.calibrateCamera(points_3D, points_2D, gray_chess_image.shape[::-1], None, None) # kalibracija kamere

print("\nCamera calibrated:\n")
print(retval)    
print()
print("Camera matrix:\n")
print(matrix)    
print()
print("Distortion coefficients:\n")
print(distortion)
print()
print("Rotation vectors:\n")
print(rotation_vectors)
print()
print("Translation vectors:\n")
print(translation_vectors)
print()

chess_image = cv.imread('chess_images/chess_img5.jpg')
h, w = chess_image.shape[:2]
new_matrix, ROI_image = cv.getOptimalNewCameraMatrix(matrix, distortion, (w, h), 1, (w, h)) # funkcija vraća novu matricu kamere

corrected_chess_image = cv.undistort(chess_image, matrix, distortion, None, new_matrix) # transformacija slike kako bi se kompenziralo iskrivljenje

# rezanje slike
x, y, w, h = ROI_image
corrected_chess_image = corrected_chess_image[y:y+h, x:x+w]

plt.figure(figsize = (12, 8))
plt.subplot(1, 2, 1)
plt.imshow(chess_image) # prikaži originalnu sliku
plt.title('Original Chess Image')
plt.subplot(1, 2, 2)
plt.imshow(corrected_chess_image) # prikaži ispravljenu sliku
plt.title('Corrected Chess Image');
plt.show();

#izračun re-projekcijske pogreške
avg_error = 0
for i in range(len(points_3D)):
    points_2D_transformed, _ = cv.projectPoints(points_3D[i], rotation_vectors[i], translation_vectors[i], matrix, distortion) # transformacija 3D točke u točku slike
    error = cv.norm(points_2D[i], points_2D_transformed, cv.NORM_L2) / len(points_2D_transformed)
    avg_error += error

print("Re-projection Error: {}\n" .format(avg_error / len(points_3D)))