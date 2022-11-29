
import cv2 
import numpy as np

def get_nodes(x, y):
		
	co_ordinates = {'A1' : [100,100],'A2': [100,200], 'A3':[100,300], 'A4':[100,400], 'A5':[100,500], 'A6':[100,600], 'A7':[100,700], 
					'B1' : [200,100],'B2': [200,200], 'B3':[200,300], 'B4':[200,400], 'B5':[200,500], 'B6':[200,600], 'B7':[200,700],
					'c1' : [300,100],'C2': [300,200], 'C3':[300,300], 'C4':[300,400], 'C5':[300,500], 'C6':[300,600], 'C7':[300,700],
					'D1' : [400,100],'D2': [400,200], 'D3':[400,300], 'D4':[400,400], 'D5':[400,500], 'D6':[400,600], 'D7':[400,700],
					'E1' : [500,100],'E2': [500,200], 'E3':[500,300], 'E4':[500,400], 'E5':[500,500], 'E6':[500,600], 'E7':[500,700],
					'F1' : [600,100],'F2': [600,200], 'F3':[600,300], 'F4':[600,400], 'F5':[600,500], 'F6':[600,600], 'F7':[600,700],
					'G1' : [700,100],'G2': [700,200], 'G3':[700,300], 'G4':[700,400], 'G5':[700,500], 'G6':[700,600], 'G7':[700,700]}


	val = [x,y]
	for key, value in co_ordinates.items():
		if val == value:
			return key


def detect_traffic_signals(maze_image):

	"""
	Purpose:
	---
	This function takes the image as an argument and returns a list of
	nodes in which traffic signals are present in the image

	Input Arguments:
	---
	`maze_image` :	[ numpy array ]
			numpy array of image returned by cv2 library
	Returns:
	---
	`traffic_signals` : [ list ]
			list containing nodes in which traffic signals are present
	"""    
	traffic_signals = []
	hsv = cv2.cvtColor(maze_image, cv2.COLOR_BGR2HSV)

	lower_range = np.array([0,50,50], dtype = "uint8")
	upper_range = np.array([10,255,255], dtype = "uint8")
	vmask = cv2.inRange(hsv, lower_range, upper_range)

	detected_output = cv2.bitwise_and(maze_image, maze_image, mask = vmask)

	detected_output_gray = cv2.cvtColor(detected_output, cv2.COLOR_BGR2GRAY)

	contours, hierarchies = cv2.findContours(detected_output_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

	mu = [None]*len(contours)    #list to save moments of the contours
	for points in range(len(contours)):
		mu[points] = cv2.moments(contours[points])
		x = int(mu[points]["m10"] / mu[points]["m00"])
		y = int(mu[points]["m01"] / mu[points]["m00"])
		val =[x,y]
		traffic_signals.append(get_nodes(x, y))

	traffic_signals.sort()		
	return traffic_signals
	

def detect_horizontal_roads_under_construction(maze_image):
	
	"""
	Purpose:
	---
	This function takes the image as an argument and returns a list
	containing the missing horizontal links

	Input Arguments:
	---
	`maze_image` :	[ numpy array ]
			numpy array of image returned by cv2 library
	Returns:
	---
	`horizontal_roads_under_construction` : [ list ]
			list containing missing horizontal links
	"""    
	horizontal_roads_under_construction = []
	for i in range(7):
		for j in range(6):
			color = maze_image[i*100 +100, j*100 + 150 ] #make sure to give x value as second parameter, y value as first parameter.
			if((color == [255, 255, 255]).all()):
				horizontal_roads_under_construction.append(f'{get_nodes((j*100)+100, (i*100)+100)}-{get_nodes((j*100)+200, (i*100) +100)}')
	
	horizontal_roads_under_construction.sort()
	
	return horizontal_roads_under_construction	

def detect_vertical_roads_under_construction(maze_image):

	"""
	Purpose:
	---
	This function takes the image as an argument and returns a list
	containing the missing vertical links

	Input Arguments:
	---
	`maze_image` :	[ numpy array ]
			numpy array of image returned by cv2 library
	Returns:
	---
	`vertical_roads_under_construction` : [ list ]
			list containing missing vertical links
	"""    
	vertical_roads_under_construction = []

	for i in range(6):
		for j in range(7):
			color = maze_image[i*100 +150, j*100 + 100 ] #make sure to give x value as second parameter, y value as first parameter.
			if((color == [255, 255, 255]).all()):
				vertical_roads_under_construction.append(f'{get_nodes((j*100)+100, (i*100)+100)}-{get_nodes((j*100)+100, (i*100)+200)}')
	
	vertical_roads_under_construction.sort()

	return vertical_roads_under_construction


def detect_medicine_packages(maze_image):

	"""
	Purpose:
	---
	This function takes the image as an argument and returns a nested list of
	details of the medicine packages placed in different shops

	** Please note that the shop packages should be sorted in the ASCENDING order of shop numbers 
	   as well as in the alphabetical order of colors.
	   For example, the list should first have the packages of shop_1 listed. 
	   For the shop_1 packages, the packages should be sorted in the alphabetical order of color ie Green, Orange, Pink and Skyblue.

	Input Arguments:
	---
	`maze_image` :	[ numpy array ]
			numpy array of image returned by cv2 library
	Returns:
	---
	`medicine_packages` : [ list ]
			nested list containing details of the medicine packages present.
			Each element of this list will contain 
			- Shop number as Shop_n
			- Color of the package as a string
			- Shape of the package as a string
			- Centroid co-ordinates of the package
	"""    
	medicine_packages = []

	gray = cv2.cvtColor(maze_image, cv2.COLOR_BGR2GRAY)
	for cell in range(6):
		cell_packages = []
		cropped = gray[110:190, (cell*100)+110:(cell*100)+190]    #changoing second parameter which is x cooridnates to acess different shops in the images. 1st coordinate is y coordinates.

		_, threshold = cv2.threshold(cropped, 200, 220, cv2.THRESH_BINARY)

		contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

		
		for cnt in contours:
			if(cv2.contourArea(cnt) < 5000):
				cell_package = []
				coordinate = []
				cell_package.append(f'Shop_{cell+1}')

				mu = cv2.moments(cnt)
				x = int(mu["m10"] / mu["m00"]) + 110 + (cell * 100)
				y = int(mu["m01"] / mu["m00"]) + 110
				coordinate = [x, y]

				color = maze_image[y, x]
				if((color == [180,0,255]).all()): cell_package.append("Pink")
				if((color == [0,127,255]).all()): cell_package.append("Orange")
				elif((color == [0,255,0]).all()): cell_package.append("Green")
				elif((color == [255,255,0]).all()): cell_package.append("Skyblue")

				peri = cv2.arcLength(cnt, True) 
				approx = cv2.approxPolyDP(cnt , 0.02*peri, True)
				if(len(approx) == 3): cell_package.append("Triangle")
				elif(len(approx) == 4): cell_package.append("Square")
				elif(len(approx) >= 5): cell_package.append("Circle")

				cell_package.append(coordinate)
				cell_packages.append(cell_package)
		
		cell_packages.sort(key = lambda cell_packages : cell_packages[1])
		medicine_packages.extend(cell_packages)

	return medicine_packages

def detect_arena_parameters(maze_image):

	"""
	Purpose:
	---
	This function takes the image as an argument and returns a dictionary
	containing the details of the different arena parameters in that image

	The arena parameters are of four categories:
	i) traffic_signals : list of nodes having a traffic signal
	ii) horizontal_roads_under_construction : list of missing horizontal links
	iii) vertical_roads_under_construction : list of missing vertical links
	iv) medicine_packages : list containing details of medicine packages

	These four categories constitute the four keys of the dictionary

	Input Arguments:
	---
	`maze_image` :	[ numpy array ]
			numpy array of image returned by cv2 library
	Returns:
	---
	`arena_parameters` : { dictionary }
			dictionary containing details of the arena parameters
	"""    
	arena_parameters = {}
	arena_parameters['traffic_signals'] = detect_traffic_signals(maze_image)
	arena_parameters['horizontal_roads_under_construction'] = detect_horizontal_roads_under_construction(maze_image)
	arena_parameters['vertical_roads_under_construction'] = detect_vertical_roads_under_construction(maze_image)
	arena_parameters['medicine_packages_present'] = detect_medicine_packages(maze_image)

	
	return arena_parameters

if __name__ == "__main__":

    # path directory of images in test_images folder
	img_dir_path = "test_images/"

    # path to 'maze_0.png' image file
	file_num = 0
	img_file_path = img_dir_path + 'maze_' + str(file_num) + '.png'
	
	# read image using opencv
	maze_image = cv2.imread(img_file_path)
	
	print('\n============================================')
	print('\nFor maze_' + str(file_num) + '.png')

	# detect and print the arena parameters from the image
	arena_parameters = detect_arena_parameters(maze_image)

	print("Arena Prameters: " , arena_parameters)

	# display the maze image
	cv2.imshow("image", maze_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	choice = input('\nDo you want to run your script on all test images ? => "yes" or "no": ')
	
	if choice == 'yes':

		for file_num in range(1, 15):
			
			# path to maze image file
			img_file_path = img_dir_path + 'maze_' + str(file_num) + '.png'
			
			# read image using opencv
			maze_image = cv2.imread(img_file_path)
	
			print('\n============================================')
			print('\nFor maze_' + str(file_num) + '.png')
			
			# detect and print the arena parameters from the image
			arena_parameters = detect_arena_parameters(maze_image)

			print("Arena Parameter: ", arena_parameters)
				
			# display the test image
			cv2.imshow("image", maze_image)
			cv2.waitKey(2000)
			cv2.destroyAllWindows()