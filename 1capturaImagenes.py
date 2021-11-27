# importamos las librerias de reconocimiento de rostros
import cv2
import os
import imutils
#pip install imutils
#path_python.exe -m pip install --upgrade pip
contador=1
persona ="s"
while persona!= "n":
	#Consultamos el nombre de la persona que vamos a caprturar
	personName = input("Ingrese el nombre de la persona "+str(contador)+" que vamos a capturar su rostros: ")

	#Carpeta donde se va almacenar las imagenes
	dataPath = 'imagenes'

	#Concatenamos la ruta del rostro a capturar
	personPath = dataPath + '/' + personName

	# Número de la camara 0, 1 son los índices de la cámara por defecto
	numCamara = 0

	#Validamos si la carpeta existe o si no la creamos
	if not os.path.exists(personPath):
		print('Carpeta creada: ',personPath)
		os.makedirs(personPath)
	#Abrimos la camara para realizar las capturas
	try:
		cap = cv2.VideoCapture(numCamara,cv2.CAP_DSHOW)  
			#cap = cv2.VideoCapture('Video.mp4')
	except Exception as error:
		print('Error con algo de la cámara ' + str(error))

	#Inicializamos el claisifcador con el modelo haar
	faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
	count = 0

	while True:
		ret, frame = cap.read()
		if ret == False: break
		frame =  imutils.resize(frame, width=640)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		auxFrame = frame.copy()

		faces = faceClassif.detectMultiScale(gray,1.3,5)

		for (x,y,w,h) in faces:
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
			rostro = auxFrame[y:y+h,x:x+w]
			rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
			cv2.imwrite(personPath + '/rotro_{}.jpg'.format(count),rostro)
			count = count + 1
		cv2.imshow('frame',frame)

		k =  cv2.waitKey(1)
		if k == 27 or count >= 300:
			break
	cap.release()
	if contador >= 1000:
		persona = "n"
	else:
		persona = input("Desea ingresar otra persona (y) o (n): ")

		if persona == "y":
			contador+=1
		if persona == "n":
			print('Ahora vamos al entrenamiento de las imagenes capturadas')
cv2.destroyAllWindows()
