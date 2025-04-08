print('Setting Up')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

from utils import *
from sklearn.model_selection import train_test_split

#step 1&2: importing data and  trim data
path = 'data_test'
data = importDataInfo(path)

#step 3: Visualization and distribution of data
data = balanceData(data, display = False)

#step 4: Processing
imagesPath, steerings = loadData(path,data)
print(imagesPath[0], steerings[0])

#step 5: splitting data
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steering, test_size=0.2, random_state=5)

print('Total Training images',len(xTrain))
print('Total Validation images',len(x))

#step 6: images augmentation
     ##    in utls.py
#step 7: pre-processing of img
      ## in utils.py

#step 8: creating model proposed by NVIDIA
model = createModel()
model.summary()

#step 9: training Model
history = model.fit(
    batchGen(xTrain, yTrain, 100, 1),steps_per_epoch=300,epochs=10,
    validation_data = batchGen(xVal, yVal, 100, 0), validation_steps=200
)

#step 10: save the model
model.save('model.h5')
print('Model saved')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.ylabel([0,1])
plt.title('Loss')
plt.xlabe('Epoch')
plt.show()