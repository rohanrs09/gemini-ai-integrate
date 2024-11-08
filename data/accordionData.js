const accordionData = [
  {
    title: "Assignment_2_MNIST - FNN",
    content: `# Import TensorFlow and relevant libraries
  import tensorflow as tf
  from tensorflow import keras
  from tensorflow.keras.preprocessing.image import ImageDataGenerator
  import matplotlib.pyplot as plt
  import numpy as np
  import seaborn as sns
  
  # Define the paths to the training and testing data directories
  train_data_dir = 'mnist-jpg/train'
  test_data_dir = 'mnist-jpg/test'
  
  # Set up an ImageDataGenerator to rescale pixel values to [0, 1]
  image_data_generator = ImageDataGenerator(rescale=1.0/255)
  
  # Define batch sizes
  train_batch_size = 10000
  test_batch_size = 5000
  
  # Create data generators for training and testing
  train_generator = image_data_generator.flow_from_directory(
      train_data_dir,
      target_size=(28, 28),  # Resize images to 28x28 pixels
      batch_size=train_batch_size,  # Number of images per training batch
      class_mode='categorical',  # One-hot encoded labels
      color_mode='grayscale',  # Convert images to grayscale
      shuffle=True,  # Shuffle the order of images during training
  )
  
  test_generator = image_data_generator.flow_from_directory(
      test_data_dir,
      target_size=(28, 28),  # Resize images to 28x28 pixels
      batch_size=test_batch_size,  # Number of images per testing batch
      class_mode='categorical',  # One-hot encoded labels
      color_mode='grayscale',  # Convert images to grayscale
      shuffle=True,  # Shuffle the order of images during testing
  )
  
  x_train, y_train = train_generator[0]
  x_test, y_test = test_generator[0]
  
  print(f"Shape of X_train {x_train.shape}")
  print(f"Shape of y_train {y_train.shape}")
  print(f"Shape of x_test  {x_test.shape}")
  print(f"Shape of y_test  {y_test.shape}")
  
  model = keras.Sequential([
      keras.layers.Flatten(input_shape=(28,28)),
      keras.layers.Dense(50,activation='relu',name='L1'),
      keras.layers.Dense(50,activation='relu',name='L2'),
      keras.layers.Dense(10,activation='softmax',name='L3')
  ])
  
  model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
  history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), batch_size=10, shuffle=True)
  
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title("Model Accuracy")
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['Train', "Validation"], loc='upper left')
  
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model Loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['Train', 'Validation'], loc='upper left')
  
  test_loss, test_acc = model.evaluate(x_test, y_test)
  print("Loss: ", test_loss)
  print("Accuracy: ", test_acc)
  
  predicted_value = model.predict(x_test)
  plt.imshow(x_test[15])
  plt.show()
  print(np.argmax(predicted_value[15], axis=0))`,
    files: [
      {
        url: "https://drive.google.com/file/d/1zxQLn01INv_nd71b5cPndsuXYh1Zmkvq/view?usp=drive_link", // Relative path to the public folder
        name: "DL_assign_2_cifar10_img_offline.ipynb", // File name shown in the UI
      },
    ],
  },
  {
    title: "Accordion 2",
    content: "This is the content for Accordion 2.",
    files: [
      {
        url: "https://drive.google.com/file/d/1zxQLn01INv_nd71b5cPndsuXYh1Zmkvq/view?usp=drive_link", // Relative path to the public folder
        name: "DL_assign_2_cifar10_img_offline.ipynb", // File name shown in the UI
      },
    ],
  },
  {
    title: "Accordion 3",
    content: "This is the content for Accordion 3.",
    files: [
      {
        url: "https://drive.google.com/file/d/1zxQLn01INv_nd71b5cPndsuXYh1Zmkvq/view?usp=drive_link", // Relative path to the public folder
        name: "DL_assign_2_cifar10_img_offline.ipynb", // File name shown in the UI
      },
    ],
  },
  {
    title: "Accordion 4",
    content: "This is the content for Accordion 4.",
    files: [
      {
        url: "https://drive.google.com/file/d/1zxQLn01INv_nd71b5cPndsuXYh1Zmkvq/view?usp=drive_link", // Relative path to the public folder
        name: "DL_assign_2_cifar10_img_offline.ipynb", // File name shown in the UI
      },
    ],
  },
  {
    title: "Accordion 5",
    content: "This is the content for Accordion 5.",
    files: [
      {
        url: "https://drive.google.com/file/d/1zxQLn01INv_nd71b5cPndsuXYh1Zmkvq/view?usp=drive_link", // Relative path to the public folder
        name: "DL_assign_2_cifar10_img_offline.ipynb", // File name shown in the UI
      },
    ],
  },
  {
    title: "Accordion 6",
    content: "This is the content for Accordion 6.",
    files: [
      {
        url: "https://drive.google.com/file/d/1zxQLn01INv_nd71b5cPndsuXYh1Zmkvq/view?usp=drive_link", // Relative path to the public folder
        name: "DL_assign_2_cifar10_img_offline.ipynb", // File name shown in the UI
      },
    ],
  },
];

export default accordionData;
