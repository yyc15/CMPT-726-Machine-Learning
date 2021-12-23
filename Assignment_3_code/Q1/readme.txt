Student ID: 301436160
Student Name: Yin Yu Kevani Chow

CMPT 726 Assignment 3

For Fine-Tuning a Pre-Trained Network:

Below the commment "# Do the testing" is the part for validation of the model.
The model is save by the following code:
	
	best_model_state = model.state_dict()
	torch.save(best_model_state, PATH)
	
The model result is returned with the following code:
	print('**********Model Result**********')	
	min_training_loss = min(total_training_loss)
	print("Minimum Training Loss:", min_training_loss)
	print("Training Accuracy:", training_accuracy)
	min_testing_loss = min(total_testing_loss)
	print('Minimum loss in epoch:', min_epoch + 1)
	print("Minimum Testing Loss:", min_testing_loss)
	print("Testing Accuracy:", testing_accuracy)
	

For applying L2 regularization to the coefficients, weight_decay parameter is applied to the optimizer as below:
	optimizer = optim.SGD(list(model.fc.parameters()), lr=0.001, momentum=0.9, weight_decay=0.00001)


	
After the comment "# Prediction" one of the image in the testest is picked for proving the model is working:
The prediction result is returned with the following code:
	print('**********Prediction Result of 1 image**********')
    	print("Expected result: " + classes[test_img[1]])
    	print("Prediction from the model: " + classes[prediction])
	
The original image is plotted and shown with the following code:
	plt.imshow(np.transpose(test_img[0].numpy(), (1, 2, 0)))
    	plt.show()

Here is the commond to run the code:
	python3 cifar_finetune.py