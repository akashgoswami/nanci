/**
 * Advanced Neural Network Implementation for MNIST Digit Recognition
 * We will implement an advanced neural network with MNIST handwritten digits dataset.
 * The network will train on image samples and would guess the digit in the image.
 * The input image is 28x28 pixels and flattened to 784x1 vector.
 * The output is a 10x1 vector with the probability of the digit being one of the 10 digits.

* Architecture:
 * - Input Layer: 784 neurons (28x28 pixel images) normalized to [0,1]
 * - Hidden Layer: 512 neurons with sigmoid activation
 * - Output Layer: 10 neurons (one per digit) with sigmoid activation
 * 
 * Features:
 * - Mini-batch Stochastic Gradient Descent (SGD)
 * - Cross-entropy loss function
 * - Xavier/Glorot weight initialization instead of random initialization
 * - Bias terms for both layers
 * - Learning rate decay
 * - Weight saving and loading capabilities
 * 
 * A python script is used to generate the MNIST dataset and save it to binary files.
 * The MNIST dataset is a collection of 28x28 pixel images of handwritten digits.
 * The dataset is split into training and test sets.
 * The training set is used to train the network and the test set is used to evaluate the network.
  
    Input             Hidden        Output
                                        
    +-----+                                
    |     |                                
    |     |            +----+              
    |     |            |    |              
    |     |            |    |              
    |     |            |    |          +---+
    |     |            |    |          |   | 
    |     |            |    |          |   |
    | 784 |     x      | 512|    x     | 10|
    |     |            |    |          |   |
    |     |            |    |          +---+
    |     |            |    |              
    |     |            |    |              
    |     |            |    |              
    |     |            +----+              
    |     |                                
    |     |                                
    +-----+                                
*/
 

// 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// Network architecture parameters
#define NUM_TRAINING_IMAGES 60000
#define NUM_TEST_IMAGES 10000
#define INPUT_SIZE 784   // 28x28 pixels
#define HIDDEN_SIZE 512
#define OUTPUT_SIZE 10   // 10 digits (0-9)
#define BATCH_SIZE 32

// File paths for saving/loading model parameters
#define HIDDEN_WEIGHTS_FILE "hidden_weights.bin"
#define HIDDEN_BIAS_FILE "hidden_bias.bin"
#define OUTPUT_WEIGHTS_FILE "output_weights.bin"
#define OUTPUT_BIAS_FILE "output_bias.bin"

// Training data
// Large array containing training images of 784 features (28x28 pixels)
float training_images[NUM_TRAINING_IMAGES][INPUT_SIZE];

// Large array containing training labels of 10 digits (0-9)
// Each label is an integer between 0 and 9
int training_labels[NUM_TRAINING_IMAGES];

// Large array containing test images of 784 features (28x28 pixels)
float test_images[NUM_TEST_IMAGES][INPUT_SIZE];

// Large array containing test labels of 10 digits (0-9)
// Each label is an integer between 0 and 9
int test_labels[NUM_TEST_IMAGES];

// Network parameters (weights and biases)

// Weights between input and hidden layer. The goal of the training is to find the weights that will give the best prediction.
// We also save and restore the weights from files so with each training we could improve the model.
float hidden_weights[HIDDEN_SIZE][INPUT_SIZE];  

// Bias terms for hidden layer. The goal of the training is to find the bias that will give the best prediction.
float hidden_bias[HIDDEN_SIZE];                

// Weights between hidden and output layer. The goal of the training is to find the weights that will give the best prediction.
// We also save and restore the weights from files so with each training we could improve the model.
float output_weights[OUTPUT_SIZE][HIDDEN_SIZE]; 

// Bias terms for output layer. The goal of the training is to find the bias that will give the best prediction.
// We also save and restore the bias from files so with each training we could improve the model.
float output_bias[OUTPUT_SIZE];                

// Gradient storage for backpropagation
// The gradients are calculated during the backward pass and used to update the weights and biases.
// Each gradient corresponds to a weight or bias. 

// Important to note that the gradients are calculated for each batch and then accumulated.
// This is done to improve the stability of the training process and to prevent overfitting.
// If we calculate gradient (aka backpropagation) for each image, the model will be too slow and will try to overfit.
// Instead we calculate the gradient for a batch of images and then update the weights and biases.
// This way the model will be more stable and will not overfit.

float hidden_weights_grad[HIDDEN_SIZE][INPUT_SIZE];
float hidden_bias_grad[HIDDEN_SIZE];
float output_weights_grad[OUTPUT_SIZE][HIDDEN_SIZE];
float output_bias_grad[OUTPUT_SIZE];

// Mini-batch gradient accumulation
// This is the same as the gradients but accumulated for each batch of 32 forward passes.
float hidden_weights_batch_grad[HIDDEN_SIZE][INPUT_SIZE];
float hidden_bias_batch_grad[HIDDEN_SIZE];
float output_weights_batch_grad[OUTPUT_SIZE][HIDDEN_SIZE];
float output_bias_batch_grad[OUTPUT_SIZE];



/**
 * Sigmoid activation function with numerical stability bounds
 * Bounds prevent overflow/underflow in exp() calculation
 */
float sigmoid(float x) {
    if (x < -45.0) return 0;      // Prevent underflow
    if (x > 45.0) return 1;       // Prevent overflow
    return 1.0 / (1.0 + exp(-x));
}

/**
 * Initialize network weights using Xavier/Glorot initialization. We could also use random initialization, 
 * but Xavier/Glorot initialization helps prevent vanishing/exploding gradients by keeping
 * the variance of activations roughly constant across layers
 */
void init_weights() {
    // Calculate scaling factors for each layer
    float hidden_scale = sqrt(6.0 / (INPUT_SIZE + HIDDEN_SIZE));
    float output_scale = sqrt(6.0 / (HIDDEN_SIZE + OUTPUT_SIZE));
    
    // Initialize hidden layer weights and bias
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            // Random values between -scale and +scale
            hidden_weights[i][j] = ((float)rand() / RAND_MAX * 2 - 1) * hidden_scale;
        }
        hidden_bias[i] = 0;  // Initialize bias to zero
        memset(hidden_weights_batch_grad[i], 0, INPUT_SIZE * sizeof(float));
    }
    memset(hidden_bias_batch_grad, 0, HIDDEN_SIZE * sizeof(float));
    
    // Initialize output layer weights and bias
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            output_weights[i][j] = ((float)rand() / RAND_MAX * 2 - 1) * output_scale;
        }
        output_bias[i] = 0;  // Initialize bias to zero
        memset(output_weights_batch_grad[i], 0, HIDDEN_SIZE * sizeof(float));
    }
    memset(output_bias_batch_grad, 0, OUTPUT_SIZE * sizeof(float));
}

/**
 * Reset gradient accumulators for new mini-batch. Resetting is required to prevent the gradients from accumulating.
 * In each batch, gradient changes as weight changed due to previous batch.
 */
void clear_batch_gradients() {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        memset(hidden_weights_batch_grad[i], 0, INPUT_SIZE * sizeof(float));
    }
    memset(hidden_bias_batch_grad, 0, HIDDEN_SIZE * sizeof(float));
    
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        memset(output_weights_batch_grad[i], 0, HIDDEN_SIZE * sizeof(float));
    }
    memset(output_bias_batch_grad, 0, OUTPUT_SIZE * sizeof(float));
}

/**
 * Forward pass through the network. This is easiest part of the network. We just multiply the input by the weights and add the bias.
 * Computes activations for hidden and output layers
 */
void forward_pass(const float input[INPUT_SIZE], float hidden[HIDDEN_SIZE], float output[OUTPUT_SIZE]) {
    // Compute hidden layer activations
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        float sum = hidden_bias[i];  // Start with bias term
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += hidden_weights[i][j] * input[j];
        }
        hidden[i] = sigmoid(sum);
    }
    
    // Compute output layer activations
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        float sum = output_bias[i];  // Start with bias term
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += output_weights[i][j] * hidden[j];
        }
        output[i] = sigmoid(sum);
    }
}

/**
 * Backward pass through the network. This is the most critical part of the network. 
 * We need to calculate the gradients for the weights and biases.
 * This is done by calculating the error (difference between the predicted output and the target output) and then propagating the error back through the network.

 * Surprisingly, calculating the gradient for a neuron is as simple as multiplying the error by the input.
 * for a sigmoid activation function, the gradient is the derivative of the sigmoid function, which is sigmoid(x) * (1 - sigmoid(x)).
 * so a neuron's gradient is the error multiplied by the input and the derivative of the sigmoid function.

 * The distributed error is then used to calculate gradients which are used to update the weights and biases.
 */
void backward_pass(const float input[INPUT_SIZE], const float hidden[HIDDEN_SIZE], 
                  const float output[OUTPUT_SIZE], const float target[OUTPUT_SIZE]) {
    
    // YOU MAY BE SURPRISED about why do we need input layer values here. 

    // The first step is to calculate the error (difference between the predicted output and the target output).
    // This is simple, we just subtract the target from the output for each output neuron.

    // Compute output layer error (cross-entropy derivative with respect to output)

    float output_error[OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_error[i] = output[i] - target[i];  
    }
    
    // Now we have a list of errors for each output neuron.

    // This step is to distribute the error from the output layer to the hidden layer.
    // This is done by multiplying the error by the weights between the hidden and output layer.
    // This way we get the error for each hidden neuron.

    // Compute hidden layer error (backpropagated from output layer)
    float hidden_error[HIDDEN_SIZE];
    for (int i = 0; i < HIDDEN_SIZE; i++) {  // for every neuron in the hidden layer
        hidden_error[i] = 0;
        // Sum up error contributions from each output neuron multiplied by the weight between the hidden and output neuron.
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            hidden_error[i] += output_error[j] * output_weights[j][i];
        }
        // Multiply by derivative of sigmoid activation function.
        hidden_error[i] *= hidden[i] * (1 - hidden[i]);
    }
    //  What do we have here? We have a list of distributed errors for each hidden neuron.
    // We aren't yet into gradient calculation, we just distributed the error from the output layer to the hidden layer.

    // Now we are ready to calculate the gradients.
    // Compute gradients for output layer weights and bias.

    // for output weight gradient we need to multiply the output_error by the hidden layer inputs.
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            output_weights_grad[i][j] = output_error[i] * hidden[j];
        }
        // for output bias gradient we need to multiply the output_error by 1.
        output_bias_grad[i] = output_error[i];
    }
    
    // Now we are ready to calculate the gradients for the hidden layer.
    // Compute gradients for hidden layer weights and bias.
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            hidden_weights_grad[i][j] = hidden_error[i] * input[j];
        }
        hidden_bias_grad[i] = hidden_error[i];
    }
    
    // Accumulate calculated gradients into the mini-batch accumulator
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            output_weights_batch_grad[i][j] += output_weights_grad[i][j];
        }
        output_bias_batch_grad[i] += output_bias_grad[i];
    }
    
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            hidden_weights_batch_grad[i][j] += hidden_weights_grad[i][j];
        }
        hidden_bias_batch_grad[i] += hidden_bias_grad[i];
    }
}

/**
 * Update network parameters using accumulated gradients
 */
void update_weights(float learning_rate, int batch_size) {
    float scale = learning_rate / batch_size;
    
    // Update output layer parameters
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            output_weights[i][j] -= scale * output_weights_batch_grad[i][j];
        }
        output_bias[i] -= scale * output_bias_batch_grad[i];
    }
    
    // Update hidden layer parameters
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            hidden_weights[i][j] -= scale * hidden_weights_batch_grad[i][j];
        }
        hidden_bias[i] -= scale * hidden_bias_batch_grad[i];
    }
}

/**
 * Create one-hot encoded target vector from label
 */
void create_target(int label, float target[OUTPUT_SIZE]) {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        target[i] = (i == label) ? 1.0f : 0.0f;
    }
}

/**
 * Find index of maximum value in array
 */
int argmax(const float arr[], int size) {
    int max_idx = 0;
    float max_val = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] > max_val) {
            max_val = arr[i];
            max_idx = i;
        }
    }
    return max_idx;
}

/**
 * Randomly shuffle training data
 */
void shuffle_data() {
    for (int i = NUM_TRAINING_IMAGES - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        // Swap images
        for (int k = 0; k < INPUT_SIZE; k++) {
            float temp = training_images[i][k];
            training_images[i][k] = training_images[j][k];
            training_images[j][k] = temp;
        }
        // Swap labels
        int temp_label = training_labels[i];
        training_labels[i] = training_labels[j];
        training_labels[j] = temp_label;
    }
}

/**
 * Load MNIST image data
 */
const int load_mnist_images(const char *filename, float images[][INPUT_SIZE], int num_images) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error opening image file: %s\n", filename);
        return -1;
    }
    
    for (int i = 0; i < num_images; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;
            if (fread(&pixel, 1, 1, fp) != 1) {
                printf("Error reading image data\n");
                fclose(fp);
                return -1;
            }
            images[i][j] = pixel / 255.0f;  // Normalize to [0,1]
        }
    }
    
    fclose(fp);
    return 0;
}

/**
 * Load MNIST label data. This is specially prepared label with no header.
 * Each label is an integer between 0 and 9.
 */
const int load_mnist_labels(const char *filename, int labels[], int num_labels) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Error opening label file: %s\n", filename);
        return -1;
    }
    
    for (int i = 0; i < num_labels; i++) {
        unsigned char label;
        if (fread(&label, 1, 1, fp) != 1) {
            printf("Error reading label data\n");
            fclose(fp);
            return -1;
        }
        labels[i] = label;
    }
    
    fclose(fp);
    return 0;
}

/**
 * Save network parameters to files
 */
void save_weights() {
    FILE *fp;
    
    // Save hidden weights
    fp = fopen(HIDDEN_WEIGHTS_FILE, "wb");
    fwrite(hidden_weights, sizeof(float), HIDDEN_SIZE * INPUT_SIZE, fp);
    fclose(fp);
    
    // Save hidden bias
    fp = fopen(HIDDEN_BIAS_FILE, "wb");
    fwrite(hidden_bias, sizeof(float), HIDDEN_SIZE, fp);
    fclose(fp);
    
    // Save output weights
    fp = fopen(OUTPUT_WEIGHTS_FILE, "wb");
    fwrite(output_weights, sizeof(float), OUTPUT_SIZE * HIDDEN_SIZE, fp);
    fclose(fp);
    
    // Save output bias
    fp = fopen(OUTPUT_BIAS_FILE, "wb");
    fwrite(output_bias, sizeof(float), OUTPUT_SIZE, fp);
    fclose(fp);
    
    printf("Model parameters saved to files\n");
}

/**
 * Load network parameters from files if found, otherwise initialize new weights
 */
int load_weights() {
    FILE *fp;
    
    // Load hidden weights
    fp = fopen(HIDDEN_WEIGHTS_FILE, "rb");
    if (!fp) return -1;
    fread(hidden_weights, sizeof(float), HIDDEN_SIZE * INPUT_SIZE, fp);
    fclose(fp);
    
    // Load hidden bias
    fp = fopen(HIDDEN_BIAS_FILE, "rb");
    if (!fp) return -1;
    fread(hidden_bias, sizeof(float), HIDDEN_SIZE, fp);
    fclose(fp);
    
    // Load output weights
    fp = fopen(OUTPUT_WEIGHTS_FILE, "rb");
    if (!fp) return -1;
    fread(output_weights, sizeof(float), OUTPUT_SIZE * HIDDEN_SIZE, fp);
    fclose(fp);
    
    // Load output bias
    fp = fopen(OUTPUT_BIAS_FILE, "rb");
    if (!fp) return -1;
    fread(output_bias, sizeof(float), OUTPUT_SIZE, fp);
    fclose(fp);
    
    printf("Model parameters loaded from files\n");
    return 0;
}

/**
 * Evaluate model on test set
 * Returns accuracy as a percentage
 */
float evaluate(const float test_images[][INPUT_SIZE], const int test_labels[], int num_images) {
    int correct = 0;
    float hidden[HIDDEN_SIZE];
    float output[OUTPUT_SIZE];
    
    for (int i = 0; i < num_images; i++) {
        // Forward pass
        forward_pass(test_images[i], hidden, output);
        
        // Check if prediction is correct
        if (argmax(output, OUTPUT_SIZE) == test_labels[i]) {
            correct++;
        }
    }
    
    return 100.0f * correct / num_images;
}

/**
 * Calculate cross-entropy loss for a batch
 * This is a loss function that measures the performance of the network.
 * The loss is the sum of the negative log-likelihood of the predicted output and the target output.
 */
float calculate_loss(const float output[OUTPUT_SIZE], const float target[OUTPUT_SIZE]) {
    float loss = 0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        // Clip output to avoid log(0)
        float clipped_output = fmax(fmin(output[i], 0.9999f), 0.0001f);
        // Calculate the loss for each neuron.
        loss -= target[i] * log(clipped_output) + (1 - target[i]) * log(1 - clipped_output);
    }
    return loss;
}

int main() {
    srand(time(NULL));
    
    printf("Loading MNIST dataset...\n");
    
    // Load training data
    if (load_mnist_images("./mnist/mnist_train_images.bin", training_images, NUM_TRAINING_IMAGES) != 0 ||
        load_mnist_labels("./mnist/mnist_train_labels.bin", training_labels, NUM_TRAINING_IMAGES) != 0) {
        printf("Failed to load training data\n");
        return 1;
    }
    
    // Load test data
    if (load_mnist_images("./mnist/mnist_test_images.bin", test_images, NUM_TEST_IMAGES) != 0 ||
        load_mnist_labels("./mnist/mnist_test_labels.bin", test_labels, NUM_TEST_IMAGES) != 0) {
        printf("Failed to load test data\n");
        return 1;
    }
    
    printf("Dataset loaded successfully\n");
    
    // Try to load pre-trained weights, initialize new ones if loading fails
    if (load_weights() != 0) {
        printf("Initializing new weights...\n");
        init_weights();
    }
    else {
        printf("!!! Continuing with pre-trained weights...\n");
    }
    
    // Training parameters
    float learning_rate = 0.1f;
    const int num_epochs = 10;
    float best_accuracy = 0.0f;
    
    // Training loop
    printf("\nStarting training...\n");
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        printf("\nEpoch %d/%d\n", epoch + 1, num_epochs);
        
        // Shuffle training data
        shuffle_data();
        
        // Training metrics
        int correct_predictions = 0;
        float epoch_loss = 0.0f;
        
        // Mini-batch training
        for (int i = 0; i < NUM_TRAINING_IMAGES; i += BATCH_SIZE) {
            clear_batch_gradients();
            int batch_correct = 0;
            float batch_loss = 0.0f;
            
            // Calculate actual batch size (might be smaller for last batch)
            int current_batch_size = (i + BATCH_SIZE <= NUM_TRAINING_IMAGES) ? 
                                    BATCH_SIZE : (NUM_TRAINING_IMAGES - i);
            
            // Process each example in the batch
            for (int j = 0; j < current_batch_size; j++) {
                float hidden_layer[HIDDEN_SIZE];
                float output_layer[OUTPUT_SIZE];
                float target[OUTPUT_SIZE];
                
                // Prepare target and forward pass
                create_target(training_labels[i + j], target);
                forward_pass(training_images[i + j], hidden_layer, output_layer);
                
                // Calculate loss and check prediction
                batch_loss += calculate_loss(output_layer, target);
                if (argmax(output_layer, OUTPUT_SIZE) == training_labels[i + j]) {
                    batch_correct++;
                }
                
                // Backward pass: We take the error from the output layer and propagate it back through the network.
                backward_pass(training_images[i + j], hidden_layer, output_layer, target);
            }
            
            // Update weights after processing the batch
            update_weights(learning_rate, current_batch_size);
            
            // Update metrics
            correct_predictions += batch_correct;
            epoch_loss += batch_loss;
            
            // Print batch progress
            if ((i / BATCH_SIZE) % 100 == 0) {
                printf("Batch %d/%d: Loss=%.4f, Accuracy=%.2f%%\n", 
                       i / BATCH_SIZE, 
                       NUM_TRAINING_IMAGES / BATCH_SIZE,
                       batch_loss / current_batch_size,
                       100.0f * batch_correct / current_batch_size);
            }
        }
        
        // Calculate epoch metrics
        float train_accuracy = 100.0f * correct_predictions / NUM_TRAINING_IMAGES;
        float avg_loss = epoch_loss / NUM_TRAINING_IMAGES;
        
        // Evaluate on test set
        float test_accuracy = evaluate(test_images, test_labels, NUM_TEST_IMAGES);
        
        printf("\nEpoch %d results:\n", epoch + 1);
        printf("Training Loss: %.4f\n", avg_loss);
        printf("Training Accuracy: %.2f%%\n", train_accuracy);
        printf("Test Accuracy: %.2f%%\n", test_accuracy);
        
        // Save best model if the test accuracy is better than the previous best accuracy.
        // for the very first epoch, it would always be the best as best_accuracy is initialized to 0.
        if (test_accuracy > best_accuracy) {
            best_accuracy = test_accuracy;
            save_weights();
            printf("New best model saved!\n");
        }
        
        // Learning rate decay
        learning_rate *= 0.95f;
        printf("Learning rate adjusted to: %.4f\n", learning_rate);
    }
    
    printf("\nTraining complete!\n");
    printf("Best test accuracy achieved: %.2f%%\n", best_accuracy);
    
    return 0;
}