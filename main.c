#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

float w1 = 0;
float w2 = 0;

float g1 = 0;
float g2 = 0;

//This is the function we want to approximate using a neural network
// The actual weights of the function could change but our model 
// would be able to approximate it after sufficient training
float black_box(float x1, float x2) {
    
    return (2*x1 + 5*x2 );
}

// Initialize the weights to random values
void init_weights() {
    w1 =  rand() % 100 / 100.0;
    w2 =  rand() % 100 / 100.0;
}

// Adjust the weights based on the gradient and learning rate
void adjust_weights(float learning_rate) {
    w1 += g1*learning_rate;
    w2 += g2*learning_rate;
}

// Calculate the output of the model
float model(int x1, int x2) {
    return w1*x1 + w2*x2 ;
}

// Calculate the gradient of the error with respect to the weights
void calculate_gradient(float x1, float x2, float error) {
    // Gradient is the derivative of the error with respect to each weight
    // E = (target - y)
    // g1 calculation: how much the error changes with respect to w1
    // g2 calculation: how much the error changes with respect to w2

    // dE/dw1 =  dE/dy * dy/dw1
    // Where dE/dy is how much error changes with respect to the output 
    // and dy/dw1 is the derivative of the model with respect to w1 which is x1
    // dE/dw1 =  2*E * x1
    g1 = 2*error * x1;
    g2 = 2*error * x2;
}

int main() {
    srand(time(NULL));
    init_weights();
    float learning_rate = 0.0001;  // Smaller learning rate due to squared error
    float total_error = 0;
    float x1, x2;

    for (int epoch = 0; epoch < 10000; epoch++) {
        total_error = 0;
        
        x1 = rand() % 100;
        x2 = rand() % 100;
        
        float target = black_box(x1, x2);
        float output = model(x1, x2);
        float error = target - output;
        
        // Accumulate error
        total_error += error  * error; // We square the error for calculation of mean squared error
        
        // Calculate gradients for this sample
        calculate_gradient(x1, x2, error);
        adjust_weights(learning_rate);
        
        // Print average error every 100 epochs
        if (epoch % 100 == 0) {
            printf("Epoch %d, Average MSE: %f\n", epoch, total_error);
        }
        
        // Early stopping condition
        if (total_error  < 0.0001) {
            printf("Converged at epoch %d\n", epoch);
            break;
        }
    }

    printf("Final weights - w1: %f, w2: %f\n", w1, w2);
    // Test the model
    printf("\nTesting the model:\n");
    for (int i = 0; i < 5; i++) {
        x1 = rand() % 100;
        x2 = rand() % 100;
        float target = black_box(x1, x2);
        float output = model(x1, x2);
        printf("Input: (%f, %f), Target: %f, Prediction: %f\n", 
               x1, x2, target, output);
    }
    
    return 0;
}