## Model Hyperparameter Tuning with SageMaker & TensorFlow
This project covers model hyperparameter tuning (HPT) across a number of different deep learning problems/datasets using TensorFlow and Amazon SageMaker. Hyperparameter tuning works by finding the best version of a model by running many training jobs on your dataset using the algorithm and ranges of hyperparameters that you specify. It then chooses the hyperparameter values that result in a model that performs the best, as measured by a metric that you choose.

A `conda_tensorflow2_p36` kernel was used with the Amazon SageMaker notebook instance.

## Additional Resources
- [Amazon SageMaker HPT Overview](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html)
- [Amazon SageMaker HPT Examples](https://github.com/aws/amazon-sagemaker-examples/tree/master/hyperparameter_tuning/tensorflow2_mnist)
- [Amazon SageMaker Automatic Model Tuning](https://www.youtube.com/watch?v=c_jv3llDKU0)
- [SageMaker Model Data Issue](https://stackoverflow.com/questions/48310237/sagemaker-could-not-find-model-data-when-trying-to-deploy-my-model)