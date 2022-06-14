include { HyperParameterSearch }            from './modules/hyperparameter_search'
include { ModelTraining }                   from './modules/train_conv_net'
include { ModelTestEval }                   from './modules/test_conv_net'
include { ChaosTrain }                      from './workflows/training_with_chaos'
include { ChaosTestEval }                   from './workflows/testing_with_chaos'
include { Main }                            from './workflows/extract_transfor_train_test'

if (params.type != "Complete") {
    Channel.fromPath(
        params.data,
        type: "dir"
        )
        .set{ data }
}

workflow {
    switch (params.type) {
        case "GetBestParams":
            HyperParameterSearch(data)
            break;
        case "TrainOnly":
            ModelTraining(data)
	        break;
        case "TestOnly":
            ModelTestEval(data.collect())
	        break;
        case "ExtractTrain":
            ChaosTrain()
	        break;
        case "ExtractTest":
            ChaosTestEval()
	        break;
        case "TrainTest":
            ModelTraining(data)
            ModelTestEval(ModelTraining.out.collect())
	        break;
        case "Complete":
            Main()
	        break;
    }
}
