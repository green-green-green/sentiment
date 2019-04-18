import Foundation
import CreateML

let jsonUrl = Bundle.main.url(forResource: "negaposiDatasetStr", withExtension: "json")
let data = try MLDataTable(contentsOf: jsonUrl!)
print(data)
let (trainingData, testingData) = data.randomSplit(by: 0.8, seed: 5)
let sentimentClassifier = try MLTextClassifier(trainingData: trainingData,textColumn: "text",labelColumn: "label")

let trainingAccuracy = (1.0 - sentimentClassifier.trainingMetrics.classificationError) * 100

let validationAccuracy = (1.0 - sentimentClassifier.validationMetrics.classificationError) * 100

let evaluationMetrics = sentimentClassifier.evaluation(on: testingData)

let evaluationAccuracy = (1.0 - evaluationMetrics.classificationError) * 100

let metadata = MLModelMetadata(author: "John Appleseed", shortDescription: "A model trained to classify movie review sentiment", version: "1.0")





//let saveUrl = Bundle.main.url(forResource: "negaposiModel", withExtension: "mlmodel")



try sentimentClassifier.write(to: URL(fileURLWithPath: "negaposiModel.mlmodel"),
                              
                              metadata: metadata)

