# Image Classification
## Description of the problem
Classify images into the five groups: Cosmos, Paintings, Food, Interiors, People.
## Solution authors
Sebastian Bobrowski (s17603@pjwstk.edu.pl), Katarzyna Czerwińska (s17098@pjwstk.edu.pl)
## Instructions for use
### Requirements
#### Runtime
.NET Runtime 5.0.0
#### Library on non-Windows operating systems (tested on macOS)
https://www.mono-project.com/docs/gui/libgdiplus/
### .NET CLI
#### Train the model
Execute the "dotnet run --project dotnet run --project ImageClassification.Train/ImageClassification.Train.csproj" command in the solution directory.
#### Consume the model - Make a prediction
Execute the "dotnet run --project ImageClassification.Predict/ImageClassification.Predict.csproj" command in the solution directory.
### Rider / Visual Studio
#### Train the model / Consume the model
Select the appropriate project to run and click the Run button
## References
### Image sets
#### Cosmos
https://nasa.tumblr.com/
#### Paintings
https://sollertias.tumblr.com/
#### Food
https://cravefoodie.com/
#### Interiors
https://vintagehomecollection.tumblr.com/
#### People
https://gucci.tumblr.com/
### Code base
https://github.com/dotnet/machinelearning-samples/tree/master/samples/csharp/getting-started/DeepLearning_ImageClassification_Training