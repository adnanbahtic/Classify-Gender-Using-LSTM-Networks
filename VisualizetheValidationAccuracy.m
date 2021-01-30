[prediction,probabilities] = classify(net,featuresValidation);
figure
cm = confusionchart(categorical(labelsValidation),prediction,'title','Validation Set Accuracy');
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';